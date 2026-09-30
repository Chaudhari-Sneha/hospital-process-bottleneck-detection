"""
STEP 10 - THE EVIDENCE EXTRACTOR. Turn ANY event log into structured findings.

WHY THIS EXISTS

Step 6 (`14_evidence_package.py`) produces 22 findings, and every one of them was
written by hand about THIS project's two logs. That was the right thing to build -
those findings carry parameter dependence, robustness status and allowed wording
that only a human who did the analysis can assign. But it means the chain

    event log -> analysis -> FINDINGS -> payload -> the model writes prose

has a hand-made link in the middle. Point the system at a new hospital and that
link is empty: there are no findings, so there is nothing for the narration layer
to narrate and nothing for a diagnosis layer to reason over.

This module fills that link generically. It runs the deterministic analysis on any
log and emits findings in EXACTLY the schema `15_narrate_findings.py` already
consumes, so the existing hallucination guard applies unchanged.

THE DIVISION OF LABOUR (this is the project's central design rule)

    Python/pm4py  ->  computes every number
    this module   ->  states them as findings, with their caveats
    the LLM       ->  interprets, hypothesises causes, recommends - and computes NOTHING

A finding here is a measured fact plus the conditions under which it may be
repeated. The model receives the fact and the conditions; it never receives the
log.

THE PART WORTH UNDERSTANDING: THE GATE GENERATES THE FORBIDDEN LIST

`18_process_profile.py` reports what a log can SUPPORT, not just what it contains.
This module turns each missing capability into a FORBIDDEN CLAIM rather than
silently omitting it:

  - Sepsis has `lifecycle:transition = complete` only, so queueing and processing
    cannot be separated -> "the delay is queueing" becomes forbidden, not absent.
  - Hospital Billing's resource column names individuals and is 45% empty, so the
    internal-vs-handoff split is meaningless -> that claim is forbidden too.

This is the same mechanism that stopped `03` PRINTING a meaningless "100% handoff"
now stopping the MODEL from claiming it. A capability the log lacks becomes an
explicit prohibition carried all the way to the prose layer, instead of a gap
somebody fills in with a plausible sentence.

USAGE

    python src/20_extract_findings.py                         # the Sepsis log
    python src/20_extract_findings.py --log "<any log>" --prefix myhospital

Writes `outputs/<prefix>_findings.csv` and `outputs/<prefix>_findings.json`.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

# How much of the total waiting the top edges must cover before we call it
# "concentrated". Declared here so it can be argued with.
CONCENTRATION_TOP_N = 5
REWORK_NOTABLE = 0.05          # >=5% of transitions being self-loops is worth saying
MIN_FIRES = 20                 # an edge must fire this often to be a pattern
# How many SECONDARY bottlenecks to write out as findings. Purely a presentation
# limit - every step that clears the statistical bar is kept in the `materiality`
# block of the JSON, and the last finding says how many were not listed. An
# 18-activity process produced 9; an administrator can act on a handful.
MAX_SECONDARY_FINDINGS = 5
# Plain-language name for each signature, so the finding states the KIND of
# bottleneck rather than leaving the model to infer it from the numbers.
SIGNATURE_WORDS = {
    "batched": "its median is at or near zero, so work is held and released "
               "together rather than flowing continuously",
    "rare_slow_path": "it touches only a small share of cases, so it is an "
                      "exception route rather than the main path",
    "rework": "it is a step immediately repeating itself",
    "handoff": "it crosses from one department to another",
    "constraint": "it is slow within a single department, on the main path",
}
# Thresholds for classifying WHAT KIND of bottleneck a step is. Declared here so
# they can be argued with, like every other threshold in this project.
BATCH_MEDIAN_H = 0.1        # a median at/below 6 minutes reads as batch release
RARE_PATH_CASE_SHARE = 0.10  # a step touching <=10% of cases is an exception route
EDGE_SEP = "  ->  "          # exactly how src/03 builds its edge keys


def load_module(filename: str, name: str):
    """Import a numbered script by path (a module name cannot start with a digit)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fmt_h(hours: float) -> str:
    """Hours, days or minutes - whichever a hospital administrator would say."""
    # Thousands separators matter here: these strings are read by a person and
    # then by a language model. "5409984.5 days" is a number nobody can parse at
    # a glance, and an unreadable figure is one that gets misquoted.
    if hours >= 48:
        return f"{hours / 24:,.1f} days"
    if hours < 1:
        return f"{hours * 60:,.0f} min"
    return f"{hours:,.1f} h"


def finding(fid: str, category: str, text: str, *, kind: str = "measured",
            source: str = "", dependence: str = "none",
            robustness: str = "single log, as measured",
            wording: str = "", caveat: str = "", patterns: list | None = None) -> dict:
    """
    One finding, in the schema `15_narrate_findings.py` already reads.

    `allowed_wording` and `caveat` are not decoration - they are the contract the
    narration guard enforces. A finding without a caveat is a claim without a
    limit, and that is how a hedged result turns into a headline.
    """
    return {
        "id": fid,
        "category": category,
        "finding": text,
        # Which computation produced this number. The diagnosis layer cites these
        # so a recommendation can be traced back to the code that measured it.
        "evidence_source": source,
        "kind": kind,
        "parameter_dependence": dependence,
        "robustness": robustness,
        "allowed_wording": wording or text,
        "caveat": caveat,
        # JSON-encoded so the list survives the CSV round trip. The narration
        # guard reads this column to enforce the prohibition mechanically; without
        # it the gate states a limit that nothing checks.
        "patterns": json.dumps(patterns or []),
    }


def bottleneck_materiality(gaps, edge_col: str = "edge",
                           value_col: str = "gap_hours",
                           n_permutations: int = 400,
                           seed: int = 20260923,
                           n_cases: int = 0) -> dict:
    """
    Does the top edge absorb MORE time than chance allocation would give it?

    Why this exists: ranking is not significance testing. The pipeline reports
    the relative worst step, and on any log some step is worst - so it named a
    bottleneck on all three no-defect control processes in Step 22. That was the
    single clearest defect ground-truth validation exposed.

    The obvious fix - "flag it if the top edge is 3x the median" - is a number
    chosen to make our own controls pass, which is fitting to the generator we
    also wrote. So instead, a PERMUTATION TEST, where the threshold comes from
    the data:

      H0: waiting time is EXCHANGEABLE across transitions. No edge is special;
          an edge with many transitions simply accumulates more of it.

      Shuffle which transition each observed gap belongs to, keeping every edge's
      COUNT fixed, and record the largest edge total. Repeat. That is the
      distribution of "how big would the top edge look by luck alone", and it
      already accounts for some edges firing far more often than others.

      p = fraction of shuffles whose top total reaches the observed one.

    On a healthy process gaps really are exchangeable, so the observed top sits
    inside the null and p is large. On a defective one the injected delay piles
    onto one edge and p collapses. Nothing here was tuned by hand.

    This does NOT establish that the bottleneck is fixable, costly, or clinically
    important - only that its concentration is more than chance.
    """
    import numpy as np

    values = gaps[value_col].to_numpy(dtype=float)
    labels, uniques = pd.factorize(gaps[edge_col])
    n_edges = len(uniques)
    if len(values) < 30 or n_edges < 3:
        return {"testable": False, "reason": "too few transitions or edges"}

    # Per-edge median and whether the edge crosses departments - both needed to
    # classify the signature below, both derived from the same frame.
    grouped = gaps.groupby(edge_col)[value_col]
    med_map = grouped.median()
    medians = np.array([med_map.get(u, np.nan) for u in uniques], dtype=float)
    handoff_edges = set()
    if "is_handoff" in gaps.columns:
        handoff_edges = set(
            gaps.loc[gaps["is_handoff"].astype(bool), edge_col].unique())

    counts = np.bincount(labels, minlength=n_edges)
    eligible = counts >= MIN_FIRES
    if not eligible.any():
        return {"testable": False, "reason": "no edge fires often enough"}

    # THE STATISTIC IS DISPROPORTION, NOT TOTAL - and that is a correction.
    #
    # The first version compared the largest edge TOTAL against the largest total
    # under permutation. That looked sound and was nearly meaningless: the biggest
    # total goes to the highest-COUNT edge on both sides, so the test compared a
    # frequent edge with itself and returned ~1.0 by construction.
    #
    # It showed up on the real Sepsis log, which it declared free of any
    # bottleneck (p = 0.77). Inspection explained why: `CRP -> Leucocytes` holds
    # 12.3% of the waiting but 10.4% of the transitions - a ratio of 1.18. It is
    # the largest absorber because it happens constantly, not because it is slow
    # (its median is zero: results are entered in batches). Meanwhile
    # `Admission NC -> Release A` carries 6.2% of the waiting on 0.8% of the
    # transitions, a ratio of 7.4, and the old statistic never looked at it.
    #
    # So the question is not "is the biggest total bigger than chance?" but "does
    # ANY edge absorb more than its share of transitions?" - which is what an
    # administrator means by a bottleneck.
    expected_per_transition = values.sum() / len(values)
    totals = np.bincount(labels, weights=values, minlength=n_edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = totals / (counts * expected_per_transition)
    observed = float(np.nanmax(ratios[eligible]))
    worst_idx = int(np.flatnonzero(eligible)[np.nanargmax(ratios[eligible])])

    rng = np.random.default_rng(seed)
    shuffled = values.copy()
    null_max = np.empty(n_permutations)
    for i in range(n_permutations):
        rng.shuffle(shuffled)
        # labels stay put, durations move: each edge keeps its COUNT while the
        # durations that landed on it are randomised.
        t = np.bincount(labels, weights=shuffled, minlength=n_edges)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = t / (counts * expected_per_transition)
        null_max[i] = np.nanmax(r[eligible])

    # +1 on both sides: the observed arrangement is itself one of the possible
    # ones, so p is never claimed to be exactly zero from a finite number of draws.
    p = float((np.sum(null_max >= observed) + 1) / (n_permutations + 1))

    # EVERY eligible edge is now tested, not just the winner.
    #
    # The previous version computed `ratios` for all edges and then kept only the
    # argmax, so a process with two real bottlenecks reported one and discarded
    # the other however large it was. On a 250-case test log the second step held
    # 1,449 h against the first's 1,922 h - 75% of it - and was invisible. That
    # was not a ranking, threshold or materiality failure: the second edge was
    # correctly ranked second and simply never looked at.
    #
    # Comparing every edge against the SAME null-of-the-maximum is the
    # Westfall-Young max-T procedure, and it controls the family-wise error rate:
    #
    #     under H0, P(any edge exceeds the 95th percentile of the max-null)
    #             = P(max exceeds its own 95th percentile) = 0.05
    #
    # So this reports more bottlenecks at EXACTLY the false-positive rate the
    # single-edge test already carried. Nothing was loosened to achieve it - the
    # information was computed all along and thrown away.
    cutoff = float(np.quantile(null_max, 1 - 0.05))
    material_edges = []
    for idx in np.flatnonzero(eligible):
        ratio = float(ratios[idx])
        if not np.isfinite(ratio) or ratio < cutoff:
            continue
        # CLASSIFY THE SIGNATURE HERE, NOT IN THE PROMPT.
        #
        # The diagnosis layer was being handed a catalogue of mechanisms plus the
        # instruction "check the signature holds before using an entry" - and it
        # got it wrong, calling a step with 1,445 occurrences a "rare slow path"
        # and applying a batching lever to a step whose median is high.
        #
        # That was a design error, not a model limitation. "Is this median near
        # zero?" and "does this edge cross departments?" are ARITHMETIC, and this
        # project's central rule is that arithmetic happens in Python and the
        # model only interprets. Asking the model to evaluate a numeric condition
        # put a computation back in the prompt, which is exactly what the whole
        # architecture exists to prevent.
        #
        # So the signature is decided here from numbers already computed, and the
        # model receives a mechanism that is already determined. It chooses the
        # lever and writes the reasoning - which is interpretation, its actual job.
        edge_label = str(uniques[idx])
        med = float(medians[idx]) if medians is not None else float("nan")
        share_of_cases = counts[idx] / max(n_cases, 1)
        src, _, dst = edge_label.partition(EDGE_SEP)

        if src == dst:
            signature = "rework"
        elif np.isfinite(med) and med <= BATCH_MEDIAN_H:
            # Most instances record no elapsed time; a minority carry it all.
            signature = "batched"
        elif share_of_cases <= RARE_PATH_CASE_SHARE:
            signature = "rare_slow_path"
        elif edge_label in handoff_edges:
            signature = "handoff"
        else:
            signature = "constraint"

        material_edges.append({
            "edge": edge_label,
            "disproportion_ratio": round(ratio, 2),
            "total_hours": round(float(totals[idx]), 2),
            "occurrences": int(counts[idx]),
            "median_hours": round(med, 3) if np.isfinite(med) else None,
            "share_of_cases": round(float(share_of_cases), 4),
            # Determined from the numbers above, so the model never has to.
            "signature": signature,
            # Per-edge p against the max-null: conservative by construction,
            # since the max is stochastically larger than any single edge.
            "p_value": round(float((np.sum(null_max >= ratio) + 1)
                                   / (n_permutations + 1)), 4),
        })
    material_edges.sort(key=lambda m: -m["disproportion_ratio"])

    return {"testable": True,
            "most_disproportionate_edge": str(uniques[worst_idx]),
            "disproportion_ratio": round(observed, 2),
            "null_median_ratio": round(float(np.median(null_max)), 2),
            "fwer_cutoff_ratio": round(cutoff, 2),
            "material_edges": material_edges,
            "n_material": len(material_edges),
            "p_value": round(p, 4), "n_permutations": n_permutations,
            "material": p < 0.05}


def extract(log_path: Path, prefix: str) -> dict:
    """Run the deterministic analysis and reduce it to findings."""
    profiler = load_module("18_process_profile.py", "profiler")
    handoffs = load_module("03_department_handoffs.py", "handoffs")

    # --- load ONCE, through the profiler --------------------------------------
    # Deliberately not `handoffs.load_events()`: that keeps only four columns and
    # drops `lifecycle:transition`, so capability detection run on its output
    # reports "no lifecycle information" even for a log that has it. The profiler's
    # loader keeps every column and sorts identically (stable, by case then time),
    # so the downstream numbers are unchanged.
    events = profiler.load_log(log_path, handoffs.CASE_ID, handoffs.ACTIVITY,
                               handoffs.TIMESTAMP)

    resource_col = next((c for c in profiler.DEF_RESOURCE_CANDIDATES
                         if c in events.columns), None)
    # A LOG WITHOUT A DEPARTMENT COLUMN IS NOT AN ERROR - it is a weaker log.
    #
    # This used to `raise SystemExit`, which made the `X-NORESOURCE` prohibition
    # further down unreachable: the code that says "must not claim anything about
    # which department is responsible" could never run, because the analysis died
    # first. Plenty of real exports carry only case/activity/timestamp, and the
    # right answer for those is the same as everywhere else in this project -
    # do the analysis the log supports, and forbid the claims it does not.
    #
    # A constant placeholder column is added purely so `03`'s add_gaps() can build
    # its dept_pair column without special-casing. Nothing downstream reports on
    # it: `resource_col` stays None, so capabilities, roles and every department
    # finding are skipped, and X-NORESOURCE is issued.
    if resource_col is None:
        print("  NOTE: no department/resource column found. Activity-level "
              "analysis only;\n        department findings will be omitted and "
              "the claim forbidden.")
        events = events.copy()
        events["__no_resource__"] = "(none recorded)"
        handoffs.GROUP = "__no_resource__"
    else:
        # `03`'s functions read this global to build department pairs.
        handoffs.GROUP = resource_col

    caps = profiler.detect_capabilities(events, "lifecycle:transition",
                                        handoffs.CASE_ID)
    ep = profiler.detect_episode_boundaries(
        events, handoffs.CASE_ID, handoffs.ACTIVITY, handoffs.TIMESTAMP)
    rework = profiler.detect_rework(events, handoffs.CASE_ID, handoffs.ACTIVITY)
    roles = profiler.cluster_departments_by_behaviour(
        events, handoffs.ACTIVITY, resource_col)

    with_episodes = handoffs.add_episodes(events)
    gaps, _ = handoffs.add_gaps(with_episodes)
    edges = handoffs.summarise(gaps, "edge")
    depts = handoffs.summarise(gaps, "dept_pair")

    total_h = float(gaps["gap_hours"].sum())
    n_cases = int(events[handoffs.CASE_ID].nunique())

    found: list[dict] = []
    blocked: list[dict] = []
    # Defined up front: the ranked block below may be skipped entirely on
    # a log with no edge firing MIN_FIRES times, and the artifact still
    # needs to say so rather than omit the key.
    mat: dict = {"testable": False, "reason": "no eligible edges"}

    # --- 1. process structure -------------------------------------------------
    found.append(finding(
        "S1", "1. Process structure",
        f"The log holds {len(events):,} events across {n_cases:,} cases and "
        f"{events[handoffs.ACTIVITY].nunique()} distinct activities.",
        source="18_process_profile.load_log",
            caveat="Descriptive only - says nothing about whether the process is good."))

    if ep["episode_end_activities"]:
        # `add_episodes` numbers episodes WITHIN each case, so the episode count
        # is the number of distinct (case, episode) pairs - not nunique() of a
        # single column, which would collapse episode 0 of every patient into one.
        n_eps = int(len(with_episodes[[handoffs.CASE_ID, "episode"]]
                        .drop_duplicates()))
        # Extra episodes and repeating cases are NOT the same number - one case can
        # re-enter several times. Count the cases separately rather than implying
        # that 296 extra episodes means 296 different patients came back.
        eps_per_case = with_episodes.groupby(handoffs.CASE_ID)["episode"].nunique()
        n_repeat_cases = int((eps_per_case > 1).sum())
        found.append(finding(
            "S2", "1. Process structure",
            f"{n_repeat_cases:,} of {n_cases:,} cases re-enter the process after "
            f"ending it, producing {n_eps - n_cases:,} additional episodes, so the "
            f"log contains {n_eps:,} episodes of care in total.",
            wording=f"{n_repeat_cases:,} cases re-entered the process",
            source="03_department_handoffs.add_episodes",
            caveat="Time between episodes is time OUTSIDE the process and is "
                   "excluded from every waiting figure below."))
    else:
        found.append(finding(
            "S3", "1. Process structure",
            "No episode boundary was detected: each case is one continuous "
            "episode, with no evidence of cases ending and restarting.",
            source="18_process_profile.detect_episode_boundaries",
            caveat="Correct for a process participants do not leave and re-enter. "
                   "Verify that is true of this process before relying on it."))

    if roles.get("available"):
        found.append(finding(
            "S4", "1. Process structure",
            f"{roles['n_departments']} departments perform only "
            f"{roles['n_roles']} distinct jobs, judged by which activities each "
            f"one actually carries out.",
            wording=f"the {roles['n_departments']} departments reduce to "
                    f"{roles['n_roles']} functional roles",
            source="18_process_profile.cluster_departments_by_behaviour",
            caveat="Roles are grouped by behaviour, not by name. Two departments "
                   "with the same signature may still be organisationally "
                   "separate."))

    # --- 2. where the time is lost -------------------------------------------
    # `summarise` already returns rows sorted by total_hours descending.
    ranked = edges[edges["count"] >= MIN_FIRES]
    if len(ranked):
        top = ranked.iloc[0]
        share = top["total_hours"] / total_h * 100 if total_h else 0
        mat = bottleneck_materiality(gaps, n_cases=n_cases)
        found.append(finding(
            "T1", "2. Where time is lost",
            f"The single largest absorber of time is '{top.name}': "
            f"{fmt_h(top['total_hours'])} in total across {int(top['count']):,} "
            f"occurrences ({share:.1f}% of all measured waiting), median "
            f"{fmt_h(top['median_h'])}.",
            wording=f"'{top.name}' absorbs the most total time "
                    f"({share:.1f}% of measured waiting)",
            source="03_department_handoffs.summarise(edge)",
            # The caveat has to carry the materiality result, not just sit next to
            # it. Holdout of the first fix showed why: on a process with NO defect,
            # the model read T1 and wrote "time is lost in the transition from X to
            # Y" - avoiding the word "bottleneck" that X-NOBOTTLENECK blocks, while
            # making exactly the claim it was meant to prevent. Policing vocabulary
            # does not police meaning; the evidence itself has to say what it is.
            caveat=("Ranked by TOTAL time absorbed, not by median. A slow step that "
                    "happens rarely is a smaller problem than a quick one that "
                    "happens constantly."
                    + ("" if mat.get("material", True) else
                       " IMPORTANT: this step is NOT absorbing more than its share "
                       "of transitions would give it (see T3). It is the busiest "
                       "step, not a problem. Some step is always busiest. Do not "
                       "present it as where time is being lost."))))

        if mat.get("testable"):
            if mat["material"]:
                found.append(finding(
                    "T3", "2. Where time is lost",
                    f"The step most out of proportion to how often it happens is "
                    f"'{mat['most_disproportionate_edge']}': it absorbs "
                    f"{mat['disproportion_ratio']}x the time its share of "
                    f"transitions would give it (permutation test, "
                    f"p = {mat['p_value']}). Classified as "
                    f"{mat['material_edges'][0]['signature'].replace('_', ' ')} "
                    f"because {SIGNATURE_WORDS[mat['material_edges'][0]['signature']]}.",
                    source="20_extract_findings.bottleneck_materiality",
                    robustness=f"{mat['n_permutations']} permutations",
                    wording=f"'{mat['most_disproportionate_edge']}' is slow "
                            f"relative to how often it occurs",
                    caveat="This is a DIFFERENT question from the largest absorber "
                           "of time. A step can dominate the total simply by "
                           "happening constantly. Says the disproportion is real, "
                           "not that it is fixable, costly or clinically "
                           "important."))

                # SECONDARY bottlenecks - T4, T5, ... one per additional step
                # that clears the same family-wise cutoff. Emitted as separate
                # findings so the diagnosis layer can cite them individually;
                # a fix for the second bottleneck needs its own evidence id.
                secondaries = mat["material_edges"][1:]
                for n, extra in enumerate(
                        secondaries[:MAX_SECONDARY_FINDINGS], start=4):
                    found.append(finding(
                        f"T{n}", "2. Where time is lost",
                        f"A further step is also out of proportion: "
                        f"'{extra['edge']}' absorbs "
                        f"{extra['disproportion_ratio']}x the time its share of "
                        f"transitions would give it "
                        f"({fmt_h(extra['total_hours'])} across "
                        f"{extra['occurrences']:,} occurrences, p = "
                        f"{extra['p_value']}). Classified as "
                        f"{extra['signature'].replace('_', ' ')} because "
                        f"{SIGNATURE_WORDS[extra['signature']]}.",
                        source="20_extract_findings.bottleneck_materiality",
                        robustness=f"{mat['n_permutations']} permutations, "
                                   "family-wise error controlled",
                        wording=f"'{extra['edge']}' is also slow relative to how "
                                f"often it occurs",
                        caveat="A SECONDARY bottleneck, held to the same evidence "
                               "bar as the primary - every step was tested against "
                               "the same threshold, so this is not a weaker claim. "
                               "Fixing the largest step will not address this one."))

                hidden = len(secondaries) - MAX_SECONDARY_FINDINGS
                if hidden > 0:
                    found.append(finding(
                        "T-MORE", "2. Where time is lost",
                        f"{hidden} further step(s) also cleared the same bar and "
                        f"are listed in the `materiality` block of this file "
                        f"rather than as findings.",
                        source="20_extract_findings.bottleneck_materiality",
                        wording=f"{hidden} further steps also qualified",
                        caveat="Omitted for readability, not for weakness of "
                               "evidence. They met the identical threshold."))
            else:
                found.append(finding(
                    "T3", "2. Where time is lost",
                    f"No step is out of proportion to how often it happens: the "
                    f"most extreme absorbs {mat['disproportion_ratio']}x its share "
                    f"of transitions, which random allocation reproduces "
                    f"(permutation test, p = {mat['p_value']}). This process has a "
                    f"busiest step, but not a bottleneck.",
                    source="20_extract_findings.bottleneck_materiality",
                    robustness=f"{mat['n_permutations']} permutations",
                    wording="no step stands out beyond chance",
                    caveat="Absence of a dominant bottleneck is not evidence the "
                           "process is efficient - only that the waiting is "
                           "spread rather than concentrated."))
                blocked.append({
                    "id": "X-NOBOTTLENECK",
                    "must_not_claim": "that this process has a bottleneck, or that "
                                      "any particular step is the problem",
                    "because": f"a permutation test (p = {mat['p_value']}) cannot "
                               "distinguish the busiest step from ordinary "
                               "variation. Some step is always busiest; that alone "
                               "is not a finding.",
                    "patterns": [r"\bthe bottleneck\b", r"\bis the bottleneck\b",
                                 r"\bmain bottleneck\b", r"\bkey bottleneck\b",
                                 r"\bprimary constraint\b"],
                })

        head = ranked.head(CONCENTRATION_TOP_N)
        conc = head["total_hours"].sum() / total_h * 100 if total_h else 0
        found.append(finding(
            "T2", "2. Where time is lost",
            # State the BASELINE alongside the number. Without it this finding is
            # close to a tautology on a small process: with 6 distinct steps, any
            # 5 of them hold ~83% of anything. A control process was described as
            # "time is lost in the top 5 steps, which account for 92.2%" - true,
            # unremarkable, and read as a finding. A share means nothing until you
            # say what share would be unremarkable.
            f"The top {len(head)} of {len(ranked)} steps account for {conc:.1f}% "
            f"of all measured waiting time. If waiting were spread evenly across "
            f"steps, {len(head)} of {len(ranked)} would account for "
            f"{len(head) / len(ranked) * 100:.1f}%.",
            wording=f"the top {len(head)} steps hold {conc:.1f}% of the waiting "
                    f"(even spread would give {len(head) / len(ranked) * 100:.1f}%)",
            source="03_department_handoffs.summarise(edge)",
            caveat="Compare the two figures before calling this concentration. "
                   "Concentration is a reason to target those steps first; it is "
                   "not evidence that they can be shortened."))

    # --- 3. rework -----------------------------------------------------------
    if rework["self_loop_share_of_transitions"] >= REWORK_NOTABLE:
        pct = rework["self_loop_share_of_transitions"] * 100
        worst = rework["top_self_looping_activities"][:1]
        detail = (f", most often '{worst[0]['activity']}' "
                  f"({worst[0]['self_loops']:,} times)") if worst else ""
        found.append(finding(
            "R1", "3. Rework",
            f"{pct:.1f}% of all steps are immediately repeated{detail}.",
            wording=f"{pct:.1f}% of steps are immediate repeats",
            source="18_process_profile.detect_rework",
            caveat="A repeat is not automatically waste - some are clinically or "
                   "procedurally required. It marks where to look, not a fault."))

    # --- 4. data quality, stated as findings in their own right ---------------
    found.append(finding(
        "Q1", "4. Data quality",
        f"Lifecycle information present: {caps['lifecycle_values'] or 'none'}.",
        source="18_process_profile.detect_capabilities",
            caveat="Determines whether waiting can be separated from work."))

    if resource_col and caps.get("resource_missing_share", 0) > 0:
        pct = caps["resource_missing_share"] * 100
        found.append(finding(
            "Q2", "4. Data quality",
            f"{pct:.1f}% of events carry no value in '{resource_col}'.",
            wording=f"{pct:.1f}% of events are unattributed",
            source="18_process_profile.detect_capabilities",
            caveat="Any per-department total covers only the portion of the log "
                   "that is filled in."))

    # --- 5. what this log CANNOT support -> forbidden claims ------------------
    # Generated from the capability profile, not typed. This is the gate.
    if not caps["can_separate_wait_from_work"]:
        blocked.append({
            "id": "X-LIFECYCLE",
            # NOT r"\bwaiting time\b": elapsed time between events is precisely
            # what this log DOES support, and the findings themselves say
            # "measured waiting time". Blocking it rejected true statements. The
            # violation is claiming the SPLIT, or naming queueing/processing as
            # the component - not using the word "waiting".
            # The last four patterns catch the claim in its PRACTICAL form.
            #
            # Without start timestamps you cannot tell whether a delay is people
            # waiting for a free resource or the work itself taking that long -
            # so you cannot know whether more staff would help. A model duly
            # avoided the words "queueing" and "processing" and then recommended
            # "adjust staffing ... increasing capacity can shrink its duration",
            # which IS that claim, made as an action instead of an assertion.
            #
            # Forbidding the vocabulary while permitting the recommendation it
            # implies is not a prohibition, it is a wish. Streamlining, removing
            # steps or measuring are unaffected - only ADDING capacity as the fix.
            # NOT a bare r"\bprocessing time\b", and that is the third time this
            # exact mistake has been made in this project. It rejected:
            #
            #   "automate discharge paperwork ... to reduce manual PROCESSING TIME
            #    within the Discharge Processing department"
            #
            # which is ordinary English for "time staff spend on paperwork", not a
            # claim about decomposing elapsed time - and the recommendation itself
            # (remove manual work) is one the catalogue explicitly permits. The
            # phrase also collides with any activity named "... Processing".
            #
            # Previously: r"rs " matched inside "hou[rs ]p90"; r"\bwaiting time\b"
            # rejected "measured waiting time" quoted from the findings. The
            # lesson each time is the same - a guard that rejects true statements
            # gets switched off, and then nothing is checked at all.
            #
            # The real violation is claiming the SPLIT, which needs both halves
            # named in contrast. "queueing" alone still catches the common form.
            "patterns": [r"\bqueue?ing\b", r"\bqueued\b", r"\bqueue\b",
                         r"\btime spent working\b", r"\bservice time\b",
                         r"\bprocess\w*\s+(?:time\s+)?(?:versus|vs\.?|rather than|"
                         r"not|compared (?:to|with))\s+(?:wait|queu)",
                         r"\b(?:wait\w*|queu\w*)\s+(?:time\s+)?(?:versus|vs\.?|"
                         r"rather than|not|compared (?:to|with))\s+process",
                         r"\bsplit between (?:waiting|queu\w+) and process",
                         r"\b(add|adding|more|extra|additional|adjust\w*)\s+"
                         r"(staff\w*|resourc\w*|capacity|headcount)\b",
                         r"\bincreas\w+\s+(the\s+)?(staff\w*|capacity|headcount|"
                         r"resourc\w*)\b",
                         r"\bhir\w+\s+(more\s+)?(staff|people|nurses|clinicians)\b",
                         r"\bstaffing\s+(level|increase|review)\w*\b"],
            "must_not_claim": "that the delay is queueing rather than processing "
                              "(or the reverse), or any split between the two",
            "because": f"this log records only {caps['lifecycle_values']}, so there "
                       "is no start timestamp. Elapsed time between events is all "
                       "that exists; it cannot be decomposed.",
        })
    if resource_col and not caps.get("handoff_split_meaningful", True):
        blocked.append({
            "id": "X-GRANULARITY",
            "patterns": [r"\bhandoff\b", r"\bhand-?off\b", r"\bbetween departments\b",
                         r"\bwithin (?:the )?department\b", r"\bcoordination between\b"],
            "must_not_claim": "any split between time lost INSIDE a department and "
                              "time lost BETWEEN departments",
            "because": f"'{resource_col}' identifies individual people "
                       f"({caps['n_resource_values']:,} distinct values, "
                       f"{caps['same_resource_transition_share'] * 100:.2f}% of "
                       "consecutive events share an actor), so nearly every "
                       "transition looks like a handoff regardless of how well "
                       "the process is coordinated.",
        })
    if not resource_col:
        blocked.append({
            "id": "X-NORESOURCE",
            "patterns": [r"\bdepartment\b", r"\bteam\b", r"\bresponsible for\b"],
            "must_not_claim": "anything about which department or team is "
                              "responsible for a delay",
            "because": "this log carries no resource or department column at all.",
        })
    if caps.get("resource_missing_share", 0) >= 0.05:
        blocked.append({
            "id": "X-INCOMPLETE",
            "patterns": [r"\bacross the whole process\b", r"\btotal for the hospital\b",
                         r"\bevery department\b"],
            "must_not_claim": "that a per-department total is a total for the whole "
                              "process",
            "because": f"{caps['resource_missing_share'] * 100:.1f}% of events have "
                       "no responsible party recorded.",
        })
    blocked.append({
        "id": "X-CAUSE",
            "patterns": [r"\bcaused by\b", r"\bbecause of the\b", r"\bis due to\b",
                         r"\bthe cause is\b", r"\bresponsible for the delay\b"],
        "must_not_claim": "that any measured delay was CAUSED by the step it is "
                          "attributed to",
        "because": "this is an observational event log. It records when things "
                   "happened, not why. Causes may be proposed as hypotheses and "
                   "must be labelled as such.",
    })

    # The blocked list is also emitted as category-5 rows, because
    # `15_narrate_findings.py` filters on `category.startswith("5")`.
    for b in blocked:
        found.append(finding(
            b["id"], "5. Cannot be claimed from this log", b["must_not_claim"],
            kind="limitation", source="18_process_profile.detect_capabilities",
            robustness="structural",
            wording="(must not be claimed)", caveat=b["because"],
            patterns=b.get("patterns", [])))

    return {
        "log": log_path.name,
        "generated_from": {
            "events": len(events), "cases": n_cases,
            "resource_column": resource_col,
            "total_measured_waiting_h": round(total_h, 2),
        },
        "capabilities": caps,
        "materiality": mat,
        "findings": found,
        "forbidden_claims": blocked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured findings from any event log.")
    parser.add_argument("--log", default=None, help="path to the event log")
    parser.add_argument("--prefix", default="20", help="output filename prefix")
    args = parser.parse_args()

    handoffs_default = ROOT / "data" / "raw" / "Sepsis Cases - Event Log.xes.gz"
    log_path = Path(args.log) if args.log else handoffs_default
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    if not log_path.exists():
        raise SystemExit(f"Log not found: {log_path}")

    result = extract(log_path, args.prefix)

    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{args.prefix}_findings.json"
    csv_path = OUT / f"{args.prefix}_findings.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(result["findings"]).to_csv(csv_path, index=False)

    print(f"\n{'=' * 92}\nEXTRACTED FINDINGS - {result['log']}\n{'=' * 92}")
    quotable = [f for f in result["findings"]
                if not f["category"].startswith("5")]
    print(f"  {len(quotable)} findings the model may use, "
          f"{len(result['forbidden_claims'])} claims it may not make\n")
    for f in result["findings"]:
        marker = "BLOCKED" if f["category"].startswith("5") else "  OK   "
        print(f"  [{marker}] {f['id']:<14} {f['finding'][:66]}")

    print(f"\n  wrote {json_path.relative_to(ROOT)}")
    print(f"  wrote {csv_path.relative_to(ROOT)}")
    print("\n  Every number above was computed by pm4py/pandas. The narration")
    print("  layer receives this file and may not introduce anything else.")


if __name__ == "__main__":
    main()
