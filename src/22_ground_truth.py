"""
STEP 22 - GROUND-TRUTH VALIDATION. Is the diagnosis actually RIGHT?

THE GAP THIS CLOSES

Every validation so far answers a weaker question than it appears to:

  Step 4.5 planted a bottleneck in OUR simulator - the one we calibrated.
  Step 9  proved the pipeline ACCEPTS an unseen log and refuses what it cannot
          support. It says nothing about whether the bottleneck it names is right.
  Step 11 proved the diagnosis is GROUNDED - every number traceable, every claim
          cited. Grounded is not the same as correct. A diagnosis can cite real
          findings impeccably and still point at the wrong step.

There is no ground truth for a real hospital. So we manufacture it: generate
random process structures, inject ONE defect we choose, hide it, run the pipeline
unchanged, and check whether what comes back is the defect we buried.

TWO METRICS THAT MUST NOT BE MERGED

  GROUNDING VALIDITY   - did the output survive the guard? Were all claims cited,
                         all numbers traceable, no forbidden claim made?
  DIAGNOSTIC CORRECTNESS - did it name the defect we actually injected?

These are independent, and conflating them is the standard way this kind of
result gets oversold. A diagnosis can be perfectly grounded and wrong. It can
also be rejected while its content was right. They are reported separately and
never averaged together.

A THIRD THING WORTH SEPARATING

  PIPELINE RECOVERY - did the deterministic analysis rank the injected defect
                      first, BEFORE any model saw it?

If the pipeline misses, the model cannot be right, and blaming the model would be
wrong. Reported separately so the two failure sources stay distinguishable.

WHAT IS VARIED
  size            6-14 activities, 150-800 cases
  noise           low / high variance, plus random extra events
  competing       a second, weaker anomaly that is NOT the answer
  deceptive       the obvious metric points the WRONG way:
                    - `decoy_median`: a decoy with a huge MEDIAN but few
                      occurrences, while the real defect wins on TOTAL
                    - `zero_median`: the real defect has a median of ZERO because
                      its events are batched, exactly the trap Step 2a fell into
                      when `median x count` scored the largest block of waiting
                      time at 0.0
  control         NO defect injected at all - anything named is a false positive

KEEPING THE ANSWER HIDDEN
The ground truth is written to its own file and never enters the findings. The
activities are called ACT00, ACT01, ... so no name can hint at which is slow. The
harness asserts, per scenario, that no ground-truth string appears in the findings
handed to the model.

THE GUARD IS NOT TOUCHED. If results are poor, that is the result.

RUN IT
    python src/22_ground_truth.py --quick        # 6 scenarios, smoke test
    python src/22_ground_truth.py                # the full sweep
    python src/22_ground_truth.py --no-model     # pipeline recovery only, no LLM
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
RUNS = OUT / "22_runs"
PY = sys.executable

EDGE = "  ->  "          # exactly how src/03 builds its edge keys
BASE_GAP_H = 1.0         # typical gap between consecutive steps, hours
DEFECT_TYPES = ["edge_delay", "self_loop", "routing", "handoff"]

# The hosted backend the project documents, so --api-key-env alone is enough to
# aim the sweep at it. See src/24_app.py and check_api_key.py for the same pair.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-120b"


def rule_line(title: str) -> None:
    print(f"\n{'=' * 92}\n{title}\n{'=' * 92}")


# ---------------------------------------------------------------------------
# 1. THE GENERATOR - random processes with one known defect
# ---------------------------------------------------------------------------
def build_structure(rng, n_activities: int, n_departments: int) -> dict:
    """
    A random but plausible process: a backbone every case follows, with an
    optional branch, and departments owning contiguous stretches of it.

    Departments are contiguous so that the resource column behaves like real
    departments (a department performs several steps in a row). That matters:
    the profiler's granularity check would otherwise flag it as person-level and
    the handoff analysis would be withheld - correctly, but it would mean these
    logs never exercise that path.
    """
    acts = [f"ACT{i:02d}" for i in range(n_activities)]
    depts = [f"DEPT{chr(ord('A') + i)}" for i in range(n_departments)]

    # contiguous blocks, so a department owns a run of consecutive activities
    edges_at = sorted(rng.choice(range(1, n_activities), size=n_departments - 1,
                                 replace=False))
    dept_of, current = {}, 0
    for i, act in enumerate(acts):
        if current < len(edges_at) and i >= edges_at[current]:
            current += 1
        dept_of[act] = depts[min(current, len(depts) - 1)]

    # an optional branch: a minority of cases skip a middle stretch
    branch_from = int(n_activities * 0.4)
    branch_to = min(branch_from + 2, n_activities - 1)

    # A RARE DETOUR, used to build valid decoys. It is named in the same scheme
    # as everything else, so nothing about the name marks it out.
    #
    # Getting this right took a correction. The first decoy was a rare ENORMOUS
    # delay on a COMMON edge - 4% of 400 occurrences at 1000 h is 16,000 h, which
    # exceeded the injected defect's total. The pipeline ranked it first
    # and was scored as wrong, when it had correctly found the largest absorber of
    # time. The test was broken, not the tool.
    #
    # A real decoy is a rare SLOW PATH: few cases take it, every one that does
    # waits a long time. That tops the median and p90 rankings while losing on
    # total - which is exactly the confusion the Step 2a metric fell into.
    rare_step = f"ACT{n_activities:02d}"
    dept_of[rare_step] = depts[rng.integers(len(depts))]
    rare_after = acts[max(1, n_activities // 2)]
    return {"activities": acts, "departments": depts, "dept_of": dept_of,
            "branch": (acts[branch_from], acts[branch_to]),
            "rare_step": rare_step, "rare_after": rare_after}


def choose_defect(rng, structure: dict, defect_type: str,
                  variant: str = "plain") -> dict:
    """
    Pick the defect(s) and describe them, including the target edge(s).

    The `two_defects` variant adds a SECOND target on a different edge. Both are
    meant to be found; recovering only the first is a failure, and is scored as
    one rather than rounded up to a pass.
    """
    acts = structure["activities"]
    mid = acts[2:-1] or acts[:1]

    if defect_type == "self_loop":
        a = str(rng.choice(mid))
        return {"defect_type": "self_loop", "activity": a,
                "target_edge": f"{a}{EDGE}{a}",
                "description": f"{a} is repeated several times per case"}

    if defect_type == "handoff":
        # a transition that crosses departments, so the defect is a coordination
        # failure rather than one team being slow
        crossings = [(acts[i], acts[i + 1]) for i in range(len(acts) - 1)
                     if structure["dept_of"][acts[i]]
                     != structure["dept_of"][acts[i + 1]]]
        a, b = crossings[rng.integers(len(crossings))] if crossings else (
            acts[0], acts[1])
        return {"defect_type": "handoff", "target_edge": f"{a}{EDGE}{b}",
                "description": f"work waits when passing {structure['dept_of'][a]} "
                               f"-> {structure['dept_of'][b]}"}

    if defect_type == "routing":
        a, b = structure["branch"]
        return {"defect_type": "routing", "target_edge": f"{a}{EDGE}{b}",
                "description": f"cases routed {a} -> {b} wait far longer"}

    i = int(rng.integers(1, len(acts) - 1))
    return {"defect_type": "edge_delay",
            "target_edge": f"{acts[i]}{EDGE}{acts[i + 1]}",
            "description": f"a delay sits on {acts[i]} -> {acts[i + 1]}"}


def add_secondary(rng, structure: dict, truth: dict) -> dict:
    """Plant a second, independent bottleneck on an edge the first does not use."""
    acts = structure["activities"]
    primary = truth.get("target_edge") or ""
    used = set(primary.split(EDGE))
    candidates = [f"{acts[i]}{EDGE}{acts[i + 1]}" for i in range(len(acts) - 1)
                  if f"{acts[i]}{EDGE}{acts[i + 1]}" != primary
                  and acts[i] not in used and acts[i + 1] not in used]
    if candidates:
        truth = dict(truth)
        truth["secondary_edge"] = str(rng.choice(candidates))
    return truth


def simulate(rng, structure: dict, truth: dict, n_cases: int, noise: str,
             variant: str, magnitude: float = 25.0) -> pd.DataFrame:
    """
    Emit an event log. Only `complete` events, as most real exports have.

    The defect is applied as EXTRA elapsed time on its target edge. Everything
    else is background noise, so any edge the pipeline ranks above the target is
    a genuine miss rather than an artefact of the generator.
    """
    acts = structure["activities"]
    dept_of = structure["dept_of"]
    sigma = 0.45 if noise == "low" else 1.0
    branch_from, branch_to = structure["branch"]
    target = truth.get("target_edge")

    # `magnitude` is the extra hours on the injected edge, against a ~1 h
    # background. The default of 25x is a LOUD defect - which is the right place
    # to start, but it means a perfect score says little about subtle problems.
    # `--magnitude` sweeps it down to find where detection actually breaks.
    RARE_SHARE = 0.04                # how many cases take the decoy detour
    RARE_DELAY = min(200.0, magnitude * 8)

    # The decoy must LOSE on total, or it stops being a decoy and the scenario
    # silently becomes unscoreable. Kept as arithmetic, not a comment, because
    # the first version of this generator broke exactly here.
    assert RARE_SHARE * RARE_DELAY < magnitude, "decoy would win on total"

    use_decoy = variant == "decoy_median"
    rare_step, rare_after = structure["rare_step"], structure["rare_after"]
    decoy_edge = f"{rare_after}{EDGE}{rare_step}" if use_decoy else None

    # A COMPETING signal: a second, genuine anomaly that is real but smaller.
    # Not a trap - just a noisier world in which the answer is less obvious.
    competing_edge = None
    if variant == "competing":
        candidates = [f"{acts[i]}{EDGE}{acts[i + 1]}"
                      for i in range(len(acts) - 1)
                      if f"{acts[i]}{EDGE}{acts[i + 1]}" != target]
        if candidates:
            competing_edge = str(rng.choice(candidates))

    # TWO REAL BOTTLENECKS, both meant to be found.
    #
    # This variant exists because of a defect the rest of this harness could not
    # possibly have caught. Every other scenario injects exactly ONE defect, which
    # makes "reports one bottleneck" and "reports THE bottleneck" produce identical
    # scores - so the pipeline reported only its argmax for a long time and passed
    # 27 scenarios, four defect types, two deceptive framings and six effect sizes
    # while doing it. A user's two-bottleneck log found it in one run.
    #
    # The secondary is LARGE by design - 75% of the primary, the same ratio as
    # that log - because the question is not whether a faint signal is detectable
    # but whether a blatant one is even looked at.
    secondary_edge = truth.get("secondary_edge") if variant == "two_defects" else None

    rows = []
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for case in range(n_cases):
        t = t0 + timedelta(hours=float(rng.uniform(0, 24 * 60)))
        takes_branch = rng.random() < 0.3
        path = []
        for i, a in enumerate(acts):
            if takes_branch and branch_from < a < branch_to:
                continue
            path.append(a)
            if truth.get("defect_type") == "self_loop" and a == truth.get("activity"):
                path.extend([a] * int(rng.integers(2, 5)))
            if use_decoy and a == rare_after and rng.random() < RARE_SHARE:
                path.append(rare_step)      # the rare, always-slow detour

        for i, a in enumerate(path):
            rows.append({"case:concept:name": f"C{case:05d}", "concept:name": a,
                         "time:timestamp": t, "org:group": dept_of[a],
                         "lifecycle:transition": "complete"})
            if i == len(path) - 1:
                break
            edge = f"{a}{EDGE}{path[i + 1]}"
            gap = float(rng.lognormal(np.log(BASE_GAP_H), sigma))

            if edge == target:
                if variant == "zero_median":
                    # Batched: most instances record no elapsed time at all, and
                    # the time lands on a minority. The MEDIAN is 0; the TOTAL is
                    # the largest in the log. This is the Step 2a trap exactly.
                    gap = 0.0 if rng.random() < 0.65 else gap + magnitude * 3
                elif truth.get("defect_type") == "routing":
                    gap += magnitude * (3.0 if takes_branch else 0.2)
                else:
                    gap += float(rng.lognormal(np.log(magnitude), 0.4))
            elif edge == decoy_edge:
                # Always slow, but only ~4% of cases ever traverse it. Tops the
                # median and p90 rankings; loses decisively on total.
                gap += float(rng.lognormal(np.log(RARE_DELAY), 0.3))
            elif edge == competing_edge:
                # real, and smaller than the injected defect
                gap += magnitude * 0.4
            elif edge == secondary_edge:
                # a second bottleneck in its own right, 75% of the primary
                gap += float(rng.lognormal(np.log(magnitude * 0.75), 0.4))

            t = t + timedelta(hours=gap)

    df = pd.DataFrame(rows)

    if noise == "high":
        # a few unrelated extra events, to make the structure less than pristine
        extra = df.sample(frac=0.05, random_state=int(rng.integers(1 << 30))).copy()
        extra["concept:name"] = rng.choice(acts, size=len(extra))
        df = pd.concat([df, extra]).sort_values(
            ["case:concept:name", "time:timestamp"], kind="stable")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. THE SCENARIOS
# ---------------------------------------------------------------------------
def build_scenarios(quick: bool) -> list:
    """A spread over size, noise, defect type and deception."""
    sizes = [("small", 6, 150, 2), ("medium", 10, 400, 3), ("large", 14, 800, 4)]
    if quick:
        return [
            {"id": "q1", "size": sizes[0], "noise": "low",
             "defect": "edge_delay", "variant": "plain"},
            {"id": "q2", "size": sizes[0], "noise": "high",
             "defect": "self_loop", "variant": "plain"},
            {"id": "q3", "size": sizes[1], "noise": "low",
             "defect": "handoff", "variant": "decoy_median"},
            {"id": "q4", "size": sizes[1], "noise": "low",
             "defect": "edge_delay", "variant": "zero_median"},
            {"id": "q5", "size": sizes[1], "noise": "high",
             "defect": "routing", "variant": "competing"},
            {"id": "q6", "size": sizes[0], "noise": "low",
             "defect": "none", "variant": "control"},
        ]

    scenarios, n = [], 0
    for size in sizes:
        for noise in ("low", "high"):
            for defect in DEFECT_TYPES:
                n += 1
                variant = ["plain", "decoy_median", "competing",
                           "zero_median", "two_defects"][n % 5]
                scenarios.append({"id": f"s{n:02d}", "size": size, "noise": noise,
                                  "defect": defect, "variant": variant})
    for i, size in enumerate(sizes):          # controls: no defect at all
        scenarios.append({"id": f"c{i + 1}", "size": size, "noise": "low",
                          "defect": "none", "variant": "control"})
    return scenarios


# ---------------------------------------------------------------------------
# 3. RUNNING THE PIPELINE - unchanged, as subprocesses
# ---------------------------------------------------------------------------
def run(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run([PY, "-u", *cmd], cwd=ROOT, capture_output=True,
                          text=True)


def ground_truth_is_attainable(log: pd.DataFrame, target: str | None) -> bool:
    """
    Is the injected defect actually the largest absorber of time in this log?

    A harness has to check its own answer key. Without this, a scenario where the
    injected delay is smaller than ordinary variation gets scored as a
    pipeline MISS, when the pipeline correctly identified the real largest
    absorber and the experiment simply had no detectable defect in it.

    That is not hypothetical - it is what `routing` + `zero_median` produced at
    low magnitudes. The routing defect lands on an edge only ~30% of cases take,
    and `zero_median` then zeroes 65% of those, so the "defect" totalled ~294 h
    against ~440 h on ordinary edges. Six scenarios were being counted as
    detection failures for finding the truth.

    Computed directly from the log, independently of the pipeline, so it cannot
    launder a pipeline error into a passing grade.
    """
    if not target:
        return True
    df = log.sort_values(["case:concept:name", "time:timestamp"], kind="stable")
    g = df.groupby("case:concept:name", sort=False)
    nxt = g["concept:name"].shift(-1)
    gap = (g["time:timestamp"].shift(-1) - df["time:timestamp"]
           ).dt.total_seconds() / 3600.0
    e = pd.DataFrame({"edge": df["concept:name"] + EDGE + nxt,
                      "gap": gap}).dropna()
    totals = e.groupby("edge")["gap"].sum()
    return bool(len(totals)) and totals.idxmax() == target


def score_materiality(findings: dict) -> dict:
    """
    Did the analysis say there is a bottleneck AT ALL, before naming one?

    This is step 23's question and nothing scored it, so the sweep could not
    show whether the permutation test was doing any work - the false-positive
    section was left asserting the pre-23 behaviour long after it changed. The
    verdict and the prohibition are recorded separately because they can
    disagree: the test can find nothing while the gate still fails to emit.
    """
    mat = findings.get("materiality", {}) or {}
    ids = {f.get("id") for f in findings.get("findings", [])}
    return {"material": mat.get("material"),
            "materiality_p": mat.get("p_value"),
            "nobottleneck_emitted": "X-NOBOTTLENECK" in ids}


def read_violation(rejected_path: Path) -> str:
    """Name the checks that actually failed, from the saved verification report."""
    if not rejected_path.exists():
        return "rejected (no report written)"
    report = json.loads(rejected_path.read_text(encoding="utf-8")).get(
        "verification", {})
    hits = []
    for key in ("missing_sections", "uncited_claims", "unknown_evidence_ids",
                "untestable_hypotheses", "unsupported_numbers"):
        if report.get(key):
            hits.append(key)
    for v in report.get("forbidden_claims_made", []):
        hits.append(f"forbidden:{v.get('id', '?')}")
    return ";".join(hits)


def score_pipeline(findings: dict, truth: dict) -> dict:
    """Did the DETERMINISTIC analysis put the injected defect on top?"""
    t1 = next((f for f in findings["findings"] if f["id"] == "T1"), None)
    if t1 is None:
        return {"pipeline_top_edge": None, "pipeline_recovered": False}
    text = t1["finding"]
    start = text.find("'")
    end = text.find("'", start + 1)
    top = text[start + 1:end] if start != -1 and end != -1 else ""
    # On a control there is nothing to recover, so recovery is UNDEFINED rather
    # than False. Scoring it False would quietly drag the headline accuracy down
    # with rows that had no right answer.
    if not truth.get("target_edge"):
        return {"pipeline_top_edge": top, "pipeline_recovered": None}

    # SECONDARY recovery is scored SEPARATELY, and only where one was planted.
    #
    # Folding it into `pipeline_recovered` would average the two together and
    # hide exactly the failure this variant exists to catch: finding the primary
    # and silently dropping the secondary would still read as a partial pass.
    # Kept as its own column so a regression shows up as a column of False, not
    # as a headline number drifting a few points.
    secondary = truth.get("secondary_edge")
    found_secondary = None
    if secondary:
        material = (findings.get("materiality") or {}).get("material_edges") or []
        found_secondary = any(m["edge"] == secondary for m in material)

    return {"pipeline_top_edge": top,
            "pipeline_recovered": top == truth["target_edge"],
            "secondary_edge": secondary,
            "secondary_recovered": found_secondary}


def score_diagnosis(diag: dict, findings: dict, truth: dict) -> dict:
    """
    Did the MODEL name the injected defect?

    Credited if the target edge appears in the bottleneck summary, or in any
    finding the bottleneck cites. Citing the right finding is the correct pass
    condition: the model is told not to restate numbers, so requiring the literal
    edge string in its own prose would punish it for following instructions.
    """
    target = truth.get("target_edge")
    if not target:
        return {"diagnosis_named": None, "diagnosis_correct": None}

    b = diag.get("bottleneck") or {}
    summary = b.get("summary", "")
    by_id = {f["id"]: f["finding"] for f in findings["findings"]}
    cited_text = " ".join(by_id.get(i, "") for i in b.get("evidence_ids", []))

    named = target in summary or target in cited_text
    return {"diagnosis_named": named,
            "diagnosis_correct": named}


def assert_no_leak(findings_path: Path, truth: dict) -> bool:
    """
    The model must not be able to read the answer out of its own input.

    Only `description` is checked, and that is a correction. The first version
    also checked `defect_type`, which reported 9 of 27 scenarios as leaking -
    wrongly. The words are not secrets: "handoff" appears in the X-GRANULARITY
    prohibition text, and "none" appears as a `parameter_dependence` value on
    every finding. Neither tells the model anything about THIS log.

    `target_edge` is not checked either. It appears in finding T1
    whenever the pipeline ranks it first - which is the pipeline's correct ANSWER,
    not a leak. Treating the answer as contamination would make a working pipeline
    look like a broken experiment.
    """
    text = findings_path.read_text(encoding="utf-8")
    description = str(truth.get("description") or "")
    if description and description != "no defect injected" and description in text:
        return False
    return True


def evaluate(scenario: dict, rng, use_model: bool, model_args: list,
             prefix: str = "22") -> dict:
    name, n_acts, n_cases, n_depts = scenario["size"]
    # Namespaced by run, so a magnitude sweep cannot overwrite the main sweep's
    # findings and destroy its ability to be re-scored later.
    tag = f"{prefix}_{scenario['id']}"
    RUNS.mkdir(parents=True, exist_ok=True)

    structure = build_structure(rng, n_acts, n_depts)
    truth = ({"defect_type": "none", "target_edge": None,
              "description": "no defect injected"}
             if scenario["defect"] == "none"
             else choose_defect(rng, structure, scenario["defect"],
                                scenario["variant"]))
    if scenario["variant"] == "two_defects" and truth["defect_type"] != "none":
        truth = add_secondary(rng, structure, truth)

    log = simulate(rng, structure, truth, n_cases, scenario["noise"],
                   scenario["variant"], scenario.get("magnitude", 25.0))
    log_path = RUNS / f"{tag}_log.csv"
    log.to_csv(log_path, index=False)
    (RUNS / f"{tag}_truth.json").write_text(
        json.dumps({**scenario, "size": list(scenario["size"]), "truth": truth},
                   indent=2), encoding="utf-8")

    row = {"id": scenario["id"], "size": name, "activities": n_acts,
           "cases": n_cases, "noise": scenario["noise"],
           "defect_type": truth["defect_type"], "variant": scenario["variant"],
           "target_edge": truth["target_edge"], "events": len(log),
           "magnitude": scenario.get("magnitude", 25.0),
           # Checked against the log itself, before the pipeline runs.
           "gt_attainable": ground_truth_is_attainable(log, truth["target_edge"])}

    # ---- the pipeline, unchanged --------------------------------------------
    p20 = run(["src/20_extract_findings.py", "--log",
               str(log_path.relative_to(ROOT)), "--prefix", tag])
    row["extract_ok"] = p20.returncode == 0
    if not row["extract_ok"]:
        row["error"] = (p20.stderr.strip().splitlines() or ["?"])[-1][:120]
        return row

    findings_path = OUT / f"{tag}_findings.json"
    findings = json.loads(findings_path.read_text(encoding="utf-8"))
    row["ground_truth_hidden"] = assert_no_leak(findings_path, truth)
    row.update(score_pipeline(findings, truth))
    row.update(score_materiality(findings))

    if not use_model:
        return row

    # ---- the diagnosis layer, unchanged -------------------------------------
    p21 = run(["src/21_diagnose.py", "--findings",
               str(findings_path.relative_to(ROOT)), "--diagnose", *model_args])
    stdout = p21.stdout
    rejected_path = OUT / f"{tag}_findings_diagnosis_REJECTED.json"

    accepted = "ACCEPTED - wrote" in stdout

    # A failed model CALL is not a guard rejection, and must not be scored as one.
    #
    # src/21 has two ways to not accept. It can reach a verdict and reject one -
    # which writes the REJECTED report. Or it can never reach a verdict at all,
    # because the call failed (no key, no network, a reply truncated mid-JSON) -
    # which writes nothing. Both used to land in the same bucket, so a run with
    # no credential reported 27 abstentions - indistinguishable from the guard
    # having judged and rejected 27 diagnoses it never saw. The absent artifact
    # is the tell; carry the error text out instead of a violation name.
    if not accepted and not rejected_path.exists():
        row["model_call_failed"] = True
        row["grounding_accepted"] = None
        row["grounding_violation"] = ""
        row["abstained"] = None
        row["diagnosis_correct"] = None
        why = [ln.strip() for ln in (stdout or "").splitlines()
               if "REJECTED -" in ln]
        row["error"] = (why or (p21.stderr or "").strip().splitlines()
                        or ["no output"])[-1][:160]
        return row

    row["model_call_failed"] = False
    row["grounding_accepted"] = accepted

    # Read the violation from the REJECTED artifact, not by grepping stdout.
    #
    # Grepping was wrong and produced a nonsense table: the marker "forbidden
    # claim" matched the INPUT SUMMARY line ("forbidden claims   : 2") that every
    # run prints, so all 27 scenarios were listed as violating - including the 20
    # that passed. The artifact carries the actual verification report, so read
    # that and there is nothing to mis-match.
    row["grounding_violation"] = read_violation(
        rejected_path) if not accepted else ""

    if not accepted:
        row["abstained"] = True
        row["diagnosis_correct"] = None
        return row

    row["abstained"] = False
    diag_path = OUT / f"{tag}_findings_diagnosis.json"
    if diag_path.exists():
        row.update(score_diagnosis(
            json.loads(diag_path.read_text(encoding="utf-8")), findings, truth))
    return row


# ---------------------------------------------------------------------------
# 4. REPORT
# ---------------------------------------------------------------------------
def write_report(df: pd.DataFrame, used_model: bool) -> str:
    lines = ["# Step 22 - Ground-truth validation", ""]
    lines.append(f"_{len(df)} randomised process structures, each with one known "
                 "injected defect (or none). The pipeline was run unchanged; the "
                 "guard was not modified._")
    lines.append("")

    defective = df[df["defect_type"] != "none"]
    controls = df[df["defect_type"] == "none"]

    # ---- pipeline recovery -------------------------------------------------
    lines += ["## 1. Pipeline recovery (deterministic, before any model)", ""]
    if "gt_attainable" in defective.columns:
        unattainable = defective[~defective["gt_attainable"].astype(bool)]
        defective = defective[defective["gt_attainable"].astype(bool)]
    else:
        unattainable = defective.iloc[0:0]

    if len(unattainable):
        lines.append(f"_{len(unattainable)} scenario(s) excluded: the injected "
                     "defect was not in fact the largest absorber of time in the "
                     "generated log, so there was no correct answer to find. "
                     "Checked against the log directly, before the pipeline ran. "
                     "Listed in section 6._")
        lines.append("")
    if len(defective):
        rec = defective["pipeline_recovered"].sum()
        lines.append(f"**{rec}/{len(defective)} "
                     f"({rec / len(defective) * 100:.0f}%)** of injected defects "
                     "were ranked first by the analysis itself.")
        lines.append("")
        lines.append("| variant | recovered | n |")
        lines.append("|---|---|---|")
        for variant, grp in defective.groupby("variant"):
            lines.append(f"| {variant} | {grp['pipeline_recovered'].sum()} "
                         f"| {len(grp)} |")
        lines.append("")
        lines.append("| defect type | recovered | n |")
        lines.append("|---|---|---|")
        for dt, grp in defective.groupby("defect_type"):
            lines.append(f"| {dt} | {grp['pipeline_recovered'].sum()} | {len(grp)} |")
        lines.append("")

    # ---- secondary bottlenecks ---------------------------------------------
    if "secondary_recovered" in df.columns:
        planted = df[df["secondary_recovered"].notna()]
        if len(planted):
            got = int(planted["secondary_recovered"].astype(bool).sum())
            lines += ["### Secondary bottlenecks", "",
                      f"**{got}/{len(planted)}** scenarios with a SECOND planted "
                      "bottleneck had it reported as well as the first.", "",
                      "This variant exists because every other scenario injects "
                      "exactly one defect, which makes *reporting one bottleneck* "
                      "and *reporting the bottleneck* score identically. The "
                      "pipeline reported only its argmax for a long time and "
                      "passed everything above while doing it.", ""]
            missed = planted[~planted["secondary_recovered"].astype(bool)]
            if len(missed):
                lines += ["| scenario | primary | secondary MISSED |",
                          "|---|---|---|"]
                for _, r in missed.iterrows():
                    lines.append(f"| {r['id']} | `{r['target_edge']}` "
                                 f"| `{r['secondary_edge']}` |")
                lines.append("")

    if not used_model:
        lines.append("_Model layer not run (`--no-model`)._")
        return "\n".join(lines) + "\n"

    # ---- grounding validity ------------------------------------------------
    lines += ["## 2. Grounding validity (is the output verifiable?)", ""]

    # Scenarios where the model call never returned are held OUT of this metric
    # rather than counted as rejections. The guard did not decide anything about
    # them, and a run that fails to reach the provider would otherwise publish a
    # perfect abstention rate as though caution had been demonstrated.
    failed = (df[df["model_call_failed"].astype(bool)]
              if "model_call_failed" in df.columns else df.iloc[0:0])
    scored = df.drop(index=failed.index)
    if len(failed):
        lines += [f"_{len(failed)} scenario(s) excluded: the model call itself "
                  "failed, so the guard never ran on them. Not counted as "
                  "abstentions - see section 7._", ""]

    acc = int(scored["grounding_accepted"].astype(bool).sum())
    lines.append(f"**{acc}/{len(scored)}** diagnoses passed the guard. "
                 f"**{len(scored) - acc}** were rejected and nothing was written.")
    viol = df[df["grounding_violation"].astype(bool)]
    if len(viol):
        lines.append("")
        lines.append("| scenario | violation |")
        lines.append("|---|---|")
        for _, r in viol.iterrows():
            lines.append(f"| {r['id']} | {r['grounding_violation']} |")
    lines.append("")
    lines.append("A rejection is an ABSTENTION, not a wrong answer. It is counted "
                 "separately from correctness throughout.")
    lines.append("")

    # ---- diagnostic correctness -------------------------------------------
    lines += ["## 3. Diagnostic correctness (did it name the real defect?)", "",
              "**Read this metric carefully.** By design, the model does not "
              "search for the bottleneck - the deterministic pipeline ranks it and "
              "hands it over as finding `T1`. So this measures whether the model "
              "pointed at the right finding among the several it was given, not "
              "whether it located the bottleneck itself. Finding it is the "
              "pipeline's job (section 1); that separation is the architecture, "
              "not an accident. A high score here is evidence the model does not "
              "wander off the evidence - not evidence that it is analytically "
              "clever.", ""]
    judged = defective[defective["abstained"] == False]          # noqa: E712
    if len(judged):
        correct = judged["diagnosis_correct"].sum()
        lines.append(f"Of the **{len(judged)}** accepted diagnoses on defective "
                     f"processes, **{correct} ({correct / len(judged) * 100:.0f}%)** "
                     "named the injected defect.")
        wrong = judged[judged["diagnosis_correct"] == False]      # noqa: E712
        if len(wrong):
            lines += ["", "### Incorrect diagnoses", "",
                      "| scenario | variant | injected | pipeline ranked #1 | "
                      "whose fault |", "|---|---|---|---|---|"]
            for _, r in wrong.iterrows():
                # The attribution that matters: if the analysis already ranked the
                # wrong edge first, the model repeating it faithfully is not a
                # model failure. Blaming the LLM for a measurement error would be
                # the easiest wrong conclusion to draw from this table.
                blame = ("pipeline (model was faithful to it)"
                         if not r["pipeline_recovered"] else "model")
                lines.append(f"| {r['id']} | {r['variant']} | `{r['target_edge']}` "
                             f"| `{r['pipeline_top_edge']}` | {blame} |")
            n_pipeline = int((~wrong["pipeline_recovered"].astype(bool)).sum())
            lines += ["", f"Of {len(wrong)} incorrect diagnoses, **{n_pipeline}** "
                      "followed a pipeline that had already ranked the wrong edge "
                      f"first, and **{len(wrong) - n_pipeline}** contradicted a "
                      "pipeline that had it right."]
    lines.append("")

    # ---- false positives ---------------------------------------------------
    lines += ["## 4. False positives (controls with NO defect)", ""]
    controls = controls.drop(index=controls.index.intersection(failed.index))
    if len(controls):
        named = int(controls["grounding_accepted"].astype(bool).sum())
        lines.append(f"{len(controls)} control processes had no defect injected. "
                     f"**{named}** still produced an accepted diagnosis naming a "
                     "bottleneck.")
        lines.append("")

        # This paragraph used to assert that nothing distinguished an injected
        # delay from ordinary variation. That was true when it was written and
        # stopped being true when the permutation test was added, so it is now
        # derived from the controls rather than stated.
        if "material" in controls.columns:
            immaterial = int((controls["material"] == False).sum())   # noqa: E712
            gated = int(controls["nobottleneck_emitted"].astype(bool).sum())
            ps = ", ".join(f"{v:.4g}" for v in controls["materiality_p"].dropna())
            lines.append(
                f"The permutation test found no material bottleneck in "
                f"**{immaterial}/{len(controls)}** of them (p = {ps}), and the "
                f"capability gate emitted the `X-NOBOTTLENECK` prohibition on "
                f"**{gated}/{len(controls)}**. The analysis therefore "
                "distinguishes a healthy process from a defective one, which it "
                "could not do when this sweep was first run.")
            lines.append("")
            lines.append(
                "The prohibition is not what stopped the output. It matches "
                "phrases containing the word *bottleneck*, and the model avoided "
                "that word while writing that waiting time was *concentrated in* "
                "a particular step - the claim the prohibition exists to "
                "prevent. The diagnoses were rejected by a separate check, for "
                "returning empty `probable_drivers` and `recommendations`. "
                "Matching a phrasing is not the same as constraining a claim; "
                "§11 of REPORT.md states that limit in full.")
    lines.append("")
    if len(unattainable):
        lines += ["## 6. Scenarios with no attainable answer", "",
                  "These were generated with a defect that turned out smaller than "
                  "ordinary variation in the same log. The pipeline naming a "
                  "different edge is correct behaviour, not a miss - so they are "
                  "excluded from the accuracy above rather than counted as "
                  "failures.", "",
                  "| scenario | defect | variant | injected | actually largest |",
                  "|---|---|---|---|---|"]
        for _, r in unattainable.iterrows():
            lines.append(f"| {r['id']} | {r['defect_type']} | {r['variant']} "
                         f"| `{r['target_edge']}` | `{r['pipeline_top_edge']}` |")
        lines.append("")

    if len(failed):
        lines += ["## 7. Scenarios the model layer never reached", "",
                  "The call to the provider failed before the guard could judge "
                  "anything, so these carry no grounding or correctness result. "
                  "They are listed rather than dropped, because a sweep that "
                  "quietly reported fewer scenarios than it ran would hide the "
                  "very failure this section exists to show.", "",
                  "| scenario | error |", "|---|---|"]
        for _, r in failed.iterrows():
            lines.append(f"| {r['id']} | {str(r.get('error', '')).strip()} |")
        lines.append("")

    lines += ["## 5. Ground truth was hidden", ""]
    leaked = (~df["ground_truth_hidden"].astype(bool)).sum()
    lines.append(f"{len(df) - leaked}/{len(df)} scenarios verified to contain no "
                 "ground-truth string in the findings handed to the model.")
    return "\n".join(lines) + "\n"


def rebuild_rows(tag: str) -> list:
    """
    Re-score a completed sweep from the artifacts it left on disk.

    Every input still exists - the truth files, the findings, the accepted
    diagnoses and the REJECTED verification reports - so a scoring bug can be
    corrected without spending another half hour of model time. Re-running would
    also resample the model and quietly change the results being corrected.

    One input does NOT survive: the generated event logs are ~10 MB per sweep and
    get cleaned up. `gt_attainable` is derived from them, so it is carried over
    from the previous results file instead, and any scenario where neither is
    available is named rather than silently dropped.
    """
    rows = []
    missing_logs: list = []

    prior_path = OUT / f"{tag}_results.csv"
    prior = None
    if prior_path.exists():
        try:
            prior = pd.read_csv(prior_path).set_index("id")
            if "gt_attainable" not in prior.columns:
                prior = None
        except Exception:                                      # noqa: BLE001
            prior = None
    for truth_path in sorted(RUNS.glob(f"{tag}_*_truth.json")):
        meta = json.loads(truth_path.read_text(encoding="utf-8"))
        truth, sid = meta["truth"], meta["id"]
        run_tag = f"{tag}_{sid}"
        findings_path = OUT / f"{run_tag}_findings.json"
        if not findings_path.exists():
            continue
        findings = json.loads(findings_path.read_text(encoding="utf-8"))
        size = meta["size"]
        row = {"id": sid, "size": size[0], "activities": size[1],
               "cases": size[2], "noise": meta["noise"],
               "defect_type": truth["defect_type"], "variant": meta["variant"],
               "target_edge": truth["target_edge"], "extract_ok": True,
               "ground_truth_hidden": assert_no_leak(findings_path, truth)}
        log_path = RUNS / f"{run_tag}_log.csv"
        if log_path.exists():
            log = pd.read_csv(log_path, parse_dates=["time:timestamp"])
            row["gt_attainable"] = ground_truth_is_attainable(
                log, truth["target_edge"])
        elif prior is not None and run_tag.split("_", 1)[-1] in prior.index:
            # The generated logs are large and get deleted; attainability is the
            # one column that cannot be recomputed without them. Carrying the
            # previous value forward is not cosmetic - section 1 EXCLUDES
            # unattainable scenarios from the denominator, so losing the column
            # silently enlarges it and inflates the recovery rate. Recovered
            # from the last results file, and if that is gone too, said out loud
            # rather than dropped.
            row["gt_attainable"] = bool(
                prior.loc[run_tag.split("_", 1)[-1], "gt_attainable"])
        else:
            missing_logs.append(run_tag)
        row.update(score_pipeline(findings, truth))
        row.update(score_materiality(findings))

        diag = OUT / f"{run_tag}_findings_diagnosis.json"
        rej = OUT / f"{run_tag}_findings_diagnosis_REJECTED.json"
        accepted = diag.exists() and not rej.exists()

        # Neither artifact present means src/21 never reached a verdict, so this
        # scenario has no grounding result to rebuild - the same distinction
        # evaluate() draws. Scoring it as an abstention here would reintroduce,
        # at rebuild time, exactly the failure the live path now avoids.
        row["model_call_failed"] = not diag.exists() and not rej.exists()
        if row["model_call_failed"]:
            row["grounding_accepted"] = None
            row["grounding_violation"] = ""
            row["abstained"] = None
            row["diagnosis_correct"] = None
            rows.append(row)
            continue

        row["grounding_accepted"] = accepted
        row["grounding_violation"] = "" if accepted else read_violation(rej)
        row["abstained"] = not accepted
        if accepted:
            row.update(score_diagnosis(
                json.loads(diag.read_text(encoding="utf-8")), findings, truth))
        else:
            row["diagnosis_correct"] = None
        rows.append(row)

    if missing_logs:
        named = ", ".join(missing_logs[:8]) + (" ..." if len(missing_logs) > 8
                                               else "")
        print(f"  NOTE: {len(missing_logs)} scenario(s) have neither a generated "
              "log nor a previous", flush=True)
        print("        result, so attainability could not be established and "
              "section 1 will", flush=True)
        print(f"        not exclude any: {named}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Ground-truth validation harness.")
    ap.add_argument("--rebuild-report", action="store_true",
                    help="re-score a finished sweep from its artifacts, without "
                         "calling the model again")
    ap.add_argument("--quick", action="store_true", help="6 scenarios only")
    ap.add_argument("--no-model", action="store_true",
                    help="pipeline recovery only, skip the LLM")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--magnitude", type=float, default=None,
                    help="defect size in hours above a ~1h background "
                         "(default 25 = loud). Lower values probe the floor.")
    ap.add_argument("--tag", default="22",
                    help="output prefix, so a magnitude sweep does not "
                         "overwrite the main run")
    # Left unset, these follow --api-key-env rather than defaulting blindly.
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--api-key-env", default=None,
                    help="env var holding the provider's API key, e.g. "
                         "GROQ_API_KEY. Omit for Ollama, which needs none.")
    args = ap.parse_args()

    # The endpoint defaults follow the credential, because the two are not
    # independent. They used to default to Ollama's localhost unconditionally,
    # so `--api-key-env GROQ_API_KEY` sent the key to a local Ollama that wanted
    # no key and, on a machine not running one, failed every call - while the
    # error text talked about `ollama serve` in a run the operator believed was
    # going to Groq. A key names a hosted provider; honour that.
    if args.base_url is None:
        args.base_url = (GROQ_BASE_URL if args.api_key_env
                         else "http://localhost:11434/v1")
    if args.model is None:
        args.model = GROQ_MODEL if args.api_key_env else "llama3.1-16k"

    # --api-key-env is passed through, and its absence was a real bug. This
    # harness was written against Ollama, which needs no credential, so the
    # argument never existed. Pointed at a keyed provider every call went out
    # unauthenticated and failed BEFORE verification ran - which the report then
    # counted as 27 abstentions, indistinguishable from the guard rejecting 27
    # diagnoses. An infrastructure failure must not be able to masquerade as a
    # result, so the sweep now refuses to start if the named variable is unset.
    model_args = ["--provider", "openai-compat", "--base-url", args.base_url,
                  "--model", args.model]
    if args.api_key_env:
        if not os.environ.get(args.api_key_env):
            raise SystemExit(
                f"{args.api_key_env} is not set, so every model call would fail "
                "and be recorded as an abstention.\n"
                "  Set it, or run with --no-model to test the pipeline alone.")
        model_args += ["--api-key-env", args.api_key_env]

    if args.rebuild_report:
        rows = rebuild_rows(args.tag)
        if not rows:
            raise SystemExit(f"No artifacts found for tag '{args.tag}'.")
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"{args.tag}_results.csv", index=False)
        report = write_report(df, used_model=True)
        (OUT / f"{args.tag}_report.md").write_text(report, encoding="utf-8")
        rule_line(f"REBUILT FROM ARTIFACTS - {len(rows)} scenarios")
        print(report)
        return

    scenarios = build_scenarios(args.quick)
    if args.magnitude is not None:
        for sc in scenarios:
            sc["magnitude"] = args.magnitude

    rule_line(f"GROUND-TRUTH VALIDATION - {len(scenarios)} scenarios")
    print(f"  seed {args.seed}; model layer: "
          f"{'OFF' if args.no_model else args.model}")
    print("  The pipeline and the guard are run UNCHANGED.\n")

    rows = []
    for i, scenario in enumerate(scenarios, 1):
        rng = np.random.default_rng(args.seed + i * 977)
        print(f"  [{i:>2}/{len(scenarios)}] {scenario['id']:<4} "
              f"{scenario['size'][0]:<7} {scenario['noise']:<5} "
              f"{scenario['defect']:<11} {scenario['variant']}", flush=True)
        row = evaluate(scenario, rng, not args.no_model, model_args,
                       prefix=args.tag)
        rows.append(row)
        verdict = []
        if row.get("pipeline_recovered") is not None:
            verdict.append("pipeline " + ("HIT" if row["pipeline_recovered"]
                                          else "miss"))
        if "grounding_accepted" in row:
            verdict.append("guard " + ("accepted" if row["grounding_accepted"]
                                       else "REJECTED"))
        if row.get("diagnosis_correct") is not None:
            verdict.append("diagnosis " + ("CORRECT" if row["diagnosis_correct"]
                                           else "wrong"))
        print(f"           -> {', '.join(verdict)}", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{args.tag}_results.csv", index=False)
    report = write_report(df, not args.no_model)
    (OUT / f"{args.tag}_report.md").write_text(report, encoding="utf-8")

    rule_line("SUMMARY")
    print(report)
    print(f"  wrote outputs/{args.tag}_results.csv and outputs/{args.tag}_report.md")


if __name__ == "__main__":
    main()
