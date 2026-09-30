"""
STEP 9a - Discover a process profile from ANY event log.

WHY THIS EXISTS

  The analysis in Steps 2-8 is correct for the Sepsis log and would be SILENTLY
  WRONG on any other hospital's data. Two constants carry domain knowledge that
  a new log does not share:

      RELEASE_PREFIX  = "Release"      (src/03, src/05)
      RETURN_ACTIVITY = "Return ER"    (src/03, src/05)

  They drive the "episode cut" - the rule that stops the analysis measuring the
  time a discharged patient spends AT HOME before returning. On the Sepsis log
  that cut removed 23,800 days and demoted a bogus 47-day "bottleneck" that had
  been topping the ranking. On a log where no activity is called "Release", the
  cut quietly becomes a no-op, the at-home time is counted as hospital waiting,
  and the tool reports a confident, wrong answer with NO error.

  That is the failure mode this module removes. It discovers the same structure
  from the data instead of being told the names.

  Note what is NOT a problem: the bottleneck ranking itself is already
  data-driven (`sort_values("total_hours")`), so nothing predefines which
  bottleneck to find. The system never memorised the answer - it memorised the
  process structure needed to compute the answer correctly.

WHAT IS DISCOVERED, AND ON WHAT EVIDENCE

  1. CAPABILITIES - which analyses this log can support at all.
     Does it carry a lifecycle column with more than one value (can waiting be
     separated from processing)? Does it carry a resource/department column (can
     handoffs be measured)? A new log may support neither, and the tool must say
     so rather than silently produce a weaker analysis that looks the same.

  2. EPISODE BOUNDARIES - without knowing any activity name.
     An activity ends an episode when it is disproportionately the LAST event of
     a case AND, on the occasions it is not last, the gap that follows is an
     extreme outlier against that same log's typical gaps. That is what "the
     patient went home" looks like in any process: a terminal-ish step followed
     by a pause far longer than anything inside the case.
     The partner rule finds restart activities: those that disproportionately
     FOLLOW such a gap.

  3. DEPARTMENT ROLES - by what each group does, never by its name.
     A group's role is the set of activities it performs. Reported as evidence
     ("this group performs 100% of the three lab tests"), so a reader can name it
     themselves. The Sepsis log's codes are anonymised letters, so a name-based
     map could never have generalised anyway.

  4. SEGMENTS - case attributes worth comparing cohorts on.
     Columns that are constant within a case and have low cardinality, e.g.
     payment mode, age band, admission type. Discovered, not assumed.

  5. REWORK - self-loops and repeated activities, which is where the Sepsis log
     hid 44.6% of its waiting time.

USAGE

    python src/18_process_profile.py --log "data/raw/Sepsis Cases - Event Log.xes.gz"
    python src/18_process_profile.py --log data/synthetic/indian_hospital_log.csv

  Writes outputs/18_process_profile__<name>.json for the analysis layer to read.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

# XES defaults. Overridable so a log with different column names still works.
DEF_CASE = "case:concept:name"
DEF_ACT = "concept:name"
DEF_TS = "time:timestamp"
DEF_LIFECYCLE = "lifecycle:transition"
DEF_RESOURCE_CANDIDATES = ["org:group", "org:resource", "resource", "department"]

# Discovery thresholds. Declared here, in one place, so they can be defended and
# tuned - not scattered through the logic as magic numbers.
TERMINAL_SHARE_MIN = 0.30      # >=30% of its occurrences are the case's last event
GAP_OUTLIER_FACTOR = 10.0      # following gap >= 10x the log's median inter-event gap
SEGMENT_MAX_CARDINALITY = 12   # a useful cohort split, not a free-text field
SEGMENT_MIN_CARDINALITY = 2
ROLE_COVERAGE = 0.60           # activities covering 60% of a department's work
ROLE_MAX_SIGNATURE = 3         # ...capped at 3, so the signature stays readable


def rule_line(title: str) -> None:
    print(f"\n{'=' * 92}\n{title}\n{'=' * 92}")


def load_log(path: Path, case: str, act: str, ts: str) -> pd.DataFrame:
    """Read .xes/.xes.gz via pm4py, or any CSV that carries the three columns."""
    if path.suffix.lower() in {".xes", ".gz"}:
        import pm4py
        df = pm4py.read_xes(str(path))
    else:
        df = pd.read_csv(path)

    missing = [c for c in (case, act, ts) if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Log is missing required column(s): {missing}\n"
            f"  Columns present: {list(df.columns)[:12]}\n"
            "  Pass --case-col / --activity-col / --timestamp-col to map them.")

    df[ts] = pd.to_datetime(df[ts], utc=True, format="mixed")
    return df.sort_values([case, ts], kind="stable").reset_index(drop=True)


def detect_capabilities(df: pd.DataFrame, lifecycle: str,
                        case: str = DEF_CASE) -> dict:
    """
    What can this log actually support? Stated up front so a weaker analysis is
    never mistaken for a full one.
    """
    caps = {}

    if lifecycle in df.columns:
        values = sorted(str(v) for v in df[lifecycle].dropna().unique())
        caps["lifecycle_values"] = values
        caps["can_separate_wait_from_work"] = len(values) > 1
    else:
        caps["lifecycle_values"] = []
        caps["can_separate_wait_from_work"] = False

    resource_col = next((c for c in DEF_RESOURCE_CANDIDATES if c in df.columns), None)
    caps["resource_column"] = resource_col
    caps["can_measure_handoffs"] = resource_col is not None
    if resource_col:
        caps["n_resource_values"] = int(df[resource_col].nunique())

        # How much of the column is actually filled? CLAUDE.md records that Sepsis'
        # `org:group` is 100% filled, and that fact quietly became an assumption in
        # code written against it. The holdout log is 45% EMPTY, which matters twice:
        # a missing value is not a department, and because NaN != NaN it also counts
        # as a "different actor" in any transition test. Stated, not assumed.
        missing_share = float(df[resource_col].isna().mean())
        caps["resource_missing_share"] = round(missing_share, 4)

        # IS THIS COLUMN A DEPARTMENT OR A PERSON?
        #
        # Found by the holdout test, not by design. The Sepsis log's `org:group`
        # holds 26 DEPARTMENTS; Hospital Billing's `org:resource` holds 1,150
        # INDIVIDUALS. The internal-vs-handoff split only means something for the
        # former: with 1,150 people, consecutive events are almost never the same
        # person, so 100% of transitions look like "handoffs" and the split is
        # noise. The pipeline computed it anyway, without error - exactly the kind
        # of confident-but-meaningless output this project keeps having to catch.
        #
        # The signal is structural rather than a raw count: within a real
        # department you expect consecutive steps by the same unit a fair share of
        # the time. Sepsis scores ~0.45; Hospital Billing ~0.00002.
        nxt = df.groupby(case, sort=False)[resource_col].shift(-1)
        transitions = nxt.notna()           # drop each case's final event
        share = (float((df[resource_col][transitions] == nxt[transitions]).mean())
                 if transitions.any() else 0.0)
        caps["same_resource_transition_share"] = round(share, 5)
        caps["resource_granularity"] = (
            "department-like" if share >= 0.05 else "individual-like")
        caps["handoff_split_meaningful"] = share >= 0.05
    return caps


def detect_episode_boundaries(df: pd.DataFrame, case: str, act: str,
                              ts: str) -> dict:
    """
    Find, without any activity name, the activities that END an episode of care
    and those that RESTART one.

    The signature of "the patient left" is structural, not lexical:
      - the activity is often the final event of its case, AND
      - when it is not final, the gap after it dwarfs the log's normal gaps.

    Both conditions matter. Condition (a) alone would flag any common last step;
    condition (b) alone would flag ordinary long waits.
    """
    ev = df[[case, act, ts]].copy()
    grouped = ev.groupby(case, sort=False)
    ev["next_act"] = grouped[act].shift(-1)
    ev["gap_h"] = (grouped[ts].shift(-1) - ev[ts]).dt.total_seconds() / 3600

    gaps = ev["gap_h"].dropna()
    median_gap = float(gaps.median()) if len(gaps) else 0.0
    # Guard against a log whose median gap is 0 (batch-entered events): fall back
    # to a high percentile so the outlier test stays meaningful.
    reference_gap = median_gap if median_gap > 0 else float(
        gaps[gaps > 0].median()) if (gaps > 0).any() else 1.0
    outlier_threshold = reference_gap * GAP_OUTLIER_FACTOR

    is_last = ev["next_act"].isna()
    stats = []
    for activity, grp in ev.groupby(act):
        n = len(grp)
        terminal_share = float(is_last.loc[grp.index].mean())
        after = grp["gap_h"].dropna()
        outlier_share = float((after >= outlier_threshold).mean()) if len(after) else 0.0
        stats.append({
            "activity": str(activity),
            "occurrences": int(n),
            "terminal_share": round(terminal_share, 3),
            "share_followed_by_outlier_gap": round(outlier_share, 3),
            "median_following_gap_h": round(float(after.median()), 2) if len(after) else None,
        })

    ends = [s for s in stats
            if s["terminal_share"] >= TERMINAL_SHARE_MIN
            and s["share_followed_by_outlier_gap"] >= 0.5]

    # Restart activities: those that most often FOLLOW one of those huge gaps.
    restarts = []
    if ends:
        end_names = {s["activity"] for s in ends}
        after_end = ev[ev[act].isin(end_names) & (ev["gap_h"] >= outlier_threshold)]
        if len(after_end):
            counts = after_end["next_act"].value_counts()
            total = int(counts.sum())
            restarts = [{"activity": str(a), "times_following_a_break": int(c),
                         "share": round(c / total, 3)}
                        for a, c in counts.items() if c / total >= 0.10]

    return {
        "reference_gap_h": round(reference_gap, 3),
        "outlier_threshold_h": round(outlier_threshold, 2),
        "episode_end_activities": [s["activity"] for s in ends],
        "episode_restart_activities": [r["activity"] for r in restarts],
        "evidence": {"end_candidates": sorted(
            stats, key=lambda s: -s["terminal_share"])[:8],
            "restart_candidates": restarts},
        "cut_applies": bool(ends),
    }


def detect_department_roles(df: pd.DataFrame, act: str, resource: str | None) -> list:
    """
    Describe each department by WHAT IT DOES. No name mapping - the Sepsis log's
    groups are anonymised letters, so any lexical map was never going to travel.
    """
    if not resource:
        return []
    roles = []
    for group, grp in df.groupby(resource):
        counts = grp[act].value_counts()
        top = [{"activity": str(a), "share_of_group": round(c / len(grp), 3)}
               for a, c in counts.head(4).items()]
        roles.append({
            "group": str(group),
            "events": int(len(grp)),
            "share_of_log": round(len(grp) / len(df), 4),
            "performs": top,
        })
    return sorted(roles, key=lambda r: -r["events"])


def cluster_departments_by_behaviour(df: pd.DataFrame, act: str,
                                     resource: str | None) -> dict:
    """
    Collapse many departments into a few READABLE roles, using behaviour only.

    Why this has to exist: the Sepsis department map is technically correct and
    completely unusable - 26 nodes, of which 20 are individual wards that Step 2b
    showed all behave identically. An administrator cannot read it. The old fix
    was a hand-typed dictionary ({"B": "Laboratory", "A": "ER team", ...}), which
    is exactly the kind of thing that cannot travel to a new hospital: another
    log's departments are not called A, B, C.

    The behavioural replacement: two departments play the same role if they DO
    the same things. So give each department a signature - the activities that,
    in descending order of share, cumulatively cover ROLE_COVERAGE of its events
    (capped at ROLE_MAX_SIGNATURE) - and merge departments whose signatures match.

    The role's NAME is then taken from that signature, i.e. from the data. It may
    read as "Leucocytes/CRP" rather than "Laboratory", which is less elegant than
    the typed map but is the honest label: it is what the log actually supports.
    """
    if not resource:
        return {"available": False, "roles": [], "mapping": {}}

    signatures = {}
    for group, grp in df.groupby(resource):
        shares = grp[act].value_counts(normalize=True)
        chosen, cumulative = [], 0.0
        for activity, share in shares.items():
            chosen.append(str(activity))
            cumulative += float(share)
            if cumulative >= ROLE_COVERAGE or len(chosen) >= ROLE_MAX_SIGNATURE:
                break
        signatures[str(group)] = tuple(chosen)

    # Merge groups sharing a signature; order roles by how much work they carry.
    buckets: dict[tuple, list] = {}
    for group, sig in signatures.items():
        buckets.setdefault(sig, []).append(group)

    counts = df[resource].astype(str).value_counts()
    roles, mapping = [], {}
    for sig, members in buckets.items():
        events = int(sum(counts.get(m, 0) for m in members))
        label = " / ".join(sig)
        if len(members) > 1:
            label = f"{label}  (x{len(members)})"
        roles.append({"role": label, "signature": list(sig),
                      "departments": sorted(members), "events": events})
        for m in members:
            mapping[m] = label
    roles.sort(key=lambda r: -r["events"])
    return {"available": True, "n_departments": len(signatures),
            "n_roles": len(roles), "roles": roles, "mapping": mapping}


def segment_available(df, column: str, min_cohorts: int = 2) -> bool:
    """
    Is `column` usable as a cohort split on THIS log?

    The cashless-vs-self-pay comparison is the project's headline finding, but it
    rests on `payment_mode` - a column our own SimPy generator invents. No real
    hospital export has it. Anything downstream that compares cohorts therefore
    has to ask this first and degrade gracefully, rather than assume the column
    and raise KeyError on the first unseen log.
    """
    if df is None or column not in getattr(df, "columns", []):
        return False
    return int(df[column].nunique(dropna=True)) >= min_cohorts


def detect_segments(df: pd.DataFrame, case: str, exclude: set) -> list:
    """
    Case attributes worth comparing cohorts on: constant within a case, and of
    low enough cardinality to be a cohort rather than an identifier.
    """
    segments = []
    for col in df.columns:
        if col in exclude:
            continue
        try:
            per_case = df.groupby(case)[col].nunique(dropna=True)
        except Exception:                                      # noqa: BLE001
            continue
        if per_case.empty or (per_case > 1).any():
            continue                       # varies within a case -> not a segment
        values = df[col].dropna().unique()
        if SEGMENT_MIN_CARDINALITY <= len(values) <= SEGMENT_MAX_CARDINALITY:
            counts = df.drop_duplicates(case)[col].value_counts()
            segments.append({
                "attribute": str(col),
                "values": {str(k): int(v) for k, v in counts.items()},
            })
    return segments


def detect_rework(df: pd.DataFrame, case: str, act: str) -> dict:
    """Self-loops and repetition - where the Sepsis log hid 44.6% of its waiting."""
    grouped = df.groupby(case, sort=False)
    nxt = grouped[act].shift(-1)
    self_loop = (df[act] == nxt)
    per_case_repeats = df.groupby([case, act]).size()
    return {
        "self_loop_share_of_transitions": round(float(self_loop.mean()), 4),
        "top_self_looping_activities": [
            {"activity": str(a), "self_loops": int(c)}
            for a, c in df[self_loop][act].value_counts().head(5).items()],
        "share_of_case_activity_pairs_repeated": round(
            float((per_case_repeats > 1).mean()), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="path to an .xes/.xes.gz/.csv log")
    ap.add_argument("--case-col", default=DEF_CASE)
    ap.add_argument("--activity-col", default=DEF_ACT)
    ap.add_argument("--timestamp-col", default=DEF_TS)
    ap.add_argument("--lifecycle-col", default=DEF_LIFECYCLE)
    args = ap.parse_args()

    path = Path(args.log)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise SystemExit(f"Log not found: {path}")

    df = load_log(path, args.case_col, args.activity_col, args.timestamp_col)
    case, act, ts = args.case_col, args.activity_col, args.timestamp_col

    rule_line(f"PROCESS PROFILE - {path.name}")
    print(f"  events            : {len(df):,}")
    print(f"  cases             : {df[case].nunique():,}")
    print(f"  activities        : {df[act].nunique()}")
    span = df[ts].max() - df[ts].min()
    print(f"  span              : {span.days:,} days")

    caps = detect_capabilities(df, args.lifecycle_col, case)
    rule_line("1. CAPABILITIES - what can this log support?")
    print(f"  lifecycle values          : {caps['lifecycle_values'] or 'none'}")
    print(f"  separate wait from work?  : {caps['can_separate_wait_from_work']}")
    print(f"  resource/department column: {caps['resource_column'] or 'none'}")
    print(f"  measure handoffs?         : {caps['can_measure_handoffs']}")
    if caps.get("resource_column"):
        print(f"  distinct resource values  : {caps['n_resource_values']:,}")
        print(f"  resource column filled    : "
              f"{(1 - caps['resource_missing_share']) * 100:.1f}%")
        print(f"  same-resource transitions : "
              f"{caps['same_resource_transition_share']*100:.2f}%")
        print(f"  resource granularity      : {caps['resource_granularity']}")
        if not caps["handoff_split_meaningful"]:
            print()
            print("  WARNING: this column looks like INDIVIDUAL PEOPLE, not")
            print("  departments. Consecutive events are almost never the same")
            print("  actor, so an internal-vs-handoff split would read ~100%")
            print("  handoff on any process and mean nothing. Handoff ANALYSIS is")
            print("  still valid (who hands to whom); the internal/handoff SPLIT")
            print("  is not. Supply a department column to restore it.")
        if caps["resource_missing_share"] >= 0.05:
            print(f"\n  WARNING: {caps['resource_missing_share'] * 100:.1f}% of events "
                  "carry NO resource at all. Those events")
            print("  cannot be attributed to anyone, so any per-department total is a")
            print("  total over the part of the log that happens to be filled in.")
    if not caps["can_separate_wait_from_work"]:
        print("\n  NOTE: only elapsed time is measurable on this log. Queueing and")
        print("  processing cannot be separated - state that in any output.")

    ep = detect_episode_boundaries(df, case, act, ts)
    rule_line("2. EPISODE BOUNDARIES - discovered, not named")
    print(f"  reference (median) gap    : {ep['reference_gap_h']} h")
    print(f"  outlier threshold         : {ep['outlier_threshold_h']} h "
          f"({GAP_OUTLIER_FACTOR:g}x reference)")
    print(f"  episode-END activities    : {ep['episode_end_activities'] or 'none found'}")
    print(f"  episode-RESTART activities: {ep['episode_restart_activities'] or 'none found'}")
    print("\n  top terminal candidates (evidence):")
    print(f"    {'activity':<34} {'n':>7} {'term%':>7} {'outlier-gap%':>13}")
    for s in ep["evidence"]["end_candidates"][:6]:
        print(f"    {s['activity']:<34} {s['occurrences']:>7,} "
              f"{s['terminal_share']*100:>6.1f}% {s['share_followed_by_outlier_gap']*100:>12.1f}%")
    if ep["cut_applies"]:
        print("\n  => An episode cut IS needed on this log. Without it, the time")
        print("     between episodes would be counted as in-hospital waiting.")
    else:
        print("\n  => No episode boundary detected; every case looks like one")
        print("     continuous episode. No cut required.")

    roles = detect_department_roles(df, act, caps["resource_column"])
    if roles:
        rule_line("3. DEPARTMENT ROLES - by what each group does")
        print(f"    {'group':<10} {'events':>8} {'share':>7}  performs")
        for r in roles[:8]:
            top = ", ".join(f"{p['activity']} ({p['share_of_group']*100:.0f}%)"
                            for p in r["performs"][:3])
            print(f"    {r['group']:<10} {r['events']:>8,} "
                  f"{r['share_of_log']*100:>6.1f}%  {top}")

    clusters = cluster_departments_by_behaviour(df, act, caps["resource_column"])
    if clusters["available"]:
        rule_line("3b. BEHAVIOURAL ROLES - departments that do the same job, merged")
        print(f"  {clusters['n_departments']} departments -> "
              f"{clusters['n_roles']} roles (this is what the role map draws)")
        print(f"\n    {'role (named from its own activities)':<52} {'depts':>5} "
              f"{'events':>9}")
        for r in clusters["roles"][:10]:
            print(f"    {r['role'][:52]:<52} {len(r['departments']):>5} "
                  f"{r['events']:>9,}")

    exclude = {case, act, ts, args.lifecycle_col, caps["resource_column"]}
    exclude.discard(None)
    segments = detect_segments(df, case, exclude)
    rule_line("4. SEGMENTS - cohorts worth comparing")
    if segments:
        for s in segments[:6]:
            print(f"  {s['attribute']:<28} {s['values']}")
    else:
        print("  none found (no low-cardinality case-constant attribute)")

    rework = detect_rework(df, case, act)
    rule_line("5. REWORK")
    print(f"  transitions that are self-loops : "
          f"{rework['self_loop_share_of_transitions']*100:.1f}%")
    for a in rework["top_self_looping_activities"][:4]:
        print(f"    {a['activity']:<34} {a['self_loops']:>7,}")

    profile = {
        "log": path.name,
        "columns": {"case": case, "activity": act, "timestamp": ts,
                    "lifecycle": args.lifecycle_col,
                    "resource": caps["resource_column"]},
        "size": {"events": int(len(df)), "cases": int(df[case].nunique()),
                 "activities": int(df[act].nunique()), "span_days": int(span.days)},
        "capabilities": caps,
        "episodes": ep,
        "department_roles": roles,
        "behavioural_roles": clusters,
        "segments": segments,
        "rework": rework,
        "thresholds": {
            "terminal_share_min": TERMINAL_SHARE_MIN,
            "gap_outlier_factor": GAP_OUTLIER_FACTOR,
            "segment_cardinality": [SEGMENT_MIN_CARDINALITY, SEGMENT_MAX_CARDINALITY],
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    stem = path.name.split(".")[0].replace(" ", "_")
    out_path = OUT / f"18_process_profile__{stem}.json"
    out_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    rule_line("WRITTEN")
    print(f"  {out_path.relative_to(ROOT)}")
    print("\n  Nothing in this profile is named in advance. Point it at another")
    print("  hospital's log and it describes that process instead.")


if __name__ == "__main__":
    main()
