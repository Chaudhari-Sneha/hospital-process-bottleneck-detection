"""
STEP 2b - Cut the at-home time, then ask WHICH DEPARTMENT was holding the patient.

Step 2a (`02_waiting_times.py`) measured the gap between consecutive events and
ranked the arrows. It exposed two problems, and this script fixes both.

PROBLEM 1 - the hospital does not own all the elapsed time.
    The top-ranked "bottleneck" in Step 2a was `Release A -> Return ER`: a median
    of 47 days. `Release` is discharge and `Return ER` is the patient coming back,
    so that gap is the patient AT HOME. It is readmission interval, not hospital
    waiting time, and it carried ~23,800 days that swamped every real finding.

PROBLEM 3 - found while writing this script: the Step 2a ranking metric itself.
    Step 2a ranked by `median x frequency`. But 31% of gaps are exactly zero
    (batch data entry), which pins the laboratory's median at zero - and anything
    times zero is zero. `B -> B` scored 0.0 days while actually absorbing 44.6%
    of all waiting time in the log. See `summarise()` and the zero-median section
    of the output. This script ranks by the ACTUAL SUM.

    Fix: split each patient's trace into EPISODES. A new episode begins at a
    `Return ER` event, or immediately after any `Release` event. We only measure
    gaps INSIDE an episode. The gap that straddles the boundary is not
    down-weighted, it is dropped - because it is a different quantity.

PROBLEM 2 - "this arrow is slow" is not yet an instruction to anybody.
    `org:group` is 100% filled, so every event carries the department responsible.
    That lets us split every gap into two very different kinds:

      - SAME department on both sides  -> that department is holding the patient.
        A capacity or protocol problem. Talk to that department.
      - DIFFERENT departments          -> the baton is being passed and dropped.
        A coordination problem. Get both teams in one room.

    Identical minutes, different diagnosis, different person to go and see.

The department codes are anonymised single letters, so this script also DECODES
them from the data itself - by looking at which activities each group performs.

This script computes NO process map - that is the next session. Deliberately small.
"""

from pathlib import Path

import pandas as pd
import pm4py

# Standard XES column names - same as Steps 1 and 2a, plus the department field.
CASE_ID = "case:concept:name"   # which patient this event belongs to
ACTIVITY = "concept:name"       # what happened
TIMESTAMP = "time:timestamp"    # when it happened
GROUP = "org:group"             # which department was responsible

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "data" / "raw" / "Sepsis Cases - Event Log.xes.gz"
OUT_CSV = ROOT / "outputs" / "03_department_handoffs.csv"

TOP_N = 12          # how many rows to print in each ranking
MIN_FIRES = 20      # a pair must fire this often before we call it a pattern

# An episode of care ENDS at a discharge, and a new one BEGINS when the patient
# walks back in. These are the only two activities that mark the hospital's clock
# stopping and restarting.
# GENERALIZATION: episode boundaries are now DISCOVERED, not named.
#
# These two constants used to be the only thing standing between this analysis
# and a silently wrong answer on another hospital's log. They drive the episode
# cut, which on this log removed 23,815 days of at-home time and demoted a bogus
# 47-day "bottleneck" that had been topping the ranking. On a log with no
# activity called "Release", the cut became a no-op and the at-home time was
# counted as in-hospital waiting - with no error raised.
#
# `src/18_process_profile.py` finds the same activities structurally (terminal
# share + an outlier gap afterwards), so this module now asks it rather than
# being told. On the Sepsis log the discovered set is Release A/C/D/E + Return
# ER, which reproduces the hard-coded cut to within 1 gap of 297 and 0 days;
# Release B is missed because its single non-terminal occurrence is not followed
# by an outlier gap. That residual is reported rather than tuned away - tuning a
# discovery rule until it matches the development set is the memorisation this
# change exists to prevent.
#
# The fallback below applies only when profiling fails outright, and it is
# announced loudly.
FALLBACK_RELEASE_PREFIX = "Release"
FALLBACK_RETURN_ACTIVITY = "Return ER"


def rule(title: str) -> None:
    """Print a labelled separator so the output is readable."""
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def fmt_hours(hours: float) -> str:
    """Render a duration in the unit a human would actually use."""
    if pd.isna(hours):
        return "n/a"
    if hours < 1:
        return f"{hours * 60:.0f} min"
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} days"


def load_events() -> pd.DataFrame:
    """Load the log and return it sorted into per-patient chronological order."""
    if not LOG_PATH.exists():
        raise SystemExit(
            f"Event log not found at {LOG_PATH}\n"
            "Run:  python src/00_download_data.py"
        )

    print(f"Loading {LOG_PATH.name} ...")
    log = pm4py.read_xes(str(LOG_PATH))

    events = log[[CASE_ID, ACTIVITY, TIMESTAMP, GROUP]].copy()
    events[TIMESTAMP] = pd.to_datetime(events[TIMESTAMP], utc=True, format="mixed")

    # `kind="stable"` matters here exactly as it did in Step 2a: 31% of gaps are
    # zero because lab results are entered as a batch sharing one timestamp, and
    # a stable sort breaks those ties using the order the hospital system
    # recorded them in rather than shuffling them arbitrarily.
    events = events.sort_values(
        [CASE_ID, TIMESTAMP], kind="stable"
    ).reset_index(drop=True)

    return events


def load_profiler():
    """Import the profiler by path (a module name cannot start with a digit)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "profiler", ROOT / "src" / "18_process_profile.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def episode_boundaries(events: pd.DataFrame) -> tuple[list, list]:
    """
    Ask the profiler which activities end and restart an episode of care.

    Discovered structurally from this log, so the same code works on a hospital
    whose discharge step is called something else entirely. Falls back to the old
    hard-coded names ONLY if profiling fails, and says so loudly when it does -
    a silent fallback would reintroduce exactly the failure this replaced.
    """
    try:
        ep = load_profiler().detect_episode_boundaries(
            events, CASE_ID, ACTIVITY, TIMESTAMP)
        ends = ep["episode_end_activities"]
        restarts = ep["episode_restart_activities"]
    except Exception as exc:                                   # noqa: BLE001
        print(f"\n  WARNING: process profiling failed ({type(exc).__name__}: {exc}).")
        print("  Falling back to the hard-coded Sepsis activity names. This is")
        print("  CORRECT ONLY FOR THE SEPSIS LOG - on any other log the episode")
        print("  cut will not fire and at-home time will be counted as waiting.")
        ends = [a for a in events[ACTIVITY].unique()
                if str(a).startswith(FALLBACK_RELEASE_PREFIX)]
        restarts = [FALLBACK_RETURN_ACTIVITY]

    if not ends:
        print("\n  No episode boundary discovered - treating each case as one")
        print("  continuous episode. (Correct for a process patients do not leave")
        print("  and re-enter; verify that is true of this log.)")
    else:
        print(f"\n  Episode boundaries discovered from the data:")
        print(f"    ends at   : {ends}")
        print(f"    restarts at: {restarts or '(none)'}")
    return ends, restarts


def decode_departments(events: pd.DataFrame) -> None:
    """
    `org:group` is anonymised to single letters. But a department is defined by
    what it DOES, so cross-tabulating group against activity recovers the role of
    each letter without ever knowing the real hospital.
    """
    rule("DECODING THE DEPARTMENT CODES - what does each letter actually do?")
    print(
        "`org:group` is anonymised to letters. A department is defined by what it\n"
        "does, so we recover each letter's role from the activities it performs.\n"
    )

    sizes = events[GROUP].value_counts()
    print(f"{'group':>6} {'events':>8}  {'share':>6}  what it does")
    print("-" * 78)

    for group, n_events in sizes.items():
        share = n_events / len(events) * 100
        # The activities this group performs, biggest first.
        acts = events.loc[events[GROUP] == group, ACTIVITY].value_counts()
        # Keep the description short: name the activities covering most of the work.
        top = [f"{a} ({c:,})" for a, c in acts.head(4).items()]
        more = f" +{len(acts) - 4} more" if len(acts) > 4 else ""
        print(f"{group:>6} {n_events:>8,}  {share:>5.1f}%  {', '.join(top)}{more}")

    # This paragraph used to be a hand-written reading of the Sepsis codes ("the
    # group doing 100% of CRP/LacticAcid/Leucocytes is the LABORATORY", and so
    # on). It was accurate for Sepsis and pure fiction anywhere else - it printed
    # regardless of which log was loaded. It is now derived: groups performing the
    # same activities are grouped, and the role is named from those activities.
    print("\n  Read this before trusting any department ranking below. Groups that")
    print("  perform the same activities are doing the same job:")
    try:
        clusters = load_profiler().cluster_departments_by_behaviour(
            events, ACTIVITY, GROUP)
    except Exception as exc:                                   # noqa: BLE001
        print(f"    (role clustering unavailable: {exc})")
        return

    if not clusters.get("available"):
        return
    for role in clusters["roles"][:8]:
        members = role["departments"]
        shown = ", ".join(members[:4]) + (f" +{len(members) - 4} more"
                                          if len(members) > 4 else "")
        print(f"    {role['role'][:48]:<48} <- {shown}")

    # A group value that is a placeholder rather than a name means the event had
    # no responsible staff member logged. On Sepsis that is `?` on `Return ER`:
    # the patient walked back in unaided, so the event happened outside the
    # hospital's control. Worth flagging wherever it occurs, but only if it does.
    # str(g) rather than .astype(str): under pandas 3.0 the new string dtype keeps
    # missing values as NaN floats instead of the literal "nan", so .unique() can
    # hand back a float and .strip() blows up. Coerce per value instead.
    placeholders = {"?", "", "-", "NONE", "UNKNOWN", "NAN", "NA", "NAT"}
    unstaffed = [g for g in events[GROUP].unique()
                 if str(g).strip().upper() in placeholders]
    for g in unstaffed:
        # NaN never equals itself, so isin([nan]) matches nothing - select it by
        # isna() instead. Both branches are needed: '?' is a real value, NaN is not.
        mask = events[GROUP].isna() if pd.isna(g) else events[GROUP].isin([g])
        acts = events.loc[mask, ACTIVITY].unique()
        print(f"\n    Group '{g}' carries no staff identifier, and appears only on: "
              f"{', '.join(map(str, acts[:4]))}.")
        print("    No one is logged as responsible - these events happen outside")
        print("    the hospital's control, so they cannot be managed like the rest.")


def add_episodes(events: pd.DataFrame) -> pd.DataFrame:
    """
    Split each patient's trace into episodes of care.

    A patient's trace is not one continuous stay:

        ER Registration -> ... -> Release A  ||  Return ER -> CRP -> ...
                                 clock stops || clock restarts

    A new episode begins at a `Return ER` event, or on the event immediately
    AFTER a `Release`. `cumsum()` over those boundary flags gives every event an
    episode number: it stays the same until a boundary is hit, then increments.
    """
    ends, restarts = episode_boundaries(events)
    is_release = events[ACTIVITY].isin(ends)
    is_return = events[ACTIVITY].isin(restarts)

    # An event opens a new episode if it IS a return, or if the event before it
    # (same patient) was a release. `shift(1)` looks one row BACK, and grouping
    # by case stops it reaching into the previous patient's trace.
    prev_was_release = is_release.groupby(events[CASE_ID]).shift(1).fillna(False)
    opens_episode = is_return | prev_was_release

    # Counting boundaries from the top of each patient's trace numbers the episodes.
    events["episode"] = opens_episode.groupby(events[CASE_ID]).cumsum()

    return events


def add_gaps(events: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Compute the gap to the next event, but ONLY within the same episode.

    Grouping by [case, episode] instead of just [case] is the whole fix: the last
    event of an episode now has no successor, so the discharge-to-readmission gap
    is never computed at all. Returns the surviving gaps and the number of days
    the episode cut removed, so we can report what the fix was worth.
    """
    # First compute gaps the OLD way (per case) purely so we can measure what the
    # episode cut removes. This is reporting, not analysis.
    by_case = events.groupby(CASE_ID, sort=False)
    naive_delta = by_case[TIMESTAMP].shift(-1) - events[TIMESTAMP]
    naive_hours = naive_delta.dt.total_seconds() / 3600.0

    # Now the correct way: never look across an episode boundary.
    by_episode = events.groupby([CASE_ID, "episode"], sort=False)
    events["next_activity"] = by_episode[ACTIVITY].shift(-1)
    events["next_group"] = by_episode[GROUP].shift(-1)
    delta = by_episode[TIMESTAMP].shift(-1) - events[TIMESTAMP]
    events["gap_hours"] = delta.dt.total_seconds() / 3600.0

    # A gap was CUT if the naive method found one here but the episode method did not.
    cut_mask = naive_hours.notna() & events["gap_hours"].isna()
    hours_removed = naive_hours[cut_mask].sum()

    gaps = events.dropna(subset=["next_activity"]).copy()
    gaps["edge"] = gaps[ACTIVITY] + "  ->  " + gaps["next_activity"]
    gaps["dept_pair"] = gaps[GROUP] + "  ->  " + gaps["next_group"]
    gaps["is_handoff"] = gaps[GROUP] != gaps["next_group"]

    rule("THE EPISODE CUT - how much at-home time did we just remove?")
    print(f"Episodes found (was {events[CASE_ID].nunique():,} patients) : "
          f"{events.groupby([CASE_ID, 'episode']).ngroups:,}")
    print(f"Gaps before the cut                        : {naive_hours.notna().sum():,}")
    print(f"Gaps after the cut                         : {len(gaps):,}")
    print(f"Gaps removed (they straddled a discharge)  : {cut_mask.sum():,}")
    print(f"Time removed with them                     : {hours_removed / 24:,.0f} days")
    print(
        f"\n  Only {cut_mask.sum():,} of {naive_hours.notna().sum():,} gaps "
        f"({cut_mask.sum() / naive_hours.notna().sum() * 100:.1f}%) were cut - but they "
        f"carried\n  {hours_removed / 24:,.0f} days with them. That is why the Step 2a "
        "ranking was dominated by a\n  delay nobody in the hospital could do anything "
        "about.\n"
        "\n  Lesson worth keeping: an event log records ELAPSED time. Only domain\n"
        "  knowledge tells you which elapsed time the organisation is responsible for."
    )

    return gaps, hours_removed


def print_ranking(stats: pd.DataFrame, title: str, blurb: str, label: str) -> None:
    """Print one ranked table. Used for the activity ranking and the dept ranking."""
    rule(title)
    print(blurb)
    header = (f"{'#':>3}  {label:<46} {'fires':>6} {'median':>9} "
              f"{'p90':>9} {'TOTAL':>10}")
    print(header)
    print("-" * len(header))
    for i, (name, row) in enumerate(stats.head(TOP_N).iterrows(), start=1):
        print(
            f"{i:>3}. {name:<46} {int(row['count']):>6,} "
            f"{fmt_hours(row['median_h']):>9} {fmt_hours(row['p90_h']):>9} "
            f"{fmt_hours(row['total_hours']):>10}"
        )


def summarise(gaps: pd.DataFrame, key: str) -> pd.DataFrame:
    """
    Aggregate gaps by any key (activity edge or department pair).

    CORRECTION TO STEP 2a. That script ranked by `median x frequency`, and that
    metric is wrong. 31% of all gaps are exactly zero because lab results are
    entered as a batch, which drags the LAB's median to zero - and anything
    multiplied by zero is zero. So `B -> B` (lab to lab) scored 0.0 days in the
    ranking while actually absorbing 2,711 days, 44.6% of all waiting time in
    the log. The metric silently deleted the single largest block of time.

    `median x frequency` is neither a robust statistic nor a true total. So:

      - `total_hours` is now the ACTUAL SUM of the gaps. It ranks the rows,
        because it answers the administrator's question: if I fix ONE thing, how
        many hours do I really get back?
      - `median_h` stays, but as a DESCRIPTION - what a typical patient waits.
      - `p90_h` is the unlucky patient. Where p90 sits far above the median the
        distribution is bimodal, and the two halves need separate explanations.

    Keeping all three is the point: a median of 0 next to a p90 of 48 h is not a
    contradiction, it is the finding.
    """
    stats = gaps.groupby(key)["gap_hours"].agg(
        count="size",
        median_h="median",
        p90_h=lambda s: s.quantile(0.90),
        total_hours="sum",
    )
    return stats.sort_values("total_hours", ascending=False)


def print_zero_median_trap(gaps: pd.DataFrame) -> None:
    """
    Show WHY the Step 2a ranking metric had to be corrected, using the row it
    broke on. A median of zero does not mean no time is being spent - it means
    more than half the gaps are batch-entry artefacts, and the real waiting is
    hiding in the other half.
    """
    stats = summarise(gaps, "dept_pair")
    zero_median = stats[stats["median_h"] == 0]
    if zero_median.empty:
        return

    pair = zero_median.index[0]          # the biggest zero-median pair
    rows = gaps[gaps["dept_pair"] == pair]["gap_hours"]
    zeros = rows[rows == 0]
    real = rows[rows > 0]

    rule("THE ZERO-MEDIAN TRAP - why the Step 2a ranking metric was wrong")
    print(
        f"Step 2a ranked by `median x frequency`. Look at what that does to "
        f"`{pair}`:\n"
    )
    print(f"  gaps on this pair                : {len(rows):,}"
          f"  ({len(rows) / len(gaps) * 100:.1f}% of ALL gaps in the log)")
    print(f"  of those, exactly zero           : {len(zeros):,}"
          f"  ({len(zeros) / len(rows) * 100:.1f}%)  <- batch data entry")
    print(f"  median                           : {fmt_hours(rows.median())}")
    print(f"  Step 2a score (median x count)   : {rows.median() * len(rows) / 24:,.1f} days")
    print(f"  ACTUAL time absorbed (sum)       : {rows.sum() / 24:,.1f} days"
          f"  ({rows.sum() / gaps['gap_hours'].sum() * 100:.1f}% of all waiting)")
    print()
    print(f"  ignoring the batch-entry zeros, the {len(real):,} real gaps look like:")
    print(f"    median                         : {fmt_hours(real.median())}")
    print(f"    90th percentile                : {fmt_hours(real.quantile(0.90))}")
    print(
        "\n  So the distribution is BIMODAL, and the median alone describes neither\n"
        "  half. Half these rows are results typed into the system together - a data\n"
        "  entry artefact worth zero minutes of anyone's attention. The other half is\n"
        "  a genuine wait for the next blood draw, and it is the largest single block\n"
        "  of time in the entire hospital process.\n"
        "\n  The ranking below therefore uses the ACTUAL SUM, with median and p90 kept\n"
        "  as description. A median of 0 next to a p90 in days is not a contradiction\n"
        "  to be smoothed away - it IS the finding."
    )


def same_resource_transition_share(gaps: pd.DataFrame) -> float:
    """
    Fraction of consecutive event pairs handled by the SAME resource value.

    This is the granularity test: a department-level column scores high (the
    same unit often performs two steps in a row), a person-level column scores
    near zero. Same definition the profiler uses, so the two agree.
    """
    return float((~gaps["is_handoff"]).mean()) if len(gaps) else 0.0


def print_handoff_split(gaps: pd.DataFrame) -> None:
    """
    The headline department finding: is the time lost INSIDE departments, or
    BETWEEN them? These point at completely different fixes.
    """
    rule("INTERNAL WAIT vs. HANDOFF WAIT - where is the time actually lost?")

    # Does the resource column name DEPARTMENTS or PEOPLE? The split is only
    # interpretable for the former. Holdout testing on the Hospital Billing log
    # showed this failing silently: `org:resource` there is 1,150 individuals,
    # so 100.0% of waiting was reported as "handoff" - a confident number that
    # meant nothing. Refuse to print it rather than let it be quoted.
    share = same_resource_transition_share(gaps)
    if share < 0.05:
        print(
            f"  NOT REPORTED. The `{GROUP}` column looks like individual people,\n"
            f"  not departments: only {share * 100:.2f}% of consecutive event pairs\n"
            f"  share an actor (a department-level column scores ~50%). With one\n"
            f"  person per event, essentially every pair is a different actor, so\n"
            f"  this split would read ~100% handoff on ANY process and carry no\n"
            f"  information. The handoff RANKING below is still valid - who hands\n"
            f"  to whom, and how long it costs. Only the internal/handoff SPLIT is\n"
            f"  withheld. Supply a department-level column to restore it."
        )
        return

    total_all = (gaps["gap_hours"]).sum()
    for is_handoff, label in [(False, "INTERNAL (same department both sides)"),
                              (True, "HANDOFF  (baton passed between departments)")]:
        subset = gaps[gaps["is_handoff"] == is_handoff]
        hours = subset["gap_hours"].sum()
        print(
            f"{label:<46} {len(subset):>6,} gaps  "
            f"{hours / 24:>8,.0f} days  ({hours / total_all * 100:>4.1f}% of all waiting)"
        )

    print(
        "\n  These are different management problems with different owners:\n"
        "  - INTERNAL time is a capacity or protocol problem. One department controls\n"
        "    it, and the fix is their schedule, their staffing, their repeat policy.\n"
        "  - HANDOFF time is a coordination problem. No single department owns it,\n"
        "    which is exactly why it persists - and why it needs both teams in a room.\n"
        "  An activity-level ranking cannot tell these apart. This is the added value\n"
        "  of `org:group` and the reason Step 1 flagged it as a useful surprise."
    )


def configure_from_args() -> None:
    """
    Point this analysis at ANY event log.

    Three things were hard-coded to the Sepsis dataset: the file path, the
    department column (`org:group`), and the output name. The holdout log uses
    `org:resource` instead, so the column cannot be assumed either - it is
    auto-detected from the same candidate list the profiler uses, and can be
    overridden.

    Reassigning module globals is blunt, but it is the SMALLEST change that
    makes every existing function work unmodified on a new log. Restructuring
    the module to thread a config object through would touch every function and
    risk the validated Step 2 numbers for no analytical gain.
    """
    global LOG_PATH, OUT_CSV, GROUP

    import argparse
    ap = argparse.ArgumentParser(description="Department handoff analysis")
    ap.add_argument("--log", default=None,
                    help="event log to analyse (default: the Sepsis log)")
    ap.add_argument("--group-col", default=None,
                    help="department/resource column; auto-detected if omitted")
    ap.add_argument("--out", default=None, help="output CSV name")
    args = ap.parse_args()

    if args.log:
        p = Path(args.log)
        LOG_PATH = p if p.is_absolute() else ROOT / p
        stem = LOG_PATH.name.split(".")[0].replace(" ", "_")
        OUT_CSV = ROOT / "outputs" / f"03_department_handoffs__{stem}.csv"
    if args.out:
        # Accept both conventions, because callers use both. `--out report.csv`
        # means "put it in outputs/"; `--out outputs/report.csv` is a path from
        # the project root. Unconditionally prefixing "outputs/" turned the second
        # form into `outputs/outputs/report.csv` - which is where src/19's results
        # were quietly landing.
        out = Path(args.out)
        if out.is_absolute():
            OUT_CSV = out
        elif out.parent == Path("."):
            OUT_CSV = ROOT / "outputs" / out
        else:
            OUT_CSV = ROOT / out

    if args.group_col:
        GROUP = args.group_col
    elif args.log:
        # Auto-detect: the Sepsis log uses org:group, Hospital Billing uses
        # org:resource. Assuming either one silently breaks the other.
        import pm4py
        probe = (pm4py.read_xes(str(LOG_PATH)) if LOG_PATH.suffix.lower() in
                 {".xes", ".gz"} else pd.read_csv(LOG_PATH, nrows=200))
        for candidate in ["org:group", "org:resource", "resource", "department"]:
            if candidate in probe.columns:
                GROUP = candidate
                break
        else:
            raise SystemExit(
                f"No department/resource column found in {LOG_PATH.name}.\n"
                f"  Columns: {list(probe.columns)[:12]}\n"
                "  Pass --group-col to name it, or use a log that has one.")
        print(f"  department column auto-detected: {GROUP}")


def main() -> None:
    configure_from_args()
    events = load_events()
    decode_departments(events)

    events = add_episodes(events)
    gaps, _ = add_gaps(events)

    print_zero_median_trap(gaps)

    # ---- The corrected activity ranking (compare this against Step 2a) --------
    print_ranking(
        summarise(gaps, "edge"),
        f"CORRECTED ACTIVITY RANKING - top {TOP_N} (at-home time removed, ranked by\n"
        "actual total)",
        # This used to name `Release A -> Return ER` outright - true of Sepsis, and
        # printed on every log regardless. The two corrections are general; the
        # example that illustrates them is not, so it is described, not named.
        "Two corrections against Step 2a. Any edge that straddled an episode\n"
        "boundary is gone, because that gap is no longer computed at all - it was\n"
        "the patient at home rather than waiting. And the ranking is now the real\n"
        "sum, not median x count, so batch-entry rows can no longer score zero.\n"
        "What is left is in-hospital time, honestly totalled.\n",
        "activity edge",
    )

    # ---- The department view -------------------------------------------------
    print_handoff_split(gaps)

    dept_stats = summarise(gaps, "dept_pair")
    print_ranking(
        dept_stats[dept_stats["count"] >= MIN_FIRES],
        f"DEPARTMENT RANKING - top {TOP_N} pairs by ACTUAL total time absorbed",
        f"Pairs firing at least {MIN_FIRES} times, so these are patterns rather than\n"
        "anecdotes. `X -> X` means one department holding the patient; `X -> Y` means\n"
        "a handoff between two.\n",
        "department pair (from -> to)",
    )

    handoffs_only = dept_stats[
        dept_stats.index.map(lambda p: p.split("  ->  ")[0] != p.split("  ->  ")[1])
    ]
    print_ranking(
        handoffs_only[handoffs_only["count"] >= MIN_FIRES],
        f"HANDOFFS ONLY - top {TOP_N} places the baton gets dropped",
        ("Same-department rows stripped out, so every row here is a coordination\n"
         "problem between two teams - the kind no single department owns.\n"
         if same_resource_transition_share(gaps) >= 0.05 else
         f"`{GROUP}` names individuals, not departments, so each row is a pass\n"
         "between two PEOPLE. Read it as who-waits-on-whom, not as a team-level\n"
         "coordination failure - the same two names may sit in one department.\n"),
        "department pair (from -> to)",
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    dept_stats.to_csv(OUT_CSV)
    rule("STEP 2b COMPLETE")
    print(f"Full department table written to: {OUT_CSV.relative_to(ROOT)}")
    print(
        "Next session (Step 2c): cross-check 2-3 SPECIFIC published numbers about\n"
        "this Sepsis log against what we just computed. That is what turns 'my\n"
        "pipeline produced a number' into 'my pipeline produced the RIGHT number',\n"
        "and it is the falsifiable validation the project plan requires before we\n"
        "are allowed to trust this pipeline on synthetic Indian data in Step 3."
    )


if __name__ == "__main__":
    main()
