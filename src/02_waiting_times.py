"""
STEP 2a - Where does the clock actually run between activities?

  *** SUPERSEDED BY `03_department_handoffs.py`. Kept because the two mistakes
      it makes are worth being able to point at. Do not quote its ranking. ***

      1. It measures the gap from discharge to readmission - that is the patient
         at HOME, ~23,800 days of it, and it dominates the ranking.
      2. It ranks by `median x frequency`. 31% of gaps are exactly zero (batch
         data entry), which pins the laboratory's median at zero, so `B -> B`
         scores 0.0 days while really absorbing 44.6% of all waiting time.

      Step 2b fixes both: it cuts the trace at discharge, and ranks by actual sum.

Step 1 proved every event carries ONE timestamp (`lifecycle:transition` is 100%
filled but always `complete`). So the only honest measurement available in this
log is the GAP between two consecutive events inside one patient's trace.

This script does exactly three things:

  1. Computes that gap for every pair of consecutive events, per patient.

  2. Attributes each gap to the EDGE it sits on (from_activity -> to_activity),
     not to a single activity. This is the process-mining move: the same
     activity can be preceded by different steps and wait a very different
     amount of time in each case.

  3. Ranks the edges by TOTAL time absorbed (median gap x how often the edge
     fires), not by longest median. A 6-hour delay that happens 4 times costs
     24 hours. A 12-minute delay that happens 3,000 times costs 600 hours.
     The second one is the bottleneck worth an administrator's attention.

It also prints the "dashboard view" (one average wait per activity) next to the
edge view, so the difference between them is visible in our own data rather
than merely asserted in the report.

This script computes NO process map and NO department handoffs - those are the
next two sessions. Deliberately small.
"""

from pathlib import Path

import pandas as pd
import pm4py

# Standard XES column names - same as Step 1.
CASE_ID = "case:concept:name"   # which patient this event belongs to
ACTIVITY = "concept:name"       # what happened
TIMESTAMP = "time:timestamp"    # when it happened

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "data" / "raw" / "Sepsis Cases - Event Log.xes.gz"
OUT_CSV = ROOT / "outputs" / "02_transition_waits.csv"

TOP_N = 15        # how many rows to print in each ranking
MIN_FIRES = 20    # ignore incoming paths rarer than this in the contrast section


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

    # Keep only the three columns this step needs, so nothing else can confuse us.
    events = log[[CASE_ID, ACTIVITY, TIMESTAMP]].copy()
    events[TIMESTAMP] = pd.to_datetime(events[TIMESTAMP], utc=True, format="mixed")

    # Sort by patient, then by time. `kind="stable"` matters: many Sepsis events
    # share an identical timestamp (lab results entered as a batch), and a stable
    # sort breaks those ties using the order they appear in the file, which is
    # the order the hospital system recorded them in. An unstable sort would
    # shuffle them arbitrarily and invent transitions that never happened.
    events = events.sort_values(
        [CASE_ID, TIMESTAMP], kind="stable"
    ).reset_index(drop=True)

    return events


def add_gaps(events: pd.DataFrame) -> pd.DataFrame:
    """
    For every event, look at the NEXT event of the same patient and record:
      - which activity comes next   (`next_activity`)
      - how long the clock ran      (`gap_hours`)

    `groupby(...).shift(-1)` pulls the following row's value UP onto this row,
    and the groupby makes sure it never reaches across from one patient into the
    next patient. The final event of each patient has nothing after it, so its
    gap is NaT/NaN - that is correct, not missing data, and we drop those rows.
    """
    grouped = events.groupby(CASE_ID, sort=False)

    events["next_activity"] = grouped[ACTIVITY].shift(-1)
    events["next_timestamp"] = grouped[TIMESTAMP].shift(-1)

    delta = events["next_timestamp"] - events[TIMESTAMP]
    events["gap_hours"] = delta.dt.total_seconds() / 3600.0

    n_before = len(events)
    gaps = events.dropna(subset=["next_activity"]).copy()

    # An edge is one arrow in the process map: "this step -> that step".
    gaps["edge"] = gaps[ACTIVITY] + "  ->  " + gaps["next_activity"]

    print(
        f"\n{n_before:,} events produce {len(gaps):,} gaps "
        f"({n_before - len(gaps):,} patients' final events have nothing after them)."
    )
    return gaps


def rank_edges(gaps: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the gaps per edge.

    We use the MEDIAN, not the mean. Step 1 showed a long right tail (median 13
    events per patient, max 185), and a handful of extreme patients would drag
    every mean upwards and make ordinary edges look broken.

    `total_hours` = median x frequency = the time this edge absorbs across the
    whole log. This is the number that decides the ranking, because it answers
    the administrator's real question: if I fix ONE arrow, how many hours do I
    get back?
    """
    stats = gaps.groupby("edge")["gap_hours"].agg(
        count="size",
        median_h="median",
        p90_h=lambda s: s.quantile(0.90),
        max_h="max",
    )
    stats["total_hours"] = stats["median_h"] * stats["count"]
    return stats.sort_values("total_hours", ascending=False)


def print_ranking_table(stats: pd.DataFrame, title: str, blurb: str) -> None:
    """Print one ranked table of edges. Used for both rankings below."""
    rule(title)
    print(blurb)
    header = f"{'#':>3}  {'edge':<52} {'fires':>6} {'median':>9} {'p90':>9} {'TOTAL':>10}"
    print(header)
    print("-" * len(header))

    for i, (edge, row) in enumerate(stats.head(TOP_N).iterrows(), start=1):
        print(
            f"{i:>3}. {edge:<52} {int(row['count']):>6,} "
            f"{fmt_hours(row['median_h']):>9} {fmt_hours(row['p90_h']):>9} "
            f"{fmt_hours(row['total_hours']):>10}"
        )


def print_dashboard_contrast(gaps: pd.DataFrame) -> None:
    """
    The "why not just use Power BI?" evidence.

    A dashboard reads the table column-wise and gives ONE average wait per
    activity. Process mining reads it row-sequence-wise and gives one wait per
    INCOMING PATH. If an activity's waits differ wildly depending on which step
    preceded it, the dashboard's single number is an average of two different
    problems - and hides both.
    """
    rule("THE POWER BI CONTRAST - one number per activity vs. one per incoming path")

    # The dashboard view: attribute each gap to the activity it leads INTO.
    per_activity = gaps.groupby("next_activity")["gap_hours"].agg(
        arrivals="size", median_h="median"
    ).sort_values("median_h", ascending=False)

    print("DASHBOARD VIEW - median wait before each activity:\n")
    print(f"  {'activity':<28} {'arrivals':>9} {'median wait':>13}")
    print("  " + "-" * 52)
    for activity, row in per_activity.iterrows():
        print(
            f"  {activity:<28} {int(row['arrivals']):>9,} "
            f"{fmt_hours(row['median_h']):>13}"
        )

    # Now break activities apart by which step preceded them. We only consider
    # incoming paths that fire at least MIN_FIRES times, so we are comparing
    # real routes rather than one-off outliers. We report the activity whose
    # incoming paths disagree with each other the MOST, because that is the one
    # the dashboard's single average distorts worst.
    best_activity, best_split, best_spread = None, None, 0.0
    for activity in per_activity.index:
        incoming = gaps[gaps["next_activity"] == activity]
        split = incoming.groupby(ACTIVITY)["gap_hours"].agg(
            fires="size", median_h="median"
        )
        split = split[split["fires"] >= MIN_FIRES].sort_values(
            "median_h", ascending=False
        )
        if len(split) < 2:
            continue
        spread = split["median_h"].iloc[0] - split["median_h"].iloc[-1]
        if spread > best_spread:
            best_activity, best_split, best_spread = activity, split, spread

    if best_activity is None:
        print("\n  No activity has 2+ incoming paths above the frequency floor.")
        return

    dashboard_number = per_activity.loc[best_activity, "median_h"]
    print(
        f"\nPROCESS-MINING VIEW - the single '{best_activity}' number, split by\n"
        f"which step the patient arrived from "
        f"(paths firing at least {MIN_FIRES} times):\n"
    )
    print(f"  {'arrived from':<28} {'fires':>9} {'median wait':>13}")
    print("  " + "-" * 52)
    for prev_activity, row in best_split.iterrows():
        print(
            f"  {prev_activity:<28} {int(row['fires']):>9,} "
            f"{fmt_hours(row['median_h']):>13}"
        )

    slowest, fastest = best_split.iloc[0], best_split.iloc[-1]
    print(
        f"\n  => The dashboard reports ONE number for '{best_activity}': "
        f"{fmt_hours(dashboard_number)}.\n"
        f"     But patients arriving from '{best_split.index[0]}' wait "
        f"{fmt_hours(slowest['median_h'])}, while those\n"
        f"     arriving from '{best_split.index[-1]}' wait "
        f"{fmt_hours(fastest['median_h'])}. The delay is path-dependent, so the\n"
        "     single average is an average of two different problems and hides both.\n"
        "     THIS is the answer to 'why not just build a Power BI dashboard?'"
    )


def main() -> None:
    events = load_events()
    gaps = add_gaps(events)
    stats = rank_edges(gaps)

    rule("SANITY CHECK - are these gaps believable?")
    print(f"Distinct edges observed        : {len(stats):,}")
    print(
        f"Gaps of exactly zero           : {(gaps['gap_hours'] == 0).sum():,}"
        f"  ({(gaps['gap_hours'] == 0).mean() * 100:.1f}%)"
    )
    print(f"Negative gaps (must be zero)   : {(gaps['gap_hours'] < 0).sum():,}")
    print(f"Longest single gap in the log  : {fmt_hours(gaps['gap_hours'].max())}")
    print(
        "\n  Zero-length gaps are expected, not a bug: lab results are entered into\n"
        "  the system in a batch, so several events share one timestamp. A median of\n"
        "  0 on a busy edge means 'recorded together', NOT 'the process is instant'.\n"
        "  Negative gaps would mean our sort is wrong - that count must stay at 0."
    )

    print_ranking_table(
        stats,
        f"THE BOTTLENECK RANKING - top {TOP_N} edges by TOTAL time absorbed",
        "Ranked by median gap x how often the edge fires. The p90 column shows the\n"
        "unlucky patient's experience: if p90 is far above the median, the delay is\n"
        "not a steady queue, it is an occasional stall worth explaining separately.\n",
    )

    print_ranking_table(
        stats.sort_values("median_h", ascending=False),
        f"FOR CONTRAST - top {TOP_N} edges by LONGEST MEDIAN (the tempting ranking)",
        "This is what sorting by 'worst delay' gives you. Check the `fires` column:\n"
        "if these edges are rare, this ranking is a list of anecdotes, not a plan.\n",
    )

    print_dashboard_contrast(gaps)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(OUT_CSV)
    rule("STEP 2a COMPLETE")
    print(f"Full edge table written to: {OUT_CSV.relative_to(ROOT)}")
    print(
        "Next session (Step 2b): the same gaps grouped by `org:group` - which\n"
        "DEPARTMENT was holding the patient when the clock ran. That turns\n"
        "'this arrow is slow' into 'this handoff between these two teams is slow'."
    )


if __name__ == "__main__":
    main()
