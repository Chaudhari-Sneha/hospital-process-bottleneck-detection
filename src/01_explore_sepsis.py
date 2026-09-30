"""
STEP 1 - Look at what is actually inside the Sepsis event log.

Deliberately small. This script computes NO bottlenecks. Its only job is to answer
four questions before we trust any analysis built on top of this data:

  1. How many cases (patients) and how many distinct activities are there?
     -> Does the file match the 1,000 cases / 15,000 events / 16 activities the
        official readme promises? If not, something is wrong with our loading.

  2. What columns exist, and is there a `lifecycle:transition` column?
     -> This decides whether we can separate WAITING time from PROCESSING time.
        If every event has only one timestamp, we can only measure the gap
        between events, not how long the work itself took. This single fact
        shapes the whole rest of the project.

  3. What period does the log cover?
     -> The 4TU readme says timestamps were randomised for anonymity, but that
        the time BETWEEN events inside a case was left untouched. So absolute
        dates are meaningless; durations within a case are trustworthy. Good
        news for us: bottleneck analysis only needs durations.

  4. How many events per case (min / median / max)?
     -> Tells us whether patients follow a short standard path or a long messy
        one, and hints at how much rework/looping exists.
"""

from pathlib import Path

import pandas as pd
import pm4py

# Standard XES column names. XES is the international standard format for event
# logs, so these names are the same in almost any process-mining dataset.
CASE_ID = "case:concept:name"   # which patient/case this event belongs to
ACTIVITY = "concept:name"       # what happened
TIMESTAMP = "time:timestamp"    # when it happened

LOG_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "raw" / "Sepsis Cases - Event Log.xes.gz"
)


def rule(title: str) -> None:
    """Print a labelled separator so the output is readable."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    if not LOG_PATH.exists():
        raise SystemExit(
            f"Event log not found at {LOG_PATH}\n"
            "Run:  python src/00_download_data.py"
        )

    # pm4py reads the .xes.gz directly - no need to unzip it first.
    # In recent pm4py versions this returns an ordinary pandas DataFrame,
    # one row per EVENT (not one row per patient).
    print(f"Loading {LOG_PATH.name} ...")
    log = pm4py.read_xes(str(LOG_PATH))

    # ---- Q1: size of the log -------------------------------------------------
    rule("Q1. HOW BIG IS THIS LOG?")
    n_events = len(log)
    n_cases = log[CASE_ID].nunique()
    n_activities = log[ACTIVITY].nunique()

    print(f"Events (rows in the file)      : {n_events:,}")
    print(f"Cases (individual patients)    : {n_cases:,}")
    print(f"Distinct activities            : {n_activities}")
    print("\nOfficial 4TU readme claims     : ~1,000 cases / 15,000 events / 16 activities")

    # ---- The 16 activities, most frequent first ------------------------------
    rule("THE ACTIVITIES - the steps a sepsis patient goes through")
    counts = log[ACTIVITY].value_counts()
    for i, (activity, count) in enumerate(counts.items(), start=1):
        share = count / n_events * 100
        print(f"{i:>3}. {activity:<28} {count:>7,} events  ({share:>5.1f}%)")

    # ---- Q2: what columns do we have? ---------------------------------------
    rule("Q2. WHAT COLUMNS EXIST? (can we split waiting time from work time?)")
    print(f"Total columns: {len(log.columns)}\n")
    for col in sorted(log.columns):
        non_null = log[col].notna().sum()
        fill = non_null / n_events * 100
        print(f"  {col:<38} {fill:>5.1f}% filled")

    has_lifecycle = "lifecycle:transition" in log.columns
    print()
    if has_lifecycle:
        values = log["lifecycle:transition"].dropna().unique()
        print(f"`lifecycle:transition` IS present. Values: {list(values)}")
        if len(values) <= 1:
            print(
                "  BUT there is only ONE value, so every event is a single point in\n"
                "  time. We still CANNOT separate waiting from processing."
            )
    else:
        print("`lifecycle:transition` is NOT present.")

    print(
        "\n  => Consequence for this project: with one timestamp per event we can\n"
        "     only measure the GAP between consecutive activities. That gap mixes\n"
        "     queueing and actual work together.\n"
        "     This is exactly why our SimPy simulator (Step 3) will record BOTH a\n"
        "     start and a complete timestamp - the synthetic Indian log will be\n"
        "     richer than the real one, which is a real justification for building it."
    )

    # ---- Q3: time period -----------------------------------------------------
    rule("Q3. WHAT PERIOD DOES THE LOG COVER?")
    ts = pd.to_datetime(log[TIMESTAMP], utc=True, format="mixed")
    print(f"Earliest event : {ts.min()}")
    print(f"Latest event   : {ts.max()}")
    print(f"Span           : {(ts.max() - ts.min()).days:,} days")
    print(
        "\n  Note: 4TU randomised these timestamps for anonymity, but kept the time\n"
        "  BETWEEN events within a patient's trace unchanged. So ignore the calendar\n"
        "  dates - the durations inside a case are the trustworthy part, and that is\n"
        "  all bottleneck analysis needs."
    )

    # ---- Q4: events per case -------------------------------------------------
    rule("Q4. HOW MANY EVENTS PER PATIENT?")
    per_case = log.groupby(CASE_ID).size()
    print(f"Minimum  : {per_case.min()} events")
    print(f"Median   : {per_case.median():.0f} events")
    print(f"Mean     : {per_case.mean():.1f} events")
    print(f"Maximum  : {per_case.max()} events")
    print(
        f"\n  A {per_case.max()}-event patient against a median of {per_case.median():.0f} "
        "means some patients loop\n  through the process many times over. Those loops are "
        "where time gets lost -\n  and finding them is the point of Step 2."
    )

    rule("STEP 1 COMPLETE")
    print("The log loads, and we know what we are holding. Next: Step 2, find the bottlenecks.")


if __name__ == "__main__":
    main()
