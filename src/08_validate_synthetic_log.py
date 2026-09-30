"""
STEP 4 (validation) - Is the generated synthetic log actually well-formed?

The generator ran without crashing. That proves nothing. A simulator can emit a
log that is internally contradictory - a `start` before its `schedule`, a patient
discharged twice, a self-pay patient carrying insurance activities - and every
downstream analysis would then be measuring an artefact.

So this script checks the log against what it CLAIMS to be, before anyone runs a
bottleneck analysis on it. Same discipline as the Step 2c literature gate: state
the property, test it, and report a failure as a failure.

The checks, and why each one earns its place:

  SCHEMA        - the columns the Step 2 pipeline needs are present and named
                  exactly as in the real log. If this fails, nothing else matters.

  LIFECYCLE     - every activity instance has exactly one schedule, one start and
                  one complete. A missing `complete` would silently drop work from
                  every processing-time calculation.

  ORDERING      - schedule <= start <= complete, always. A violation means the
                  simulation clock went backwards, and every duration is suspect.

  TRACE SHAPE   - each case starts with Registration and ends with Discharge, and
                  admitted patients pass Admission -> Clinically Fit -> Discharge
                  in that order. This is what makes the log a PROCESS rather than
                  a bag of events.

  PAYMENT LOGIC - payment_mode is constant within a case, cashless patients carry
                  authorisation activities, self-pay patients carry billing, and
                  neither carries the other's. This is the tag the whole Step 5
                  comparison is built on, so it has to be airtight.

  WARM-UP       - the first patients arrive to a completely empty hospital and
                  wait for nothing. Reported, not fixed: it is a property of any
                  terminating simulation and the honest response is to know how
                  big it is.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "data" / "synthetic" / "indian_hospital_log.csv"

CASE = "case:concept:name"
ACT = "concept:name"
INST = "concept:instance"
TS = "time:timestamp"
LIFECYCLE = "lifecycle:transition"
GROUP = "org:group"
MODE = "payment_mode"

REQUIRED_COLUMNS = [CASE, ACT, INST, TS, LIFECYCLE, GROUP, MODE]

CASHLESS_ONLY = {"Preauth Request", "Preauth Review", "Preauth Document Query",
                 "Preauth Approved", "Discharge Auth Request",
                 "Discharge Auth Review", "Discharge Document Query",
                 "Discharge Authorized"}
SELF_PAY_ONLY = {"Billing"}


class Checks:
    """Collects pass/fail rows so the script ends with one readable verdict."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((name, ok, detail))

    def report(self) -> int:
        print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
        failed = 0
        for name, ok, detail in self.rows:
            flag = "ok  " if ok else "FAIL"
            if not ok:
                failed += 1
            print(f"  [{flag}] {name}")
            if detail:
                print(f"         {detail}")
        print(f"\n  {len(self.rows) - failed} of {len(self.rows)} checks passed.")
        return failed


def rule_line(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    if not LOG_PATH.exists():
        raise SystemExit(
            f"Synthetic log not found: {LOG_PATH}\n"
            "Run:  python src/07_generate_synthetic_log.py "
            "--patients 3000 --seed 42 --out indian_hospital_log.csv"
        )

    df = pd.read_csv(LOG_PATH, parse_dates=[TS])
    chk = Checks()

    # ---- basic shape --------------------------------------------------------
    rule_line("WHAT WAS GENERATED")
    n_events = len(df)
    n_cases = df[CASE].nunique()
    print(f"  events (rows)          : {n_events:,}")
    print(f"  cases (patients)       : {n_cases:,}")
    # NOTE: count (case, instance) pairs, not distinct instance strings. The
    # instance id is only unique WITHIN a case ("Lab Test#1" recurs in every
    # admission), so df[INST].nunique() would report 55 rather than ~33,000.
    print(f"  activity instances     : "
          f"{df.groupby([CASE, INST]).ngroups:,}")
    print(f"  distinct activities    : {df[ACT].nunique()}")
    print(f"  departments            : {df[GROUP].nunique()} "
          f"{sorted(df[GROUP].unique())}")
    print(f"  events per case        : min {df.groupby(CASE).size().min()}, "
          f"median {df.groupby(CASE).size().median():.0f}, "
          f"max {df.groupby(CASE).size().max()}")
    print(f"  span                   : {df[TS].min()}  ->  {df[TS].max()}")
    print(f"                           ({(df[TS].max() - df[TS].min()).days} days)")

    manifest_path = LOG_PATH.with_suffix(".manifest.json")
    if manifest_path.exists():
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"\n  manifest: seed={man['seed']}, patients={man['patients']}, "
              f"document_query_probability={man['document_query_probability']}"
              "  <- ASSUMPTION, unsourced")
        chk.add("manifest present alongside the log", True)
    else:
        chk.add("manifest present alongside the log", False,
                "a log without its assumptions is not evidence of anything")

    # ---- 1. schema ----------------------------------------------------------
    rule_line("1. SCHEMA - can the Step 2 pipeline read this?")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    chk.add("required columns present", not missing,
            f"missing: {missing}" if missing else
            "matches the real log's column names, so the Step 2 scripts apply")
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            nulls = int(df[col].isna().sum())
            print(f"  {col:<24} {'OK' if nulls == 0 else f'{nulls} NULLS'}")
    chk.add("no nulls in required columns",
            all(df[c].isna().sum() == 0 for c in REQUIRED_COLUMNS if c in df.columns))

    # ---- 2. lifecycle completeness -----------------------------------------
    rule_line("2. LIFECYCLE - does every activity instance have all three events?")
    per_instance = df.groupby([CASE, INST])[LIFECYCLE].agg(
        lambda s: tuple(sorted(s)))
    expected = ("complete", "schedule", "start")
    bad = per_instance[per_instance != expected]
    print(f"  activity instances           : {len(per_instance):,}")
    print(f"  with exactly schedule/start/complete : "
          f"{len(per_instance) - len(bad):,}")
    print(f"  lifecycle values present     : {sorted(df[LIFECYCLE].unique())}")
    chk.add("every instance has schedule+start+complete", len(bad) == 0,
            f"{len(bad):,} malformed instances" if len(bad) else
            "so processing time is computable for every single step")

    # ---- 3. temporal ordering ----------------------------------------------
    rule_line("3. ORDERING - schedule <= start <= complete?")
    wide = df.pivot_table(index=[CASE, INST], columns=LIFECYCLE, values=TS,
                          aggfunc="first")
    wide = wide.dropna(subset=["schedule", "start", "complete"])
    bad_qs = int((wide["start"] < wide["schedule"]).sum())
    bad_sc = int((wide["complete"] < wide["start"]).sum())
    print(f"  start before schedule   : {bad_qs:,}")
    print(f"  complete before start   : {bad_sc:,}")
    chk.add("no time-travel within an activity instance",
            bad_qs == 0 and bad_sc == 0)

    wide["queue_min"] = (wide["start"] - wide["schedule"]).dt.total_seconds() / 60
    wide["work_min"] = (wide["complete"] - wide["start"]).dt.total_seconds() / 60
    print(f"  total queueing          : {wide['queue_min'].sum() / 60:,.0f} h")
    print(f"  total processing        : {wide['work_min'].sum() / 60:,.0f} h")
    chk.add("queueing and processing are separable",
            wide["queue_min"].sum() > 0 and wide["work_min"].sum() > 0,
            "the capability the real Sepsis log provably lacks")

    # events within a case must be non-decreasing in time
    ordered = df.sort_values([CASE, TS], kind="stable")
    backwards = int((ordered.groupby(CASE)[TS].diff().dt.total_seconds() < 0).sum())
    chk.add("case timelines never run backwards", backwards == 0,
            f"{backwards} backwards steps" if backwards else "")

    # ---- 4. trace shape -----------------------------------------------------
    rule_line("4. TRACE SHAPE - is this a process or a bag of events?")
    seq = ordered.groupby(CASE)[ACT].apply(list)
    starts_ok = sum(1 for s in seq if s[0] == "Registration")
    ends_ok = sum(1 for s in seq if s[-1] == "Discharge")
    print(f"  cases starting with Registration : {starts_ok:,} of {n_cases:,}")
    print(f"  cases ending with Discharge      : {ends_ok:,} of {n_cases:,}")
    chk.add("every case starts with Registration", starts_ok == n_cases)
    chk.add("every case ends with Discharge", ends_ok == n_cases)

    admitted = seq[seq.apply(lambda s: "Admission" in s)]
    print(f"  admitted patients                : {len(admitted):,} "
          f"({len(admitted) / n_cases * 100:.1f}%)")

    def ordered_ok(acts: list) -> bool:
        """Admission must precede Clinically Fit, which must precede Discharge."""
        return (acts.index("Admission")
                < acts.index("Clinically Fit for Discharge")
                < len(acts) - 1 - acts[::-1].index("Discharge"))

    good_order = sum(1 for s in admitted if "Clinically Fit for Discharge" in s
                     and ordered_ok(s))
    chk.add("admitted: Admission -> Clinically Fit -> Discharge, in order",
            good_order == len(admitted),
            f"{len(admitted) - good_order} out of order" if
            good_order != len(admitted) else
            "so 'discharge cycle time' is a well-defined interval")

    # ---- 5. payment-mode logic ---------------------------------------------
    rule_line("5. PAYMENT MODE - is the Step 5 comparison tag trustworthy?")
    modes_per_case = df.groupby(CASE)[MODE].nunique()
    chk.add("payment_mode is constant within a case",
            int((modes_per_case > 1).sum()) == 0)

    per_case_mode = df.drop_duplicates(CASE).set_index(CASE)[MODE]
    print(f"  {per_case_mode.value_counts().to_dict()}")

    acts_per_case = ordered.groupby(CASE)[ACT].apply(set)
    cashless_cases = per_case_mode[per_case_mode == "cashless"].index
    selfpay_cases = per_case_mode[per_case_mode == "self_pay"].index

    leak_1 = sum(1 for c in selfpay_cases if acts_per_case[c] & CASHLESS_ONLY)
    leak_2 = sum(1 for c in cashless_cases if acts_per_case[c] & SELF_PAY_ONLY)
    print(f"  self-pay cases carrying insurance activities : {leak_1}")
    print(f"  cashless cases carrying self-pay billing     : {leak_2}")
    chk.add("no payment-mode leakage between branches", leak_1 == 0 and leak_2 == 0)

    admitted_cashless = [c for c in cashless_cases if "Admission" in acts_per_case[c]]
    authorised = sum(1 for c in admitted_cashless
                     if "Discharge Authorized" in acts_per_case[c])
    print(f"  admitted cashless patients                   : {len(admitted_cashless):,}")
    print(f"  ... reaching 'Discharge Authorized'          : {authorised:,}")
    chk.add("every admitted cashless patient is authorised before discharge",
            authorised == len(admitted_cashless))

    # ---- 6. warm-up ---------------------------------------------------------
    rule_line("6. WARM-UP - how much does the empty-hospital start distort things?")
    first_seen = ordered.groupby(CASE)[TS].min().sort_values()
    decile = max(1, len(first_seen) // 10)
    early, late = first_seen.index[:decile], first_seen.index[decile:]
    q = wide.reset_index().groupby(CASE)["queue_min"].sum()
    print(f"  mean queueing, first 10% of patients : {q.reindex(early).mean():.1f} min")
    print(f"  mean queueing, remaining 90%         : {q.reindex(late).mean():.1f} min")
    print(
        "\n  The first patients arrive to a completely empty hospital and queue for\n"
        "  nothing. This is a property of any terminating simulation, not a bug.\n"
        "  Reported so the size of the effect is known; if Step 5 needs it removed,\n"
        "  the fix is to discard the warm-up window, not to change the model."
    )

    failed = chk.report()
    if failed == 0:
        print(
            "\n  The log is well-formed and safe to analyse.\n"
            "  It is NOT yet a source of findings: the document-query probability is\n"
            "  an unsourced ASSUMPTION, so the cashless-vs-self-pay difference must\n"
            "  not be quoted until Step 5 sweeps it."
        )
    else:
        print("\n  Fix the failures above before running any analysis on this log.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
