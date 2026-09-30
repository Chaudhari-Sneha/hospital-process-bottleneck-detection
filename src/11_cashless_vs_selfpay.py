"""
STEP 5 - The headline comparison: cashless vs self-pay discharge cycle time,
         swept across the unsourced parameter instead of guessing it.

WHY THIS IS A SWEEP AND NOT A NUMBER

  The project's central hypothesis (CLAUDE.md) is that the real cause of discharge
  delay is repeated document-query loops between the hospital insurance desk and
  the TPA, not the regulatory clock itself.

  We could not source a value for how often that loop fires. It has been marked
  `placeholder` in config/calibration.yaml since Step 3a, and no result depending
  on it has been quoted since.

  Picking a plausible value and reporting "+3.0 hours" would be the exact failure
  this project is built to avoid. So instead we sweep the whole plausible range and
  report the SHAPE of the relationship plus the threshold where the conclusion
  changes. The finding becomes a statement that survives not knowing the input:

      "across every value of q from 0 to 0.75, cashless discharge takes longer
       than self-pay; the gap scales from X to Y; it crosses the IRDAI three-hour
       deadline at q ~ Z"

THE MEASUREMENT

  Discharge cycle time = `Clinically Fit for Discharge` -> `Discharge`.
  This is bed time that no longer has any clinical purpose - the patient is
  medically ready to leave. It converts directly into bed-hours for Step 8.

ONE SUBTLETY THAT THE LOG LETS US HANDLE

  Raising q does TWO things at once: it adds more sequential authorisation steps,
  AND it loads the TPA more heavily (rho climbs from 0.41 at q=0 to 0.89 at
  q=0.75). Those are different mechanisms and a single duration would conflate
  them.

  So every discharge window is decomposed into:
      QUEUEING   = sum(start - schedule)   - congestion, fixable with capacity
      PROCESSING = sum(complete - start)   - the work itself, fixable only by
                                             removing steps from the process
  The real Sepsis log can never do this (Step 1). This one can, and here is where
  that capability finally earns its keep: it says whether the answer to an
  administrator is "hire another reviewer" or "stop asking for the documents
  twice".

AGAINST THE REGULATION

  IRDAI/HLT/CIR/PRO/84/5/2024 clause 16(a) requires final discharge authorisation
  within THREE HOURS. We report the breach rate at every q.
"""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "outputs" / "11_cashless_vs_selfpay.csv"

CASE, ACT, INST = "case:concept:name", "concept:name", "concept:instance"
TS, LIFECYCLE, MODE = "time:timestamp", "lifecycle:transition", "payment_mode"

FIT = "Clinically Fit for Discharge"
OUT = "Discharge"

N_PATIENTS = 3000
SEED = 42

_spec = importlib.util.spec_from_file_location(
    "gen", ROOT / "src" / "07_generate_synthetic_log.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def rule_line(title: str) -> None:
    print(f"\n{'=' * 92}\n{title}\n{'=' * 92}")


def discharge_windows(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per admitted patient: how long they waited to leave after being
    declared medically ready, split into queueing and processing.

    Only activity instances falling INSIDE the window are counted, so the
    decomposition describes the discharge process and not the whole admission.
    """
    # `payment_mode` is invented by OUR generator - no real hospital export has
    # it. These Step 5 experiments always run the generator themselves, so the
    # column is present by construction and this branch should never fire. It is
    # here so that the day someone points this function at a real log, they get
    # one sentence explaining why the comparison is impossible, instead of a bare
    # KeyError that looks like a bug in the analysis.
    if MODE not in df.columns:
        raise SystemExit(
            f"This comparison needs a '{MODE}' column and the log does not have "
            f"one.\n  Columns present: {list(df.columns)[:12]}\n"
            "  Cashless-vs-self-pay is a SYNTHETIC experiment: it requires each\n"
            "  case to be tagged with its billing mode. A real hospital log can\n"
            "  only support it if billing mode is exported per case.\n"
            "  Everything else in the pipeline (03, 05, 18) runs without it."
        )

    fit = df[(df[ACT] == FIT) & (df[LIFECYCLE] == "complete")].set_index(
        CASE)[TS].rename("fit_at")
    out = df[(df[ACT] == OUT) & (df[LIFECYCLE] == "complete")].groupby(
        CASE)[TS].max().rename("out_at")
    mode = df.drop_duplicates(CASE).set_index(CASE)[MODE]

    win = pd.concat([fit, out, mode], axis=1).dropna(subset=["fit_at", "out_at"])
    win["cycle_h"] = (win["out_at"] - win["fit_at"]).dt.total_seconds() / 3600

    # Per-instance queueing and processing, restricted to the window.
    wide = df.pivot_table(index=[CASE, INST], columns=LIFECYCLE, values=TS,
                          aggfunc="first").dropna(
        subset=["schedule", "start", "complete"]).reset_index()
    wide = wide.merge(win[["fit_at", "out_at"]], left_on=CASE, right_index=True)
    inside = wide[(wide["schedule"] >= wide["fit_at"])
                  & (wide["complete"] <= wide["out_at"])]
    q = (inside["start"] - inside["schedule"]).dt.total_seconds() / 3600
    p = (inside["complete"] - inside["start"]).dt.total_seconds() / 3600
    inside = inside.assign(queue_h=q, work_h=p)
    agg = inside.groupby(CASE)[["queue_h", "work_h"]].sum()

    return win.join(agg).fillna({"queue_h": 0.0, "work_h": 0.0})


def summarise(win: pd.DataFrame, q_value: float, deadline_h: float) -> list:
    rows = []
    for mode, grp in win.groupby(MODE):
        rows.append({
            "q": q_value,
            "payment_mode": mode,
            "n": len(grp),
            "median_h": round(grp["cycle_h"].median(), 2),
            "p90_h": round(grp["cycle_h"].quantile(0.90), 2),
            "mean_h": round(grp["cycle_h"].mean(), 2),
            "breach_3h_pct": round(
                (grp["cycle_h"] > deadline_h).mean() * 100, 1),
            "mean_queue_h": round(grp["queue_h"].mean(), 2),
            "mean_work_h": round(grp["work_h"].mean(), 2),
        })
    return rows


def main() -> None:
    cfg = gen.calib.load_calibration()
    sweep = cfg["sensitivity"]["document_query_probability"]["sweep"]
    deadline_h = cfg["regulatory"]["discharge_final_authorization_max"]["value"] / 60

    rule_line("STEP 5 - SENSITIVITY SWEEP")
    print(f"  parameter : document_query_probability  "
          f"(status: {cfg['sensitivity']['document_query_probability']['status']}, "
          "UNSOURCED)")
    print(f"  swept over: {sweep}")
    print(f"  deadline  : {deadline_h:.0f} h  "
          f"({cfg['regulatory']['discharge_final_authorization_max']['source']})")
    print(f"  patients  : {N_PATIENTS:,} per run, seed {SEED}, "
          "common random numbers ON")

    all_rows = []
    for q in sweep:
        df, run_cfg = gen.run(N_PATIENTS, SEED, q, common_random_numbers=True,
                              verbose=False)
        win = discharge_windows(df)
        all_rows.extend(summarise(win, q, deadline_h))
        tpa_rho = gen.check_stability(run_cfg, verbose=False)[1]["rhos"][
            "tpa_reviewers"]
        print(f"    q={q:<5} runs OK   (TPA rho {tpa_rho:.2f})")

    res = pd.DataFrame(all_rows)

    rule_line("DISCHARGE CYCLE TIME - medically ready to actually leaving")
    header = (f"  {'q':>5}  {'mode':<10} {'n':>5} {'median':>8} {'p90':>8} "
              f"{'mean':>8} {'>3h':>7} {'queue':>8} {'work':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for q in sweep:
        for mode in ["cashless", "self_pay"]:
            r = res[(res["q"] == q) & (res["payment_mode"] == mode)]
            if r.empty:
                continue
            r = r.iloc[0]
            print(f"  {q:>5}  {mode:<10} {int(r['n']):>5} "
                  f"{r['median_h']:>7.2f}h {r['p90_h']:>7.2f}h "
                  f"{r['mean_h']:>7.2f}h {r['breach_3h_pct']:>6.1f}% "
                  f"{r['mean_queue_h']:>7.2f}h {r['mean_work_h']:>7.2f}h")
        print()

    # ---- does the conclusion hold across the whole range? -------------------
    rule_line("DOES THE CONCLUSION HOLD ACROSS THE WHOLE RANGE?")
    piv = res.pivot(index="q", columns="payment_mode", values="median_h")
    piv["gap_h"] = piv["cashless"] - piv["self_pay"]
    piv["ratio"] = piv["cashless"] / piv["self_pay"]
    print(f"  {'q':>5} {'cashless':>10} {'self_pay':>10} {'gap':>9} {'ratio':>7}")
    print("  " + "-" * 45)
    for q, row in piv.iterrows():
        print(f"  {q:>5} {row['cashless']:>9.2f}h {row['self_pay']:>9.2f}h "
              f"{row['gap_h']:>+8.2f}h {row['ratio']:>6.1f}x")

    always = bool((piv["gap_h"] > 0).all())
    print(f"\n  cashless slower at EVERY swept value : {always}")
    print(f"  gap range                            : "
          f"{piv['gap_h'].min():+.2f}h to {piv['gap_h'].max():+.2f}h")

    # Where does the median cross the regulatory deadline?
    breached = piv[piv["cashless"] > deadline_h]
    if breached.empty:
        print(f"  median cashless never breaches {deadline_h:.0f}h in this sweep")
    else:
        print(f"  median cashless first breaches {deadline_h:.0f}h at q = "
              f"{breached.index.min()}")

    # ---- queueing vs processing: which fix does an administrator need? ------
    rule_line("QUEUEING vs PROCESSING - is the fix capacity, or fewer steps?")
    cash = res[res["payment_mode"] == "cashless"].set_index("q")
    print(f"  {'q':>5} {'queue':>9} {'work':>9} {'queue share':>13}")
    print("  " + "-" * 40)
    for q, row in cash.iterrows():
        total = row["mean_queue_h"] + row["mean_work_h"]
        share = row["mean_queue_h"] / total * 100 if total else 0
        print(f"  {q:>5} {row['mean_queue_h']:>8.2f}h {row['mean_work_h']:>8.2f}h "
              f"{share:>12.1f}%")
    print(
        "\n  This split is the payoff of emitting `schedule` as well as `start` and\n"
        "  `complete`. The real Sepsis log fuses these forever (Step 1), so it could\n"
        "  never tell an administrator which lever to pull. Read it as:\n"
        "    processing-dominated -> the process has too many STEPS. Remove the\n"
        "      repeat document request; more staff will not help much.\n"
        "    queueing-dominated   -> the steps are fine but there is not enough\n"
        "      CAPACITY to serve them. Hire, or re-roster."
    )

    # Write the numbers FIRST. The chart is presentation; the table is the
    # result, and a rendering problem must not be able to destroy it.
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)
    rule_line("HOW THIS MAY AND MAY NOT BE QUOTED")
    print(
        "  MAY:  'across the full swept range of an unsourced parameter, cashless\n"
        "         discharge is consistently slower than self-pay, by X to Y hours'\n"
        "  MAY:  'the median cashless patient breaches the IRDAI 3-hour rule once\n"
        "         q exceeds <threshold>'\n"
        "  MAY NOT: any single number from any single value of q, presented as the\n"
        "         effect size. The parameter is still unsourced - see\n"
        "         config/calibration.yaml, insurance_branch.preauth_query_loop_probability."
    )
    print(f"\n  wrote {OUT_CSV.relative_to(ROOT)}")
    print("  chart: run  python src/12_plot_sweep.py")
    print("         (kept in a separate process - see that file for why)")


if __name__ == "__main__":
    main()
