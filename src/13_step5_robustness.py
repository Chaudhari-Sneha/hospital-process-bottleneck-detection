"""
STEP 5 (robustness) - Two experiments before the dashboard.

Step 5 produced a clean result from ONE seed at ONE capacity configuration. Two
obvious ways that result could be an artefact, so both get tested:

EXPERIMENT A - REPLICATION ACROSS SEEDS
    Everything so far is seed 42. A single sample path can produce a threshold
    that moves the moment you reroll the dice. So the whole q-sweep is repeated
    across several seeds and the spread is reported, not just the mean. The
    question is not "what is the number" but "does the CONCLUSION survive".

EXPERIMENT B - UNDERSTAFFED TPA (HIGH UTILISATION)
    Step 5 found the cashless delay was ~100% processing and ~0% queueing, and
    flagged honestly that this held only because the insurance branch was
    generously staffed (TPA rho 0.12 to 0.89 across the sweep). If a real
    hospital's TPA runs hot, does queueing ADD to the processing delay, or does
    something else happen?

    A DESIGN WRINKLE, stated because it shapes the result: TPA offered load
    varies 7.5x across the sweep (3.5 at q=0 to 26.6 at q=0.75), because each
    extra document query is extra TPA work. A single fixed capacity therefore
    CANNOT hold utilisation constant - it would be idle at low q and saturated at
    high q, exactly the confound we are trying to remove.

    So TPA capacity is sized PER q to hold rho ~= 0.90 throughout. "High
    utilisation" is then a controlled condition rather than something that drifts
    with the parameter under test. Everything else - beds, lab, doctors,
    registration, triage, the insurance desk - stays at baseline, so any change
    is attributable to the TPA alone.

WHAT IS NOT DONE HERE
    No ARPOB, no rupee figure. `economics.arpob` is still a placeholder in
    config/calibration.yaml and Step 8 stays blocked until it has a real source.
"""

import importlib.util
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "outputs" / "13_step5_robustness.csv"

SEEDS = [42, 43, 44, 45, 46]
N_PATIENTS = 3000
TARGET_RHO = 0.90        # the understaffed condition

_gen = importlib.util.spec_from_file_location(
    "gen", ROOT / "src" / "07_generate_synthetic_log.py")
gen = importlib.util.module_from_spec(_gen)
_gen.loader.exec_module(gen)

_s5 = importlib.util.spec_from_file_location(
    "s5", ROOT / "src" / "11_cashless_vs_selfpay.py")
s5 = importlib.util.module_from_spec(_s5)
_s5.loader.exec_module(s5)
discharge_windows = s5.discharge_windows


def rule_line(title: str) -> None:
    print(f"\n{'=' * 96}\n{title}\n{'=' * 96}")


def tpa_capacity_for(cfg: dict, q: float) -> tuple[int, float]:
    """
    Capacity that puts the TPA at TARGET_RHO for this value of q.

    The load formula lives in check_stability(), so we ask it rather than
    duplicating it - the same single source of truth that caught the preauth
    load bug in Step 4.5.
    """
    probe = gen.calib.load_calibration()
    probe["sensitivity"]["document_query_probability"]["value"] = q
    _, stability = gen.check_stability(probe, verbose=False)
    load = stability["loads"]["tpa_reviewers"]
    cap = max(1, math.ceil(load / TARGET_RHO))
    return cap, load / cap


def run_one(q: float, seed: int, scenario: str, tpa_cap: int | None) -> list:
    override = {"tpa_reviewers": tpa_cap} if tpa_cap else None
    df, _ = gen.run(N_PATIENTS, seed, q, capacity_override=override,
                    common_random_numbers=True, verbose=False)
    win = discharge_windows(df)
    rows = []
    for mode, grp in win.groupby("payment_mode"):
        rows.append({
            "scenario": scenario,
            "q": q,
            "seed": seed,
            "payment_mode": mode,
            "n": len(grp),
            "median_h": grp["cycle_h"].median(),
            "p90_h": grp["cycle_h"].quantile(0.90),
            "breach_3h_pct": (grp["cycle_h"] > 3.0).mean() * 100,
            "mean_queue_h": grp["queue_h"].mean(),
            "mean_work_h": grp["work_h"].mean(),
        })
    return rows


def spread(series: pd.Series) -> str:
    """mean plus the full observed range - the range is what shows stability."""
    return (f"{series.mean():6.2f}  [{series.min():5.2f}-{series.max():5.2f}]")


def main() -> None:
    cfg = gen.calib.load_calibration()
    sweep = cfg["sensitivity"]["document_query_probability"]["sweep"]

    rule_line("SETUP")
    print(f"  seeds      : {SEEDS}")
    print(f"  q swept    : {sweep}   (UNSOURCED - still swept, never quoted singly)")
    print(f"  patients   : {N_PATIENTS:,} per run")
    print(f"  scenarios  : baseline (capacities unchanged)")
    print(f"               tight_tpa (TPA sized per-q to rho ~= {TARGET_RHO})")

    print("\n  TPA capacity used in the tight scenario, per q:")
    tight_caps = {}
    for q in sweep:
        cap, rho = tpa_capacity_for(cfg, q)
        base_rho = gen.check_stability(
            {**cfg, "sensitivity": {**cfg["sensitivity"],
             "document_query_probability": {
                 **cfg["sensitivity"]["document_query_probability"],
                 "value": q}}}, verbose=False)[1]["rhos"]["tpa_reviewers"]
        tight_caps[q] = cap
        print(f"    q={q:<5} baseline cap 30 (rho {base_rho:.2f})  ->  "
              f"tight cap {cap:>2} (rho {rho:.2f})")

    # ---- run everything ----------------------------------------------------
    rows = []
    for q in sweep:
        for seed in SEEDS:
            rows += run_one(q, seed, "baseline", None)
            rows += run_one(q, seed, "tight_tpa", tight_caps[q])
    res = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)

    cash = res[res["payment_mode"] == "cashless"]
    self_pay = res[res["payment_mode"] == "self_pay"]

    # =====================================================================
    rule_line(f"EXPERIMENT A - REPLICATION ACROSS {len(SEEDS)} SEEDS (baseline)")
    base = cash[cash["scenario"] == "baseline"]
    print("  cashless, mean across seeds with [min-max] range:\n")
    print(f"  {'q':>5}  {'median (h)':>22}  {'p90 (h)':>22}  {'breach >3h (%)':>22}")
    print("  " + "-" * 78)
    for q in sweep:
        g = base[base["q"] == q]
        print(f"  {q:>5}  {spread(g['median_h']):>22}  {spread(g['p90_h']):>22}  "
              f"{spread(g['breach_3h_pct']):>22}")

    sp = self_pay[self_pay["scenario"] == "baseline"]
    print(f"\n  self-pay for comparison (should be flat - q only touches the "
          "insurance branch):")
    print(f"    median {spread(sp['median_h'])}   "
          f"breach {spread(sp['breach_3h_pct'])}")

    # Does the CONCLUSION hold in every single seed?
    rule_line("EXPERIMENT A - DO THE CONCLUSIONS SURVIVE EVERY SEED?")
    piv = res[res["scenario"] == "baseline"].pivot_table(
        index=["q", "seed"], columns="payment_mode", values="median_h")
    piv["gap"] = piv["cashless"] - piv["self_pay"]
    n_runs = len(piv)
    print(f"  cashless slower than self-pay : "
          f"{int((piv['gap'] > 0).sum())} of {n_runs} runs")
    print(f"  gap range across all runs     : {piv['gap'].min():+.2f}h to "
          f"{piv['gap'].max():+.2f}h")

    print("\n  q at which the median cashless patient first breaches 3h, per seed:")
    thresholds = []
    for seed in SEEDS:
        g = base[base["seed"] == seed].sort_values("q")
        over = g[g["median_h"] > 3.0]["q"]
        t = over.min() if len(over) else None
        thresholds.append(t)
        print(f"    seed {seed}: q = {t if t is not None else 'never'}")
    uniq = sorted({t for t in thresholds if t is not None})
    print(f"\n  => threshold is {'IDENTICAL' if len(uniq) == 1 else 'VARIABLE'} "
          f"across seeds: {uniq}")

    print("\n  queue share of cashless delay, per q (mean across seeds):")
    for q in sweep:
        g = base[base["q"] == q]
        tot = g["mean_queue_h"].mean() + g["mean_work_h"].mean()
        share = g["mean_queue_h"].mean() / tot * 100 if tot else 0
        print(f"    q={q:<5} queue {g['mean_queue_h'].mean():5.2f}h  "
              f"work {g['mean_work_h'].mean():6.2f}h  queue share {share:4.1f}%")

    # =====================================================================
    rule_line("EXPERIMENT B - UNDERSTAFFED TPA vs BASELINE (cashless, mean of seeds)")
    print(f"  {'q':>5}  {'scenario':<10} {'median':>8} {'p90':>8} {'>3h':>7} "
          f"{'queue':>8} {'work':>8} {'q share':>8}")
    print("  " + "-" * 74)
    for q in sweep:
        for scenario in ["baseline", "tight_tpa"]:
            g = cash[(cash["q"] == q) & (cash["scenario"] == scenario)]
            queue, work = g["mean_queue_h"].mean(), g["mean_work_h"].mean()
            share = queue / (queue + work) * 100 if (queue + work) else 0
            print(f"  {q:>5}  {scenario:<10} {g['median_h'].mean():>7.2f}h "
                  f"{g['p90_h'].mean():>7.2f}h {g['breach_3h_pct'].mean():>6.1f}% "
                  f"{queue:>7.2f}h {work:>7.2f}h {share:>7.1f}%")
        print()

    rule_line("EXPERIMENT B - DOES QUEUEING ADD TO PROCESSING?")
    print(f"  {'q':>5}  {'median base':>12} {'median tight':>13} {'delta':>9}"
          f"   {'queue base':>11} {'queue tight':>12}")
    print("  " + "-" * 72)
    for q in sweep:
        b = cash[(cash["q"] == q) & (cash["scenario"] == "baseline")]
        t = cash[(cash["q"] == q) & (cash["scenario"] == "tight_tpa")]
        print(f"  {q:>5}  {b['median_h'].mean():>11.2f}h "
              f"{t['median_h'].mean():>12.2f}h "
              f"{t['median_h'].mean() - b['median_h'].mean():>+8.2f}h   "
              f"{b['mean_queue_h'].mean():>10.2f}h "
              f"{t['mean_queue_h'].mean():>11.2f}h")

    print(f"\n  wrote {OUT_CSV.relative_to(ROOT)}")
    rule_line("REMINDERS")
    print("  - q remains UNSOURCED. No single value of it may be quoted as an")
    print("    effect size; only the shape across the sweep.")
    print("  - No ARPOB, no rupee figure. economics.arpob is still a placeholder,")
    print("    so the Step 8 business case stays blocked by design.")


if __name__ == "__main__":
    main()
