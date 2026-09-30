"""
STEP 8 - Bed-hours, and an ILLUSTRATIVE rupee benchmark.

THE CHAIN

    discharge cycle time  ->  bed-hours consumed  ->  illustrative rupee value
    [measured, Step 5]        [computed here]         [external benchmark]

  The last link uses `economics.arpob_benchmark`: a real, SEBI-filed figure from
  Global Health Limited (Medanta), Q3 FY2026, verified against the filing itself.

  IT IS A BENCHMARK, NOT OUR HOSPITAL'S ARPOB. The bed-hours come from a
  SYNTHETIC hospital whose process structure was calibrated on a DUTCH log. No
  real hospital's revenue is being measured. `arpob_of_study_hospital` remains a
  deliberate placeholder in the calibration file so the audit keeps showing that
  the benchmark is standing in for something we do not have.

WHY BED-HOURS PER 1,000 ADMISSIONS

  Absolute totals depend on how many patients the simulation happened to run, so
  they are an artefact of the experiment. Normalising per 1,000 cashless
  admissions makes the figure SCALE-FREE: a hospital applies its own annual
  volume and gets its own answer, without this project having to assume an
  Indian hospital's size - which would be one more unsourced number.

THREE QUANTITIES, DELIBERATELY DISTINGUISHED

  1. TOTAL discharge bed-hours   - all time between medically-ready and leaving.
                                   Some of it is unavoidable; paperwork is not
                                   instant.
  2. EXCESS over self-pay        - the part attributable to the payment route.
                                   Self-pay patients in the same simulated
                                   hospital are the natural counterfactual: same
                                   beds, same wards, same clinicians, different
                                   discharge path.
  3. BEYOND the 3-hour deadline  - the part a regulator would call a breach.
                                   IRDAI/HLT/CIR/PRO/84/5/2024 clause 16(a).

  (2) is the honest "recoverable" number. (1) overstates it - you cannot drive
  discharge admin to zero. (3) is the regulatory framing, not the operational one.

EVERYTHING IS CONDITIONAL ON q, EXCEPT THE FLOOR
  `document_query_probability` is unsourced, so every figure is reported across
  the swept range. The q=0 row is the FLOOR: the cost of the authorisation
  process with a perfect insurer that never asks twice. That one number does not
  depend on the unsourced parameter at all, and it is the defensible headline.
"""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "outputs" / "17_bed_hours.csv"

SEEDS = [42, 43, 44, 45, 46]
N_PATIENTS = 3000
DEADLINE_H = 3.0
PER = 1000                     # report per 1,000 cashless admissions

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


def bed_hours_for(q: float, seed: int) -> dict:
    """Run one simulation and reduce it to bed-hour aggregates."""
    df, _ = gen.run(N_PATIENTS, seed, q, common_random_numbers=True, verbose=False)
    win = discharge_windows(df)

    cash = win[win["payment_mode"] == "cashless"]
    self_pay = win[win["payment_mode"] == "self_pay"]
    n = len(cash)
    if n == 0:
        return {}

    selfpay_mean = self_pay["cycle_h"].mean()

    # (1) all discharge bed-hours, per 1,000 cashless admissions
    total = cash["cycle_h"].sum() / n * PER

    # (2) excess over the self-pay counterfactual. Clipped at zero per patient:
    #     a cashless patient who happened to leave FASTER than the self-pay mean
    #     is not a saving that can be banked, and letting those offset the slow
    #     ones would understate the recoverable total.
    excess = (cash["cycle_h"] - selfpay_mean).clip(lower=0).sum() / n * PER

    # (3) hours beyond the regulatory deadline
    beyond = (cash["cycle_h"] - DEADLINE_H).clip(lower=0).sum() / n * PER

    return {
        "q": q, "seed": seed, "n_cashless": n,
        "selfpay_mean_h": selfpay_mean,
        "total_bed_h_per_1000": total,
        "excess_vs_selfpay_per_1000": excess,
        "beyond_deadline_per_1000": beyond,
    }


def band(g: pd.DataFrame, col: str) -> str:
    return f"{g[col].mean():8,.0f}  [{g[col].min():,.0f}-{g[col].max():,.0f}]"


def main() -> None:
    cfg = gen.calib.load_calibration()
    sweep = cfg["sensitivity"]["document_query_probability"]["sweep"]
    arpob = cfg["economics"]["arpob_benchmark"]

    rule_line("STEP 8 - BED-HOURS, and an ILLUSTRATIVE rupee benchmark")
    print(f"  seeds      : {SEEDS}")
    print(f"  q swept    : {sweep}  (UNSOURCED)")
    print(f"  deadline   : {DEADLINE_H:.0f}h  (IRDAI/HLT/CIR/PRO/84/5/2024 cl.16(a))")
    print(f"  normalised : per {PER:,} cashless admissions (scale-free)")
    print(f"\n  ARPOB benchmark : INR {arpob['value']:,}/bed-day  "
          f"({arpob['quarter']}, status '{arpob['status']}', "
          f"{arpob['applies_to']})")

    rows = [bed_hours_for(q, s) for q in sweep for s in SEEDS]
    res = pd.DataFrame([r for r in rows if r])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)

    rule_line(f"BED-HOURS PER {PER:,} CASHLESS ADMISSIONS  (mean [min-max] over seeds)")
    print(f"  {'q':>5}  {'total':>24}  {'excess vs self-pay':>24}  "
          f"{'beyond 3h':>24}")
    print("  " + "-" * 84)
    for q in sweep:
        g = res[res["q"] == q]
        print(f"  {q:>5}  {band(g, 'total_bed_h_per_1000'):>24}  "
              f"{band(g, 'excess_vs_selfpay_per_1000'):>24}  "
              f"{band(g, 'beyond_deadline_per_1000'):>24}")

    # ---- the floor ---------------------------------------------------------
    floor = res[res["q"] == 0.0]
    rule_line("THE FLOOR - what it costs with a PERFECT insurer (q = 0)")
    print("  This row does not depend on the unsourced parameter at all. It is the")
    print("  cost of the authorisation process itself, assuming every request is")
    print("  approved first time and no document is ever asked for twice.\n")
    print(f"  total discharge bed-hours per {PER:,} cashless admissions : "
          f"{band(floor, 'total_bed_h_per_1000')}")
    print(f"  excess over the self-pay counterfactual              : "
          f"{band(floor, 'excess_vs_selfpay_per_1000')}")
    print(f"  hours beyond the 3-hour regulatory deadline          : "
          f"{band(floor, 'beyond_deadline_per_1000')}")
    print(f"\n  self-pay comparison group averaged "
          f"{floor['selfpay_mean_h'].mean():.2f}h to discharge.")

    # ---- what a reader can do with this ------------------------------------
    rule_line("APPLYING THIS TO A HOSPITAL'S OWN ADMISSION VOLUME")
    ex = floor["excess_vs_selfpay_per_1000"].mean()
    print("  These are per 1,000 cashless admissions, so a hospital multiplies by")
    print("  its own volume. Worked example using the FLOOR (q=0) figure:\n")
    for admissions in (2000, 5000, 10000):
        print(f"    {admissions:>6,} cashless admissions/year  ->  "
              f"{ex * admissions / PER:>9,.0f} excess bed-hours/year  "
              f"({ex * admissions / PER / 24:>7,.0f} bed-days)")
    print("\n  Bed-days first. The rupee conversion follows, against an external")
    print("  benchmark that is NOT this hospital's own revenue per bed day.")

    # ---- the rupee step, against an EXTERNAL benchmark -----------------------
    rule_line("RUPEE IMPACT - against an EXTERNAL INDIAN BENCHMARK")
    print(f"  benchmark ARPOB  : INR {arpob['value']:,} per occupied bed day")
    print(f"  quarter          : {arpob['quarter']}")
    print(f"  filed            : {arpob['filing_date']}")
    print(f"  status           : {arpob['status']}  ({arpob['applies_to']})")
    print(f"\n  company definition, verbatim from the filing:")
    print(f"    {str(arpob['company_definition']).strip()}")

    per_bed_hour = arpob["value"] / 24.0
    print(f"\n  rupee impact = excess bed-hours x (ARPOB / 24)")
    print(f"               = excess bed-hours x INR {per_bed_hour:,.0f} per bed-hour")

    def inr(x: float) -> str:
        """Indian formatting - crore above 1e7, else lakh."""
        if x >= 1e7:
            return f"INR {x / 1e7:,.2f} crore"
        return f"INR {x / 1e5:,.1f} lakh"

    print(f"\n  ILLUSTRATIVE, per 1,000 cashless admissions (excess over self-pay):\n")
    print(f"  {'q':>5}  {'excess bed-hours':>18}  {'illustrative value':>22}")
    print("  " + "-" * 50)
    for q in sweep:
        g = res[res["q"] == q]
        h = g["excess_vs_selfpay_per_1000"].mean()
        print(f"  {q:>5}  {h:>18,.0f}  {inr(h * per_bed_hour):>22}")

    floor_h = floor["excess_vs_selfpay_per_1000"].mean()
    print(f"\n  Worked examples at the q=0 FLOOR "
          f"({floor_h:,.0f} excess bed-hours per 1,000):\n")
    for admissions in (2000, 5000, 10000):
        h = floor_h * admissions / PER
        print(f"    {admissions:>6,} cashless admissions/year  ->  {h:>8,.0f} "
              f"bed-hours  ->  {inr(h * per_bed_hour):>20} per year")

    rule_line("HOW THIS NUMBER MUST BE LABELLED")
    print(
        "  It is an ILLUSTRATIVE BENCHMARK, and every one of these applies:\n"
        "\n  1. NOT OUR HOSPITAL'S ARPOB. The benchmark is from a listed Indian\n"
        "     chain. The bed-hours come from a SYNTHETIC hospital whose process\n"
        "     structure was calibrated on a DUTCH log. No real hospital's revenue\n"
        "     is being measured here - the arithmetic gives an order of magnitude.\n"
        "\n  2. THE BENCHMARK SITS AT THE PREMIUM END. Medanta is a quaternary,\n"
        "     high-acuity chain (ALOS 3.02 days). A district or mid-tier hospital's\n"
        "     ARPOB is materially lower, so this is an optimistic figure, not a\n"
        "     median Indian hospital.\n"
        "\n  3. THE DEFINITION EXCLUDES PHARMACY AND OTHER INCOME, a narrower\n"
        "     revenue base than total hospital revenue. This biases the estimate\n"
        "     DOWN, while (2) biases it UP. The two do not cancel in any known\n"
        "     proportion, so the result is an order of magnitude, not an estimate.\n"
        "\n  4. REVENUE IS NOT PROFIT, AND A FREED BED IS NOT AUTOMATICALLY FILLED.\n"
        "     ARPOB x bed-hours is an UPPER BOUND ON OPPORTUNITY. It converts to\n"
        "     cash only where there is unmet demand queueing for the bed - which at\n"
        "     the benchmark chain's ~59% occupancy plainly is not everywhere true."
    )

    rule_line("WHAT MAY BE QUOTED FROM THIS SCRIPT")
    print("  MAY:  bed-hours and bed-days, as a swept range, per 1,000 admissions")
    print("  MAY:  the q=0 floor as a parameter-independent minimum")
    print("  MAY:  the rupee figure IF labelled an illustrative benchmark, with the")
    print("        chain, quarter and all four caveats above stated alongside it")
    print("  MAY NOT: the rupee figure as the study hospital's actual loss or saving")
    print("  MAY NOT: a single value of q presented as the effect size")
    print(f"\n  wrote {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
