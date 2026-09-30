"""
STEP 4.5 (extended) - Robustness sweep: does detection hold across LOCATIONS and
                      SEVERITIES, or did we get lucky once?

Step 4.5 planted one bottleneck at one severity and recovered it. That establishes
the method can work. It does not establish that it works reliably, and a single
successful test is exactly the kind of evidence that looks convincing and proves
little.

So this sweeps the whole grid: every throttleable resource, at every severity the
integer capacities allow, scored the same blind way each time.

TWO DESIGN DECISIONS THAT MAKE THE COMPARISON FAIR

  1. SEVERITY IS EXPRESSED AS rho, NOT AS "CAPACITY MINUS ONE".
     Dropping a registration clerk from 3 to 2 and a TPA reviewer from 30 to 29
     are not comparable shocks. Severity here is the resulting utilisation
     rho = offered load / capacity, so "severe" means the same thing everywhere.
     Capacities are integers, so not every resource can hit every tier - the
     grid takes what each resource can actually reach BELOW its baseline.

  2. SCORING IS AT DEPARTMENT LEVEL.
     Some resources back several activities: `insurance_desk_staff` serves
     Preauth Request, Discharge Auth Request AND Billing. Scoring per activity
     would split that resource's signal three ways and understate detection.
     `org:group` is a column in the log, so ranking by department stays blind -
     the analysis is not being told which resource was throttled, only grouping
     events by a field the log already carries.

NEGATIVE CONTROL
  The unplanted control log is analysed too. A method that always names a
  bottleneck will "recover" one by luck; knowing what it says when nothing is
  wrong is what makes the positive results meaningful.

Everything runs against throwaway in-memory logs. The Step 4 production log is
never read or written.
"""

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "outputs" / "10_robustness_sweep.csv"

N_PATIENTS = 3000
SEED = 42

_gen_spec = importlib.util.spec_from_file_location(
    "gen", ROOT / "src" / "07_generate_synthetic_log.py")
gen = importlib.util.module_from_spec(_gen_spec)
_gen_spec.loader.exec_module(gen)

# Reuse the blind ranking functions from the Step 4.5 test rather than
# reimplementing them - same pipeline, same code path.
_test_spec = importlib.util.spec_from_file_location(
    "planted", ROOT / "src" / "09_planted_bottleneck_test.py")
planted_mod = importlib.util.module_from_spec(_test_spec)
_test_spec.loader.exec_module(planted_mod)
blind_rank_lifecycle = planted_mod.blind_rank_lifecycle
blind_rank_legacy = planted_mod.blind_rank_legacy

# Which department each throttleable resource shows up as in `org:group`.
# Used ONLY for scoring, after the blind ranking has been produced.
RESOURCE_TO_GROUP = {
    "registration_clerks": "FrontDesk",
    "triage_nurses": "TriageDesk",
    "doctors": "Doctor",
    "lab_stations": "Laboratory",
    "beds": "Ward",
    "insurance_desk_staff": "InsuranceDesk",
    "tpa_reviewers": "TPA",
}
RESOURCE_TO_ACTIVITY = {
    "registration_clerks": "Registration",
    "triage_nurses": "Triage",
    "doctors": "Consultation",
    "lab_stations": "Lab Test",
    "beds": "Admission",
    "insurance_desk_staff": "Preauth Request",
    "tpa_reviewers": "Preauth Review",
}

TARGET_RHOS = [0.55, 0.80, 0.92]


def rule_line(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def build_grid(stability: dict) -> list:
    """
    For each resource, find the integer capacities that land nearest each target
    rho tier AND are strictly below the baseline capacity (a plant must be a
    throttle, not an expansion). Duplicates are collapsed.
    """
    grid = []
    for resource, load in stability["loads"].items():
        baseline = stability["capacity"][resource]
        seen = set()
        for target in TARGET_RHOS:
            cap = max(1, round(load / target))
            if cap >= baseline or cap in seen:
                continue
            seen.add(cap)
            grid.append({
                "resource": resource,
                "baseline_capacity": baseline,
                "planted_capacity": cap,
                "rho": load / cap,
                "baseline_rho": load / baseline,
            })
    return sorted(grid, key=lambda r: (r["resource"], -r["planted_capacity"]))


def rank_departments(df: pd.DataFrame) -> pd.Series:
    """
    Blind department ranking: total queueing absorbed per `org:group`.

    Built from the activity-level blind ranking, so it inherits exactly the same
    schedule->start measurement. No knowledge of which resource was throttled.
    """
    activity_stats = blind_rank_lifecycle(df)
    groups = df.drop_duplicates("concept:name").set_index(
        "concept:name")["org:group"]
    by_group = activity_stats.join(groups).groupby("org:group")[
        "total_queue_h"].sum()
    return by_group.sort_values(ascending=False)


def score_one(plant: dict, control_depts: pd.Series) -> dict:
    """Generate a planted log, rank it blind, then score against the plant."""
    df, _ = gen.run(N_PATIENTS, SEED, None,
                    capacity_override={plant["resource"]:
                                       plant["planted_capacity"]},
                    common_random_numbers=True, verbose=False)

    depts = rank_departments(df)
    activities = blind_rank_lifecycle(df)

    target_group = RESOURCE_TO_GROUP[plant["resource"]]
    target_activity = RESOURCE_TO_ACTIVITY[plant["resource"]]

    order = list(depts.index)
    dept_rank = order.index(target_group) + 1 if target_group in order else None
    false_positives = [g for g in order[:max(0, (dept_rank or 1) - 1)]]

    act_order = list(activities.index)
    act_rank = (act_order.index(target_activity) + 1
                if target_activity in act_order else None)

    # Legacy view: best edge leading INTO the throttled activity.
    legacy = blind_rank_legacy(df)
    into = [e for e in legacy.index if e.endswith(f"->  {target_activity}")]
    legacy_rank = list(legacy.index).index(into[0]) + 1 if into else None

    queue_h = float(depts.get(target_group, 0.0))
    control_h = float(control_depts.get(target_group, 0.0))

    # A SECOND ranking, for diagnosis rather than for the headline result: rank
    # departments by how much their queue GREW against the control, instead of by
    # absolute size. This separates two very different kinds of failure:
    #   - the method never saw the plant at all, versus
    #   - the method saw it but could not outrank a department that is naturally
    #     the busiest anyway.
    # It is NOT the primary metric, because a real hospital analysing one log has
    # no control group to difference against. It tells us whether the limitation
    # is in the detection or in the absence of a baseline.
    delta = (depts - control_depts.reindex(depts.index).fillna(0.0)).sort_values(
        ascending=False)
    delta_order = list(delta.index)
    delta_rank = (delta_order.index(target_group) + 1
                  if target_group in delta_order else None)

    return {
        "resource": plant["resource"],
        "activity": target_activity,
        "department": target_group,
        "capacity": f"{plant['baseline_capacity']} -> {plant['planted_capacity']}",
        "rho_baseline": round(plant["baseline_rho"], 3),
        "rho_planted": round(plant["rho"], 3),
        "queue_h_control": round(control_h, 1),
        "queue_h_planted": round(queue_h, 1),
        "queue_h_delta": round(queue_h - control_h, 1),
        "dept_rank": dept_rank,
        "delta_rank": delta_rank,
        "activity_rank": act_rank,
        "legacy_edge_rank": legacy_rank,
        "false_positives": len(false_positives),
        "false_positive_names": ", ".join(false_positives) if false_positives else "",
        "recovered": dept_rank == 1,
    }


def main() -> None:
    rule_line("BUILDING THE GRID")
    cfg = gen.calib.load_calibration()
    _, stability = gen.check_stability(cfg, verbose=False)
    grid = build_grid(stability)
    print(f"  {len(grid)} plants across {len(set(g['resource'] for g in grid))} "
          "resources")
    print(f"  severity targeted at rho tiers {TARGET_RHOS}, capped to integer "
          "capacities below baseline\n")
    for g in grid:
        print(f"    {g['resource']:<24} {g['baseline_capacity']:>4} -> "
              f"{g['planted_capacity']:<4}  rho {g['baseline_rho']:.2f} -> "
              f"{g['rho']:.2f}")

    # ---- negative control ---------------------------------------------------
    rule_line("NEGATIVE CONTROL - what does the method say when nothing is planted?")
    control, _ = gen.run(N_PATIENTS, SEED, None, common_random_numbers=True,
                         verbose=False)
    control_depts = rank_departments(control)
    print("  department queueing in the UNPLANTED log (total hours):\n")
    for i, (group, hours) in enumerate(control_depts.head(6).items(), start=1):
        print(f"    {i}. {group:<18} {hours:>9,.1f} h")
    print(
        "\n  The method always produces a ranking, so it will always name something.\n"
        "  What matters below is whether a PLANT displaces this natural order - and\n"
        "  the departments listed here are the ones that will show up as false\n"
        "  positives when a plant is too weak to outrank them."
    )

    # ---- the sweep ----------------------------------------------------------
    rule_line("RUNNING THE SWEEP (each plant analysed blind)")
    results = [score_one(p, control_depts) for p in grid]
    df = pd.DataFrame(results)

    rule_line("RESULTS")
    header = (f"  {'resource':<22} {'capacity':<12} {'rho':>5} {'ctrl q':>8} "
              f"{'plant q':>8} {'rank':>5} {'dlt':>4} {'act':>4} {'legacy':>7} {'FP':>3}  result")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        verdict = "RECOVERED" if r["recovered"] else f"MISSED (rank {r['dept_rank']})"
        print(f"  {r['resource']:<22} {r['capacity']:<12} {r['rho_planted']:>5.2f} "
              f"{r['queue_h_control']:>8,.0f} {r['queue_h_planted']:>8,.0f} "
              f"{str(r['dept_rank']):>5} {str(r['delta_rank']):>4} "
              f"{str(r['activity_rank']):>4} "
              f"{str(r['legacy_edge_rank']):>7} {r['false_positives']:>3}  {verdict}")

    # ---- summary ------------------------------------------------------------
    rule_line("SUMMARY")
    n_ok = int(df["recovered"].sum())
    print(f"  recovered at department rank 1 : {n_ok} of {len(df)}")
    print(f"  mean false positives           : {df['false_positives'].mean():.2f}")
    n_delta = int((df["delta_rank"] == 1).sum())
    print(f"  recovered when ranked by GROWTH against a control : {n_delta} of {len(df)}")
    print(
        "\n  That second figure is diagnostic, not a headline. A hospital analysing\n"
        "  one log has no control to difference against. But the gap between the two\n"
        "  numbers says where the limitation actually lives: if growth-ranking\n"
        "  recovers plants that absolute ranking misses, the method SAW every plant\n"
        "  and the missing ingredient is a baseline, not sensitivity."
    )

    print("\n  by severity tier:")
    for lo, hi, label in [(0, 0.75, "mild      rho < 0.75"),
                          (0.75, 0.88, "moderate  0.75-0.88"),
                          (0.88, 2.0, "severe    rho > 0.88")]:
        band = df[(df["rho_planted"] >= lo) & (df["rho_planted"] < hi)]
        if band.empty:
            continue
        print(f"    {label:<22} {int(band['recovered'].sum())} of {len(band)} "
              f"recovered")

    failures = df[~df["recovered"]]
    if failures.empty:
        print("\n  No failure cases: every plant was recovered at rank 1.")
    else:
        rule_line("FAILURE CASES")
        for _, r in failures.iterrows():
            print(f"  {r['resource']} ({r['capacity']}, rho {r['rho_planted']:.2f})"
                  f" -> ranked #{r['dept_rank']}")
            print(f"      outranked by: {r['false_positive_names']}")
            print(f"      queue rose {r['queue_h_control']:,.0f} h -> "
                  f"{r['queue_h_planted']:,.0f} h "
                  f"(delta {r['queue_h_delta']:+,.0f} h)")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  full table written to {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
