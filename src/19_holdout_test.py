"""
STEP 9 - THE HOLDOUT TEST. Does the pipeline work on a log it has never seen?

WHY THIS EXISTS

Everything in Steps 1-8 was written while looking at the Sepsis log. Code written
that way silently encodes the log it was written against, and you CANNOT detect
that by re-running it on the same log - it will keep passing forever. The only
real test is a log the code has never touched.

That log is Hospital Billing (Mannhardt, TU Eindhoven): 451,359 events, 100,000
cases, a different hospital and a completely different process - an ERP billing
state machine rather than a patient pathway. It was kept unused
until this point.

WHAT THIS TEST ASSERTS

Not "does it produce the right answer" - there is no ground truth for another
hospital's bottleneck. It asserts the things that must be true of ANY log for the
pipeline to be considered general:

  1. The profiler runs without being told anything about the log.
  2. The resource column is auto-detected, not assumed to be `org:group`.
  3. Resource granularity is classified, and the internal-vs-handoff split is
     WITHHELD when the column names individuals rather than departments. This
     check exists because the holdout caught the pipeline reporting "100.0% of
     waiting is handoff" - a confident, meaningless number.
  4. Episode boundaries are discovered, or correctly found not to exist.
  5. Many departments collapse to few behavioural roles, so the map is readable.
  6. `03` runs end to end and exits 0.
  7. Analyses needing `payment_mode` SKIP with an explanation instead of KeyError.
  8. NO SEPSIS ACTIVITY NAME APPEARS IN ANY HOLDOUT OUTPUT.

Assertion 8 is the sharpest one. If a hard-coded `Release A` or `Leucocytes` is
left anywhere in the logic, it will leak into the artifacts produced from a log
that contains no such activity. It is a direct test for the failure mode this
whole exercise is about.

RUN IT

    python src/19_holdout_test.py                 # fast: no map rendering
    python src/19_holdout_test.py --with-maps     # also renders the process maps
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
HOLDOUT = ROOT / "data" / "holdout" / "Hospital Billing - Event Log.xes.gz"
PREFIX = "19_holdout"

# Activity names that exist ONLY in the Sepsis log. If any of these turns up in an
# artifact built from the holdout, something is hard-coded that should not be.
SEPSIS_ONLY = ["Leucocytes", "CRP", "LacticAcid", "ER Registration", "ER Triage",
               "ER Sepsis Triage", "IV Antibiotics", "IV Liquid", "Return ER",
               "Release A", "Release B", "Admission NC", "Admission IC"]

PY = sys.executable
results: list[tuple[str, bool, str]] = []


def rule_line(title: str) -> None:
    print(f"\n{'=' * 92}\n{title}\n{'=' * 92}")


def check(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))


def load_module_by_path(filename: str, name: str):
    """Import a numbered script by path (a module name cannot start with a digit)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_profiler():
    return load_module_by_path("18_process_profile.py", "profiler")


def run_script(args: list[str]) -> subprocess.CompletedProcess:
    """Run a pipeline script as a real subprocess, so we test the real entry point."""
    return subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Holdout generalization test.")
    parser.add_argument("--with-maps", action="store_true",
                        help="also render process maps (slow on 100,000 cases)")
    args = parser.parse_args()

    if not HOLDOUT.exists():
        raise SystemExit(
            f"Holdout log not found at {HOLDOUT}\n\n"
            "  Download 'Hospital Billing - Event Log' (Mannhardt, TU Eindhoven)\n"
            "  and place the .xes.gz there:\n"
            "    https://doi.org/10.4121/uuid:76c46b83-c930-4798-a1c9-4be94dfeb741\n\n"
            "  It is not redistributed with this repository, and is\n"
            "  NOT used by any other step - that is what makes it a holdout.")

    rule_line(f"HOLDOUT TEST - {HOLDOUT.name}")
    print("  This log has never been seen by any code in this project.")

    # ---- 1. the profiler runs unaided ------------------------------------
    rule_line("1-5. STRUCTURAL DISCOVERY (src/18_process_profile.py)")
    proc = run_script(["src/18_process_profile.py", "--log", str(HOLDOUT)])
    check("profiler exits 0", proc.returncode == 0,
          proc.stderr.strip().splitlines()[-1] if proc.returncode else "")

    profile_path = next(OUT.glob("18_process_profile__Hospital_Billing*.json"), None)
    if profile_path is None:
        check("profile JSON written", False, "no file produced")
        summarise()
        return
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    caps = profile["capabilities"]

    # ---- 2. resource column auto-detected --------------------------------
    check("resource column auto-detected (not assumed 'org:group')",
          caps["resource_column"] == "org:resource",
          f"found '{caps['resource_column']}'")

    # ---- 3. granularity classified, meaningless split withheld -----------
    check("resource granularity classified as individual-like",
          caps.get("resource_granularity") == "individual-like",
          f"same-resource transitions = "
          f"{caps.get('same_resource_transition_share', 0) * 100:.2f}%")
    check("internal-vs-handoff split correctly marked NOT meaningful",
          caps.get("handoff_split_meaningful") is False)
    # Sepsis' org:group is 100% filled (CLAUDE.md records it as an established
    # fact), and that quietly became an assumption. This log is ~45% empty.
    check("incomplete resource column measured and reported",
          caps.get("resource_missing_share", 0) > 0.05,
          f"{caps.get('resource_missing_share', 0) * 100:.1f}% of events have no "
          f"resource (Sepsis: 0.0%)")

    # ---- 4. episode boundaries ------------------------------------------
    ep = profile["episodes"]
    check("no episode boundary invented on a log that has none",
          not ep["episode_end_activities"] and not ep["episode_restart_activities"],
          f"ends={ep['episode_end_activities']}, "
          f"restarts={ep['episode_restart_activities']}")
    check("outlier threshold adapted to this log's own timescale",
          ep["outlier_threshold_h"] > 100,
          f"{ep['outlier_threshold_h']} h (Sepsis: ~1.09 h)")

    # ---- 5. departments collapse to a readable number of roles -----------
    roles = profile.get("behavioural_roles", {})
    check("many departments collapsed to few behavioural roles",
          roles.get("available") and roles["n_roles"] < roles["n_departments"] / 10,
          f"{roles.get('n_departments')} departments -> {roles.get('n_roles')} roles")

    # ---- 6. the handoff analysis runs end to end -------------------------
    rule_line("6. HANDOFF ANALYSIS (src/03_department_handoffs.py)")
    out_csv = f"outputs/{PREFIX}_handoffs.csv"
    proc3 = run_script(["src/03_department_handoffs.py",
                        "--log", str(HOLDOUT), "--out", out_csv])
    check("03 exits 0 on an unseen log", proc3.returncode == 0,
          proc3.stderr.strip().splitlines()[-1] if proc3.returncode else "")
    check("03 withholds the meaningless split rather than printing it",
          "NOT REPORTED" in proc3.stdout)
    check("03 still produces a handoff ranking",
          "HANDOFFS ONLY" in proc3.stdout)

    # ---- 7. payment_mode analyses skip gracefully ------------------------
    rule_line("7. COHORT ANALYSIS WITHOUT `payment_mode`")
    spec = importlib.util.spec_from_file_location(
        "s5", ROOT / "src" / "11_cashless_vs_selfpay.py")
    s5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s5)

    real_shaped = pd.DataFrame({
        "case:concept:name": ["1"], "concept:name": ["x"],
        "time:timestamp": pd.to_datetime(["2024-01-01"], utc=True),
        "lifecycle:transition": ["complete"]})
    try:
        s5.discharge_windows(real_shaped)
        check("cashless-vs-self-pay refuses a log with no payment_mode", False,
              "it did not refuse - would fail later with KeyError")
    except SystemExit as exc:
        check("cashless-vs-self-pay refuses a log with no payment_mode, "
              "with an explanation", "payment_mode" in str(exc))
    except KeyError:
        check("cashless-vs-self-pay refuses cleanly", False,
              "raised a bare KeyError instead of explaining")

    profiler = load_profiler()
    check("segment_available() reports payment_mode absent",
          profiler.segment_available(real_shaped, "payment_mode") is False)

    # ---- 7b. evidence extraction + guard on the unseen log ---------------
    rule_line("7b. EVIDENCE CHAIN (src/20_extract_findings.py -> src/15 guard)")
    proc20 = run_script(["src/20_extract_findings.py", "--log", str(HOLDOUT),
                         "--prefix", f"{PREFIX}_ev"])
    check("20 extracts findings from an unseen log", proc20.returncode == 0,
          proc20.stderr.strip().splitlines()[-1] if proc20.returncode else "")

    ev_csv = OUT / f"{PREFIX}_ev_findings.csv"
    if ev_csv.exists():
        narrator = load_module_by_path("15_narrate_findings.py", "narrator")
        narrator.FINDINGS_CSV = ev_csv
        payload = narrator.build_payload()

        # The gate must derive MORE prohibitions here than on Sepsis: this log
        # additionally cannot support the handoff split or a whole-process total.
        check("gate derives extra prohibitions for this log's weaknesses",
              len(payload["forbidden_claims"]) >= 4,
              f"{len(payload['forbidden_claims'])} forbidden claims "
              f"(Sepsis: 2)")

        # Each prohibition must be ENFORCED, not merely stated. These two were
        # accepted before `check_forbidden_claims()` existed.
        probes = [
            ("the delay is queueing rather than processing", "X-LIFECYCLE"),
            ("time lost between departments accounts for most of the delay",
             "X-GRANULARITY"),
            ("the backlog is caused by the billing team", "X-CAUSE"),
        ]
        enforced = []
        for text, expected in probes:
            report = narrator.verify_narration(text, payload)
            hit = [v["id"] for v in report.get("forbidden_claims_made", [])]
            enforced.append(not report["ok"] and expected in hit)
        check("every derived prohibition is mechanically enforced",
              all(enforced), f"{sum(enforced)}/{len(probes)} probes rejected")

        # ...and it must not reject a sentence quoting the payload verbatim.
        true_claim = next(
            (f["finding"] for f in payload["findings"]
             if f["id"] == "T2"), "")
        if true_claim:
            report = narrator.verify_narration(true_claim, payload)
            check("a verbatim payload statement is NOT rejected",
                  report["ok"],
                  "" if report["ok"] else
                  f"false positive: {report.get('forbidden_claims_made')}")

    # ---- 8. no Sepsis vocabulary leaked into holdout artifacts -----------
    rule_line("8. NO SEPSIS VOCABULARY IN HOLDOUT OUTPUT (the anti-hard-coding test)")
    if args.with_maps:
        print("  rendering maps (slow) ...")
        proc5 = run_script(["src/05_process_map.py", "--log", str(HOLDOUT),
                            "--prefix", PREFIX])
        check("05 renders maps for an unseen log", proc5.returncode == 0,
              proc5.stderr.strip().splitlines()[-1] if proc5.returncode else "")
    else:
        print("  (skipping map rendering; pass --with-maps to include it)")

    texts = {"profiler stdout": proc.stdout, "03 stdout": proc3.stdout,
             "profile JSON": profile_path.read_text(encoding="utf-8")}
    csv_path = ROOT / out_csv
    if csv_path.exists():
        texts["03 handoff CSV"] = csv_path.read_text(encoding="utf-8")

    leaks = []
    for where, text in texts.items():
        for name in SEPSIS_ONLY:
            if name in text:
                leaks.append(f"{where}: '{name}'")
    check("no Sepsis-only activity name appears in any holdout artifact",
          not leaks, "; ".join(leaks[:4]) if leaks else
          f"checked {len(SEPSIS_ONLY)} names across {len(texts)} artifacts")

    summarise()


def summarise() -> None:
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    rule_line(f"HOLDOUT TEST RESULT: {passed}/{total} checks passed")
    for name, ok, detail in results:
        if not ok:
            print(f"  FAILED: {name}  {detail}")
    if passed == total:
        print("\n  The core pipeline accepted a hospital process log it had never")
        print("  seen, without predefined stage names, resource semantics or a")
        print("  payment field - and declined to report the one statistic that")
        print("  log cannot support.")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
