"""
STEP 3a - Make the calibration rule enforceable instead of aspirational.

The project's standing design rule is that every parameter carries a source.
`config/calibration.yaml` implements that, and adds a `status` field:

    verified    - primary document read, or measured by us from the real log
    secondary   - a real citation, but the original not read in context
    placeholder - a working guess with no source yet

The problem this script solves: once a number is a float inside a Python file,
a guess and a verified figure look exactly the same. Simulation studies go wrong
precisely there - a plausible-looking constant nobody can trace, quoted in a
conclusion as though it were measured.

So this script walks the calibration file, counts what is actually sourced, and
REFUSES to certify a run while any parameter is still a placeholder. It is also
the loader the Step 3 generator imports, so there is exactly one way into these
numbers and it is a way that checks itself.

Run it directly for the report:
    python src/06_check_calibration.py
"""

from pathlib import Path
from typing import Any, Iterator

import yaml

ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_PATH = ROOT / "config" / "calibration.yaml"

STATUSES = ("verified", "secondary", "assumption", "placeholder")


def load_calibration(path: Path = CALIBRATION_PATH) -> dict:
    """Read the calibration file. The only supported way to get these numbers."""
    if not path.exists():
        raise SystemExit(f"Calibration file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def walk_parameters(node: Any, trail: tuple[str, ...] = ()) -> Iterator[tuple[str, dict]]:
    """
    Yield every (dotted.name, block) in the file that declares a `status`.

    A "parameter" is any mapping carrying a `status` key. Walking recursively
    rather than hard-coding the section names means a parameter added later is
    picked up automatically - it cannot be introduced without being counted.
    """
    if isinstance(node, dict):
        if "status" in node:
            yield ".".join(trail), node
            return                      # do not descend into a parameter block
        for key, value in node.items():
            yield from walk_parameters(value, trail + (str(key),))


def rule_line(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def require_verified(calibration: dict, dotted_name: str) -> Any:
    """
    Fetch a parameter's value, refusing to hand back anything unverified.

    The Step 3 generator calls this for any number that reaches a headline
    result, so an unsourced guess cannot quietly become a finding.
    """
    for name, block in walk_parameters(calibration):
        if name == dotted_name:
            if block["status"] != "verified":
                raise ValueError(
                    f"'{dotted_name}' is status '{block['status']}', not verified.\n"
                    f"  source: {block.get('source', '(none)')}\n"
                    "  Fix the calibration file rather than the caller."
                )
            return block["value"]
    raise KeyError(f"No calibration parameter named '{dotted_name}'")


def main() -> None:
    cal = load_calibration()
    params = list(walk_parameters(cal))

    rule_line("CALIBRATION AUDIT - what is actually sourced?")
    counts = {s: 0 for s in STATUSES}
    for _, block in params:
        counts[block["status"]] = counts.get(block["status"], 0) + 1

    total = len(params)
    for status in STATUSES:
        n = counts.get(status, 0)
        bar = "#" * n
        print(f"  {status:<12} {n:>3} of {total}  {bar}")

    for status, heading in [
        ("verified", "VERIFIED - primary document read, or measured by us"),
        ("secondary", "SECONDARY - real citation, original not read in context"),
        ("assumption", "ASSUMPTION - declared modelling choice; results are "
                       "CONDITIONAL on these"),
        ("placeholder", "PLACEHOLDER - no source yet, BLOCKS certification"),
    ]:
        rows = [(n, b) for n, b in params if b["status"] == status]
        if not rows:
            continue
        rule_line(heading)
        for name, block in rows:
            print(f"  {name}")
            print(f"      source: {block.get('source', '(none)')}")
            # An assumption standing in for something that should be sourced does
            # NOT discharge that obligation - surface the link rather than let it
            # quietly close the gap.
            if block.get("stands_in_for"):
                print(f"      stands in for: {block['stands_in_for']}  "
                      "<- still needs a real source")
            if block.get("sweep"):
                print(f"      SWEPT over: {block['sweep']}  "
                      "<- never quote a single value")

    rule_line("VERDICT")
    blocking = [n for n, b in params if b["status"] == "placeholder"]
    if not blocking:
        print("  CERTIFIED. Every calibration parameter carries a source.")
        return

    print(f"  NOT CERTIFIED - {len(blocking)} placeholder parameter(s) remain.\n")
    print("  Any result that depends on these must NOT be quoted as a finding:")
    for name in blocking:
        print(f"    - {name}")
    print(
        "\n  This is the intended state at the start of Step 3, not a failure. The\n"
        "  point of the file is that the gaps are countable rather than hidden in\n"
        "  code. Two honest ways to clear one:\n"
        "    1. Find the source and change the status.\n"
        "    2. Declare it a SENSITIVITY PARAMETER and sweep a range in Step 5,\n"
        "       reporting 'under assumption X the effect is Y' - never a point\n"
        "       estimate dressed up as a measurement.\n"
        "\n  The regulatory deadlines and everything measured from the real log are\n"
        "  already verified, so the generator's SKELETON can be built now. What it\n"
        "  cannot do yet is produce a quotable headline number."
    )


if __name__ == "__main__":
    main()
