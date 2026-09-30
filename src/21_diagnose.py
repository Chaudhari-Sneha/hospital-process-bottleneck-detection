"""
STEP 11 - THE DIAGNOSIS LAYER. Interpretation, with every claim traceable.

WHAT THIS IS FOR, AND HOW IT DIFFERS FROM STEP 10 (`15_narrate_findings.py`)

`15` turns findings into PROSE. Prose is hard to verify: you are pattern-matching
sentences, and a fluent paragraph can smuggle an unsupported claim past a reader
who is nodding along.

Diagnosis is a different job, and it deserves a different shape. The questions a
hospital administrator actually asks are:

    Where is the bottleneck?  What is the evidence?  What is probably driving it?
    What should we do?  And why is that the relevant thing to do?

Answering in STRUCTURED JSON rather than prose changes what can be checked. Every
claim carries `evidence_ids`, and this module asserts mechanically that each one
exists in the findings file. A recommendation citing nothing is not judged to be
weak - it is REJECTED.

THE DIVISION OF LABOUR IS UNCHANGED

    Python/pm4py    computes every number          (03, 18)
    20              states them as findings         with caveats and prohibitions
    the model       interprets, hypothesises, recommends - and computes NOTHING
    this module     verifies that it did exactly that

THREE THINGS THE SCHEMA ENFORCES THAT PROSE CANNOT

  1. CAUSES ARE HYPOTHESES, NOT CONCLUSIONS.
     There is no field for "the cause is". `X-CAUSE` forbids asserting causation
     from an observational log, so the schema makes the honest form the only
     expressible one. You cannot accidentally write the dishonest version.

  2. EVERY HYPOTHESIS CARRIES `how_to_test`.
     A cause you cannot test is an opinion. Forcing the field turns each guess
     into the administrator's next concrete action.

  3. THE GUARD IS REUSED, NOT REWRITTEN.
     `15`'s number check and `check_forbidden_claims()` run over the model's JSON.
     A second implementation of a safety check is a second thing to keep in sync,
     and the copy is always the one that rots.

RUN IT

    python src/21_diagnose.py --self-test        # proves the checks have teeth
    python src/21_diagnose.py --findings outputs/20_findings.json --diagnose \
        --provider openai-compat --base-url http://localhost:11434/v1 \
        --model llama3.1-16k
"""

import argparse
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
DEFAULT_FINDINGS = OUT / "20_findings.json"


def load_narrator():
    """Reuse `15`'s guard rather than reimplementing it."""
    spec = importlib.util.spec_from_file_location(
        "narrator", ROOT / "src" / "15_narrate_findings.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rule_line(title: str) -> None:
    print(f"\n{'=' * 92}\n{title}\n{'=' * 92}")


# ---------------------------------------------------------------------------
# 1. THE CONTRACT - what the model must return
# ---------------------------------------------------------------------------
SCHEMA_DESCRIPTION = """{
  "bottleneck": {
    "summary": "<one sentence naming where time is lost>",
    "evidence_ids": ["<finding id>", ...]
  },
  "probable_drivers": [
    {
      "hypothesis": "<what may be causing it - stated as a possibility>",
      "evidence_ids": ["<finding id>", ...],
      "confidence": "low" | "medium" | "high",
      "how_to_test": "<what data or check would confirm or refute this>"
    }
  ],
  "recommendations": [
    {
      "action": "<what to do>",
      "why_relevant": "<why this addresses the evidence above>",
      "evidence_ids": ["<finding id>", ...],
      "effort": "low" | "medium" | "high"
    }
  ]
}"""

# ---------------------------------------------------------------------------
# THE INTERVENTION CATALOGUE
# ---------------------------------------------------------------------------
# Why this exists: the model was being asked to recommend a fix while being given
# no vocabulary of fixes. It received "step X absorbs N hours" and had to invent
# an intervention from nothing, so it produced tautologies - "review and simplify
# the process for step X", "the step is slow due to a time-consuming process".
# That is not the model being stupid; it is the only safe thing to say when you
# have measurements and no domain knowledge.
#
# So: a catalogue of process-improvement levers, each keyed to the EVIDENCE
# SIGNATURE that suggests it. The findings already carry these signatures - a
# median of zero next to a large total is the fingerprint of batching, and so on.
#
# THE CRITICAL CONSTRAINT. This makes hypotheses SPECIFIC; it must not turn them
# into CLAIMS. Every entry states the evidence that must be present before it
# applies, the mechanism stays a candidate, and `how_to_test` remains mandatory.
# The X-CAUSE prohibition is unaffected: an observational log still cannot
# establish why anything happened. The catalogue changes what the model can
# usefully GUESS, not what it is allowed to ASSERT.
#
# It contains no numbers, so it cannot contaminate the numeric guard.
INTERVENTION_CATALOGUE = """
CANDIDATE MECHANISMS AND LEVERS, each with the evidence signature that suggests
it. Use an entry ONLY if its signature is present in the findings you were given.
If none fits, say so rather than reaching for the closest one.

1. BATCHING / SCHEDULED RELEASE
   Signature: a step with a large total but a median at or near zero, or a median
   far below its p90. Work is held and released together, so most instances record
   almost no elapsed time and a minority carry all of it.
   Levers: shorten the release interval; release continuously rather than in
   rounds; decouple the downstream step from the batch boundary.
   Test: check whether completion timestamps cluster at regular times of day.

2. A RARE SLOW PATH
   Signature: a high median or p90 on a step with FEW occurrences.
   Levers: this is an exception route - find what puts cases on it, and whether
   they could be identified earlier; do not redesign the main path for it.
   Test: compare the characteristics of cases that take this path against those
   that do not.

3. REWORK / FAILURE CAUGHT LATE
   Signature: a high share of immediately repeated steps (self-loops), or the same
   activity recurring within a case.
   Levers: move the check upstream to where the error is introduced; fix the
   input quality rather than the repeat; ask what makes the first attempt fail.
   Test: whether repeats concentrate in particular case types or originators.

4. COORDINATION GAP AT A HANDOFF
   Signature: a disproportionate step whose two ends sit in DIFFERENT departments.
   Levers: nobody owns the wait between two teams - assign a single owner, agree
   a response-time expectation, or make the handoff a pull rather than a push.
   Test: whether the delay sits before or after the receiving team is notified.

5. A GENUINE CONSTRAINT
   Signature: a step disproportionate to its frequency, with a high median, inside
   ONE department.
   Levers: capacity, scheduling or protocol at that step; remove work from it
   before adding resource to it.
   Test: whether the delay scales with arrivals - if it does, it is load-related.

6. RETURNS / RE-ENTRY
   Signature: cases re-entering the process after ending it.
   Levers: look at what was unresolved at first exit, and whether the exit
   criteria are the problem rather than the re-entry.
   Test: compare first-visit characteristics of returning against non-returning
   cases.

CONSTRAINT ON STAFFING RECOMMENDATIONS
If `forbidden_claims` contains an entry about lifecycle or start timestamps, the
log CANNOT distinguish time queueing from time being worked on. You therefore
cannot tell whether more staff would help. Do not recommend adding capacity on
the basis of queueing; recommend removing steps, or the measurement that would
settle it.
"""

SYSTEM_PROMPT = f"""You are advising a hospital administrator on process \
improvement.

You will receive FINDINGS measured from an event log, and a list of claims that \
this particular log CANNOT support.

Your job is interpretation, not calculation:
  - identify where the process loses time, citing the findings
  - propose what may be driving it, as HYPOTHESES
  - recommend what to do, and say why each recommendation follows from the evidence

ABSOLUTE RULES
1. Do NOT introduce any number that is not present verbatim in the findings.
   Do not compute averages, totals, percentages, rankings or differences.
2. Cite finding ids in `evidence_ids` for every claim. A claim you cannot tie to
   a finding id must be omitted.
3. Never state a cause as fact. This is an observational log; it records when
   things happened, not why. Use the `probable_drivers` list, and give each entry
   a real `how_to_test`.
4. Do not make any claim listed under `forbidden_claims`.

HOW TO USE THE CATALOGUE BELOW
It lists candidate mechanisms and the evidence signature that suggests each one.
Match the findings you were given against those signatures.
  - A matching signature makes a hypothesis PLAUSIBLE. It does not make it true,
    and it is not evidence. Keep it in `probable_drivers` with a real test.
  - Prefer a step that is disproportionate to how often it happens over one that
    is merely the largest total. A step can dominate the total purely by
    occurring constantly, and "it happens a lot" is not a problem to fix.
  - Say plainly if no signature matches. "The evidence does not distinguish
    between these" is a useful answer; inventing a mechanism is not.
  - Recommend the LEVER, not the investigation. "Review and analyse step X" is
    not a recommendation - it is what the administrator asked you to do. Never
    copy an entry's "Test:" line into an `action`; the test belongs in
    `how_to_test` and the action is what you would change.

THE SIGNATURE IS GIVEN TO YOU - DO NOT RE-DERIVE IT
Each bottleneck finding says "Classified as X because ...". That classification
was computed from the numbers, not inferred, and it is correct. Use the catalogue
entry matching the stated signature. Do not decide for yourself that a step is
batched or rare; if a finding says "handoff", use the handoff entry even if
another looks appealing. A step occurring thousands of times is never a rare path,
and the classification already accounts for that.

HOW MANY TO GIVE
At most THREE drivers and THREE recommendations, and ONE PER DISTINCT SIGNATURE.
Do not walk down the findings assigning a mechanism to each - if five steps are
all classified "handoff", that is ONE problem affecting five steps, so say it once
and name them together. But if the findings contain a different signature as well,
cover that too: "rework" and "handoff" are different problems needing different
fixes, and reporting only the largest would hide one of them. Work down the
distinct signatures present, most disproportionate first.

{INTERVENTION_CATALOGUE}

Reply with JSON ONLY, in exactly this shape, and nothing else:

{SCHEMA_DESCRIPTION}
"""


# ---------------------------------------------------------------------------
# 2. THE VERIFIER
# ---------------------------------------------------------------------------
def extract_json(text: str) -> tuple[dict | None, str]:
    """
    Pull the JSON object out of a model reply.

    Models wrap JSON in prose or fences however much you ask them not to, and a
    parse failure is not a safety problem - it is a formatting one. Recovering
    the object keeps a formatting slip from being reported as a rule violation.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, depth = text.find("{"), 0
        if start == -1:
            return None, "no JSON object found in the reply"
        for i in range(start, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                candidate = text[start:i + 1]
                break
        if candidate is None:
            return None, "JSON object is not closed"
    try:
        return json.loads(candidate), ""
    except ValueError as exc:
        return None, f"JSON did not parse: {exc}"


def claims_in(diagnosis: dict) -> list:
    """Flatten every claim so each can be checked for its evidence."""
    claims = []
    b = diagnosis.get("bottleneck") or {}
    if b:
        claims.append(("bottleneck", b.get("summary", ""), b.get("evidence_ids")))
    for d in diagnosis.get("probable_drivers") or []:
        claims.append(("probable_driver", d.get("hypothesis", ""),
                       d.get("evidence_ids")))
    for r in diagnosis.get("recommendations") or []:
        claims.append(("recommendation", r.get("action", ""), r.get("evidence_ids")))
    return claims


def all_text(diagnosis: dict) -> str:
    """Every string in the reply, for the number and prohibition checks."""
    parts = []

    def walk(node):
        if isinstance(node, str):
            parts.append(node)
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(diagnosis)
    return " ".join(parts)


def verify_diagnosis(diagnosis: dict, payload: dict, narrator) -> dict:
    """
    The test. `ok` is False if the diagnosis must be rejected.

    Four independent checks, each catching a different failure:
      - structure     : the model answered the question it was asked
      - citations     : every claim points at a finding that exists
      - numbers       : nothing was computed (reuses `15`)
      - prohibitions  : nothing the log cannot support was claimed (reuses `15`)
    """
    valid_ids = {f["id"] for f in payload.get("findings", [])}

    missing_sections = [k for k in ("bottleneck", "probable_drivers",
                                    "recommendations") if not diagnosis.get(k)]

    uncited, unknown_ids = [], []
    for kind, text, ids in claims_in(diagnosis):
        if not ids:
            uncited.append(f"{kind}: {text[:60]}")
            continue
        for fid in ids:
            if fid not in valid_ids:
                unknown_ids.append(f"{kind} cites '{fid}'")

    # A driver without a test is an opinion wearing a hypothesis costume.
    untestable = [d.get("hypothesis", "")[:60]
                  for d in diagnosis.get("probable_drivers") or []
                  if not (d.get("how_to_test") or "").strip()]

    text = all_text(diagnosis)
    allowed = narrator.collect_payload_numbers(payload)
    unsupported = [v for v in narrator.numbers_in(text)
                   if not narrator.number_is_supported(v, allowed)]
    forbidden = narrator.check_forbidden_claims(text.lower(), payload)

    return {
        "ok": not (missing_sections or uncited or unknown_ids or untestable
                   or unsupported or forbidden),
        "missing_sections": missing_sections,
        "claims_checked": len(claims_in(diagnosis)),
        "uncited_claims": uncited,
        "unknown_evidence_ids": unknown_ids,
        "untestable_hypotheses": untestable,
        "unsupported_numbers": sorted(set(unsupported)),
        "forbidden_claims_made": forbidden,
    }


# ---------------------------------------------------------------------------
# 3. SELF-TEST - proves the checks have teeth, with no model and no API key
# ---------------------------------------------------------------------------
def build_self_tests(valid_id: str, real_number: str) -> list:
    """Built from the ACTUAL payload, so the tests cannot drift from the data."""
    good = {
        "bottleneck": {"summary": f"Most time sits in one step ({real_number}).",
                       "evidence_ids": [valid_id]},
        "probable_drivers": [{"hypothesis": "Results may be entered in batches.",
                              "evidence_ids": [valid_id], "confidence": "low",
                              "how_to_test": "Check whether entry timestamps cluster."}],
        "recommendations": [{"action": "Review the batching schedule.",
                             "why_relevant": "It targets the step holding the time.",
                             "evidence_ids": [valid_id], "effort": "low"}],
    }

    def variant(**changes):
        import copy
        d = copy.deepcopy(good)
        for k, v in changes.items():
            d[k] = v
        return d

    return [
        ("a well-formed diagnosis citing real findings", good, True),
        ("a recommendation citing NO evidence",
         variant(recommendations=[{"action": "Hire more staff.",
                                   "why_relevant": "It would help.",
                                   "evidence_ids": [], "effort": "high"}]), False),
        ("a claim citing an evidence id that does not exist",
         variant(bottleneck={"summary": "Time is lost at intake.",
                             "evidence_ids": ["Z99"]}), False),
        ("a hypothesis with no way to test it",
         variant(probable_drivers=[{"hypothesis": "Staff are unmotivated.",
                                    "evidence_ids": [valid_id],
                                    "confidence": "high", "how_to_test": ""}]), False),
        ("a fabricated number",
         variant(bottleneck={"summary": "Fully 91.7% of time is lost here.",
                             "evidence_ids": [valid_id]}), False),
        ("a cause asserted as fact rather than hypothesised",
         variant(bottleneck={"summary": "The delay is caused by the laboratory.",
                             "evidence_ids": [valid_id]}), False),
        ("an empty reply", {}, False),
    ]


def run_self_test(payload: dict, narrator) -> bool:
    rule_line("DIAGNOSIS SELF-TEST - do the checks actually reject bad output?")

    findings = payload.get("findings", [])
    if not findings:
        raise SystemExit("Payload has no findings - run src/20_extract_findings.py")
    valid_id = findings[0]["id"]
    numbers = sorted(narrator.collect_payload_numbers(payload))
    real_number = f"{numbers[-1]:g}" if numbers else "0"

    passed = 0
    tests = build_self_tests(valid_id, real_number)
    for label, diagnosis, should_pass in tests:
        report = verify_diagnosis(diagnosis, payload, narrator)
        ok = (report["ok"] == should_pass)
        passed += ok
        verdict = "ACCEPTED" if report["ok"] else "REJECTED"
        print(f"  [{'pass' if ok else 'FAIL'}] {verdict:<8}  {label}")
        if not report["ok"]:
            for key in ("missing_sections", "uncited_claims", "unknown_evidence_ids",
                        "untestable_hypotheses", "unsupported_numbers"):
                if report[key]:
                    print(f"              {key}: {str(report[key])[:70]}")
            if report["forbidden_claims_made"]:
                print("              forbidden: " + ", ".join(
                    v["id"] for v in report["forbidden_claims_made"]))

    print(f"\n  {passed}/{len(tests)} checks behaved as intended")
    return passed == len(tests)


# ---------------------------------------------------------------------------
# 4. THE MODEL CALL
# ---------------------------------------------------------------------------
def build_prompt_payload(findings_json: dict) -> dict:
    """What the model is allowed to see. The log itself is never included."""
    return {
        "findings": [
            {"id": f["id"], "category": f["category"], "finding": f["finding"],
             "caveat": f["caveat"]}
            for f in findings_json["findings"]
            if not f["category"].startswith("5")
        ],
        "forbidden_claims": findings_json.get("forbidden_claims", []),
    }


def render_markdown(diagnosis: dict, findings_json: dict, backend: str) -> str:
    """A readable version, with every citation kept visible."""
    by_id = {f["id"]: f["finding"] for f in findings_json["findings"]}
    lines = [f"# Diagnosis - {findings_json.get('log', 'unknown log')}", ""]
    lines.append(f"_Generated by `{backend}`. Every claim below cites a finding id; "
                 "the ids are checked to exist before this file is written. "
                 "No number here was produced by the model._")
    lines.append("")

    b = diagnosis.get("bottleneck", {})
    lines += ["## Where the time goes", "", b.get("summary", ""), "",
              "Evidence: " + ", ".join(f"`{i}`" for i in b.get("evidence_ids", [])), ""]

    lines += ["## Probable drivers (hypotheses, not conclusions)", ""]
    for d in diagnosis.get("probable_drivers", []):
        lines += [f"**{d.get('hypothesis', '')}** "
                  f"(confidence: {d.get('confidence', '?')})",
                  f"- How to test: {d.get('how_to_test', '')}",
                  "- Evidence: " + ", ".join(f"`{i}`" for i in d.get("evidence_ids", [])),
                  ""]

    lines += ["## Recommendations", ""]
    for r in diagnosis.get("recommendations", []):
        lines += [f"**{r.get('action', '')}** (effort: {r.get('effort', '?')})",
                  f"- Why relevant: {r.get('why_relevant', '')}",
                  "- Evidence: " + ", ".join(f"`{i}`" for i in r.get("evidence_ids", [])),
                  ""]

    lines += ["## Findings cited", ""]
    cited = {i for _, _, ids in claims_in(diagnosis) for i in (ids or [])}
    for fid in sorted(cited):
        lines.append(f"- `{fid}` - {by_id.get(fid, '(unknown)')}")

    blocked = findings_json.get("forbidden_claims", [])
    if blocked:
        lines += ["", "## Claims this log cannot support", ""]
        for b in blocked:
            lines.append(f"- **{b['id']}** - must not claim {b['must_not_claim']}. "
                         f"{b['because']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Structured diagnosis over findings.")
    ap.add_argument("--findings", default=str(DEFAULT_FINDINGS),
                    help="findings JSON from src/20_extract_findings.py")
    ap.add_argument("--self-test", action="store_true",
                    help="verify the checks reject bad output (no model needed)")
    ap.add_argument("--diagnose", action="store_true",
                    help="actually call a model")
    ap.add_argument("--provider", default="openai-compat",
                    choices=["anthropic", "openai-compat"])
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = Path(args.findings)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise SystemExit(
            f"Not found: {path}\n  Run: python src/20_extract_findings.py")

    findings_json = json.loads(path.read_text(encoding="utf-8"))
    narrator = load_narrator()

    # The payload the GUARD checks against must be the same shape `15` expects,
    # so its number collection and prohibition patterns apply unchanged.
    payload = {
        "findings": findings_json["findings"],
        "forbidden_claims": findings_json.get("forbidden_claims", []),
    }

    rule_line(f"DIAGNOSIS INPUT - {findings_json.get('log', path.name)}")
    print(f"  findings available : {len(payload['findings'])}")
    print(f"  forbidden claims   : {len(payload['forbidden_claims'])}")
    print(f"  distinct numbers   : {len(narrator.collect_payload_numbers(payload))}")

    if args.self_test or not args.diagnose:
        ok = run_self_test(payload, narrator)
        if not args.diagnose:
            raise SystemExit(0 if ok else 1)

    # ---- live model call --------------------------------------------------
    prompt_payload = build_prompt_payload(findings_json)
    rule_line("CALLING THE MODEL")
    if args.provider == "openai-compat":
        model = args.model or "llama3.1-16k"
        reply = narrator.narrate_openai_compatible(
            prompt_payload, args.base_url, model, args.api_key_env,
            system_prompt=SYSTEM_PROMPT)
        backend = f"{model} via {args.base_url}"
    else:
        reply = narrator.narrate(prompt_payload, system_prompt=SYSTEM_PROMPT)
        backend = "claude-opus-5 via Anthropic API"

    diagnosis, err = extract_json(reply)
    if diagnosis is None:
        print(f"\n  REJECTED - {err}")
        print("  Raw reply (first 400 chars):\n")
        # The Windows console is cp1252 and raises on characters a model will
        # readily emit - a non-breaking hyphen, an arrow, an em dash. Crashing
        # while REPORTING a failure hides the failure behind a traceback about
        # encoding, which is a worse bug than the one being reported.
        safe = reply[:400].encode("ascii", "replace").decode("ascii")
        print("    " + safe.replace("\n", "\n    "))
        raise SystemExit(1)

    report = verify_diagnosis(diagnosis, payload, narrator)
    rule_line("VERIFICATION")
    print(f"  claims checked : {report['claims_checked']}")
    for key in ("missing_sections", "uncited_claims", "unknown_evidence_ids",
                "untestable_hypotheses", "unsupported_numbers"):
        if report[key]:
            print(f"  {key}: {report[key]}")
    for v in report["forbidden_claims_made"]:
        # Show WHAT was said, not just that something was. A rejection you cannot
        # inspect is one nobody can act on - and on an unseen log this is the
        # first place a reader learns which of its weaknesses actually bit.
        print(f"  forbidden claim {v['id']} - matched '{v['matched']}'")
        print(f"    because: {v['because'][:110]}")

    if not report["ok"]:
        print("\n  REJECTED. Nothing written - an unverified diagnosis is worse")
        print("  than none, because it reads exactly like a verified one.")
        # Keep the rejected output for inspection. A rejection is evidence that
        # the guard works, and it is worth being able to show what was caught.
        rejected = OUT / f"{path.stem}_diagnosis_REJECTED.json"
        rejected.write_text(json.dumps(
            {"verification": report, "diagnosis": diagnosis}, indent=2),
            encoding="utf-8")
        print(f"  rejected output kept at {rejected.relative_to(ROOT)}")
        raise SystemExit(1)

    out_path = Path(args.out) if args.out else OUT / f"{path.stem}_diagnosis.md"
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.write_text(render_markdown(diagnosis, findings_json, backend),
                        encoding="utf-8")
    (out_path.with_suffix(".json")).write_text(
        json.dumps(diagnosis, indent=2), encoding="utf-8")
    print(f"\n  ACCEPTED - wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
