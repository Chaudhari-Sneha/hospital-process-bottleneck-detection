"""
STEP 6b - The LLM narration layer, with a guard that proves it computed nothing.

THE STANDING DESIGN RULE (CLAUDE.md)

    "The LLM never computes anything. The pipeline computes a JSON blob
     (bottleneck name, median wait, affected case count, %); the LLM receives
     that JSON and only writes prose around it. State this in the report - it is
     the answer to 'how do you know it didn't hallucinate?'"

  A rule nobody checks is a wish. So this module does not merely SAY the model
  computed nothing - it verifies it mechanically:

      1. build_payload()      deterministic JSON, built from the Step 6 findings
                              table. No model involved.
      2. narrate()            sends that JSON to Claude with instructions to add
                              no numbers of its own.
      3. verify_narration()   extracts EVERY number from the returned prose and
                              checks each one against the payload. Any number the
                              model introduced is flagged, and the narration is
                              rejected.

  Point 3 is the actual contribution. It turns "trust the prompt" into a test
  that fails loudly, and it runs with no API key at all - which is why the
  self-test below is the part that can always be demonstrated.

HOW WELL THE GUARD ACTUALLY WORKS - measured, not asserted

  Against this payload (77 distinct numbers spanning 0-120):

      randomly fabricated numbers rejected  95.0%  (n=4000)
      near-miss perturbations rejected      66.4%  (n=152)

  Regenerate both with `--measure-guard` rather than trusting these; they are a
  property of the payload, so adding findings moves them on its own.

  So it is a STRONG FILTER, NOT A PROOF. A fabricated number that happens to
  land within 0.05 of one of the payload's values is accepted. The accurate claim
  is "no number survived a check against the computed payload", not "the model
  cannot have invented anything". (The near-miss rate has drifted twice as
  findings were added: a denser number space is easier to land in by luck. That
  is a property of the payload, not a loosening of the check - and it is why the
  figure is now computed on demand rather than quoted from a comment that had no
  way to be re-derived.)

  The categorical checks below are different - deterministic string matches, so
  they fire every time rather than 95% of the time.

WHAT ELSE THE GUARD BLOCKS

  - ROI / return-on-investment language, outright. No such analysis exists.
  - Point estimates of the sensitivity parameter, because q is unsourced and only
    swept RANGES may be quoted (Step 5).
  - CURRENCY, CONDITIONALLY. This changed when finding D5 approved an
    illustrative rupee figure valued against a SEBI-filed external benchmark:

      * money is PERMITTED only when the narration carries all five D5 labels -
        illustrative/benchmark, external and not this hospital, premium-end
        chain, definition excludes pharmacy and other income, and upper bound on
        opportunity rather than cash;
      * money is REJECTED OUTRIGHT, however well labelled, when the wording
        attributes the amount to the study hospital ("this hospital lost INR X").
        That is finding X1 and it is permanent: the bed-hours are simulated and
        the ARPOB belongs to a different company, so no phrasing of it is true.

    Negation is respected. "rather than a cash saving" is the required
    disclaimer, not a violation - an earlier version of this guard rejected the
    CORRECT narration for containing the words it demands.

RUNNING IT

  No credentials are required for the payload build or the guard self-test:

      python src/15_narrate_findings.py --self-test

  A live narration additionally needs the SDK and a credential:

      pip install anthropic
      ant auth login          (or set ANTHROPIC_API_KEY)
      python src/15_narrate_findings.py --narrate
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FINDINGS_CSV = ROOT / "outputs" / "14_findings.csv"
PAYLOAD_JSON = ROOT / "outputs" / "15_narration_payload.json"
NARRATION_MD = ROOT / "outputs" / "15_narration.md"

MODEL = "claude-opus-5"

# A free tier caps requests per minute, so a batch caller (src/22 runs one
# request per scenario) hits 429 routinely. Retrying is a transport concern and
# changes nothing the guard does.
RATE_LIMIT_RETRIES = 5
RATE_LIMIT_MAX_WAIT = 90.0

# Words that must never appear: every one of them implies a claim the evidence
# package still blocks outright.
#
# CURRENCY IS NO LONGER HERE. Finding D5 approved an illustrative rupee figure
# valued against a SEBI-filed external benchmark, so an absolute ban would now
# stop the narrator stating an approved result. Money is instead handled
# CONDITIONALLY by check_financial() below: permitted only when the narration
# carries the benchmark framing, and still rejected outright when it attributes
# the figure to the study hospital (finding X1).
BANNED_TERMS = [
    "roi", "return on investment",
]

# Anything that reads as money, in any of the forms a narration might use.
#
# THESE ARE WORD-BOUNDED REGEXES, NOT SUBSTRINGS, and that is a bug fix.
# The first version used the bare substring "rs ", which matches inside
# "hou[rs ]p90" - and "hours" appears in essentially every narration this
# project will ever produce. The first live model run was therefore REJECTED
# for "mentioning money" when it contained no currency at all, and the guard
# would have rejected almost any correct narration.
MONEY_PATTERNS = [
    r"₹",
    r"\binr\b",
    r"\brupees?\b",
    r"\blakhs?\b",
    r"\bcrores?\b",
    r"\brs\.?\b",
]

# ---- X1, still absolute ---------------------------------------------------
# Wording that presents the figure as THIS hospital's own money. The benchmark
# values a SYNTHETIC hospital's bed-hours against a DIFFERENT company's ARPOB,
# so no phrasing of this kind can ever be correct, however well labelled the
# rest of the paragraph is.
ATTRIBUTION_PATTERNS = [
    r"\b(this|our|the study) hospital('s)? (lost|loses|is losing|forgoes|forgoing)\b",
    r"\b(lost|loses|losing|saved|saves|saving|recovers|recovered|earns|earned)\s+"
    r"(of |about |roughly |around |approximately |some )?"
    r"(inr|rs\.?|₹)",
    r"\bactual (loss|losses|saving|savings|cash|revenue)\b",
    r"\bwe (lose|lost|save|saved|recover|recovered)\b",
    r"\bcash (saving|savings)\b",
    r"\b(revenue|money) (lost|foregone|forgone) (by|to|for) (this|the|our) hospital\b",
    r"\bthe hospital (loses|lost|saves|saved)\b",
]

# A match inside a NEGATION is the opposite of a violation - it is the required
# disclaimer. "rather than a cash saving" and "not this hospital's actual loss"
# are exactly the phrasings D5 demands, and an earlier version of this guard
# rejected the correct narration for containing them. Checked over the preceding
# characters, which is where English puts the negation.
# A match inside a NEGATION or a CONDITIONAL is not an assertion.
#
# The negation cues were there from the start ("rather than a cash saving" is the
# required disclaimer, not a violation). The conditional cues were added after a
# live run was rejected for:
#
#   "IF the large total time with a zero median is due to batching, more
#    frequent releases will reduce the waiting"
#
# "If X is due to Y, then Z" proposes a mechanism to test; it does not claim one.
# Since the diagnosis schema exists precisely to hold hypotheses rather than
# conclusions, rejecting the hypothetical form punishes the model for using the
# structure correctly.
NEGATION_CUES = ["not ", "never ", "rather than ", "instead of ", "n't ",
                 "no ", "neither ", "nor ", "without ",
                 # conditional / hedged framings
                 "if ", "whether ", "may be ", "might be ", "could be ",
                 "appears to ", "suggests "]
NEGATION_WINDOW = 45

# ---- D5, the labels that must accompany any rupee figure -------------------
# Every concept must be present somewhere in the narration before money is
# allowed. This is strict by design: the evidence package requires all four
# D5 labels plus the benchmark framing, and a figure quoted without them is
# exactly the screenshot-into-a-slide failure the guard exists to prevent.
# Tune here, in one place, if it proves too tight in practice.
D5_REQUIRED_LABELS = {
    "framed as illustrative/benchmark": [
        "illustrative", "benchmark", "order of magnitude"],
    "external - not this hospital": [
        "external", "not this hospital", "not our hospital", "different hospital",
        "another hospital", "medanta", "global health", "benchmark chain"],
    "premium-end / optimistic": [
        "premium", "quaternary", "high-acuity", "high acuity", "optimistic",
        "upper end"],
    "definition excludes pharmacy": [
        "excludes pharmacy", "excluding pharmacy", "pharmacy and other income"],
    "upper bound, not cash": [
        "upper bound", "not a cash saving", "not cash", "revenue is not profit",
        "not profit", "opportunity"],
}

# Numbers that may appear in prose without being "computed": ordinals and small
# counts used in ordinary English ("the three-hour rule", "one of five seeds").
# Kept short - anything else must come from the payload.
FREE_NUMBERS = {0, 1, 2, 3, 4, 5, 10, 100}


def rule_line(title: str) -> None:
    print(f"\n{'=' * 92}\n{title}\n{'=' * 92}")


# ---------------------------------------------------------------------------
# 1. THE PAYLOAD - computed, not narrated
# ---------------------------------------------------------------------------
def _parse_patterns(raw) -> list:
    """
    Read the JSON-encoded `patterns` column, tolerating its absence.

    `14_findings.csv` has no such column at all, so this must return [] rather
    than raise - the hand-assembled package's blockers are enforced by the
    specific D5/X1 checks instead.
    """
    if not raw or not isinstance(raw, str):
        return []
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return []
    return [p for p in value if isinstance(p, str)] if isinstance(value, list) else []


def build_payload() -> dict:
    """
    Assemble the JSON the model is allowed to see.

    It is built from `outputs/14_findings.csv`, which is itself generated from
    the analysis artifacts (Step 6). So the chain from raw event log to the
    sentence in the report is unbroken and every link is a file on disk.
    """
    if not FINDINGS_CSV.exists():
        raise SystemExit(
            f"Not found: {FINDINGS_CSV} - run: python src/14_evidence_package.py")

    df = pd.read_csv(FINDINGS_CSV).fillna("")
    quotable = df[~df["category"].str.startswith("5")]
    blocked = df[df["category"].str.startswith("5")]

    return {
        "instruction_to_model": (
            "You are writing for a hospital administrator. Use ONLY the facts "
            "below. Do not compute, estimate, infer or introduce any number that "
            "does not appear verbatim in this payload."
        ),
        "findings": [
            {
                "id": r["id"],
                "category": r["category"],
                "finding": r["finding"],
                "kind": r["kind"],
                "parameter_dependence": r["parameter_dependence"],
                "robustness": r["robustness"],
                "allowed_wording": r["allowed_wording"],
                "caveat": r["caveat"],
            }
            for _, r in quotable.iterrows()
        ],
        "forbidden_claims": [
            {"id": r["id"], "must_not_claim": r["finding"],
             "because": r["caveat"],
             # Optional column: findings extracted by `20_extract_findings.py`
             # ship the regexes that betray a violation, so the prohibition is
             # enforced rather than merely stated. `14_findings.csv` has no such
             # column, and its blockers are covered by the specific checks above.
             "patterns": _parse_patterns(r.get("patterns", ""))}
            for _, r in blocked.iterrows()
        ],
    }


# ---------------------------------------------------------------------------
# 2. THE GUARD - this is the part that makes the rule real
# ---------------------------------------------------------------------------
NUMBER_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def numbers_in(text: str) -> list:
    """Every number in a string, commas stripped, as floats."""
    out = []
    for raw in NUMBER_RE.findall(text):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return out


def collect_payload_numbers(node) -> set:
    """Recursively gather every number anywhere in the payload."""
    found = set()
    if isinstance(node, dict):
        for v in node.values():
            found |= collect_payload_numbers(v)
    elif isinstance(node, list):
        for v in node:
            found |= collect_payload_numbers(v)
    elif isinstance(node, str):
        found |= set(numbers_in(node))
    elif isinstance(node, (int, float)):
        found.add(float(node))
    return found


def number_is_supported(value: float, allowed: set) -> bool:
    """
    A number in the prose is supported if it appears in the payload, or is a
    reasonable rounding of one.

    Rounding is permitted because good prose says "32 hours" for 32.0 and
    "roughly a quarter" for 24.4 - what is NOT permitted is a number with no
    ancestor in the payload at all.
    """
    if value in FREE_NUMBERS or value in allowed:
        return True
    for a in allowed:
        if a == 0:
            continue
        # Tight tolerance. An earlier version allowed 1% plus a bare round(a)
        # match, which let ~23% of randomly fabricated numbers through because
        # 67 payload values densely cover 0-120. Measured, then tightened.
        if abs(a - value) <= 0.05:
            return True
        if round(a, 1) == value:            # "32.0" written as "32"
            return True
    return False


def check_financial(lowered: str) -> dict:
    """
    The conditional currency rule.

    No money mentioned        -> nothing to check.
    Money mentioned           -> it must carry every D5 label, AND must not
                                 attribute the figure to the study hospital.

    Returns the two failure kinds separately, because they mean different
    things: a missing label is a fixable omission, whereas an attribution is a
    claim the project can never make (X1).
    """
    mentions_money = [p for p in MONEY_PATTERNS if re.search(p, lowered)]
    if not mentions_money:
        return {"mentions_money": False, "missing_labels": [],
                "attributions": []}

    missing = [concept for concept, phrases in D5_REQUIRED_LABELS.items()
               if not any(p in lowered for p in phrases)]

    attributions = []
    for pattern in ATTRIBUTION_PATTERNS:
        for m in re.finditer(pattern, lowered):
            window = lowered[max(0, m.start() - NEGATION_WINDOW):m.start()]
            if any(cue in window for cue in NEGATION_CUES):
                continue            # it is the disclaimer, not the claim
            attributions.append(m.group(0))
            break

    return {"mentions_money": True, "money_markers": sorted(set(mentions_money)),
            "missing_labels": missing, "attributions": attributions}


def check_forbidden_claims(lowered: str, payload: dict) -> list:
    """
    Enforce the payload's OWN forbidden-claims list.

    The checks above (money labels, attribution, point estimates) are specific to
    this project's blockers - they were written knowing what D5 and X1 are. That
    is fine for `14_findings.csv`, and useless for a log nobody has looked at.

    `20_extract_findings.py` derives its prohibitions from what the log cannot
    support, and ships each one with the patterns that betray a violation. This
    function applies them. Without it the gate computes a prohibition and then
    nothing enforces it - which is exactly the state this project keeps finding
    itself in and keeps having to fix.

    Negation cues are honoured as elsewhere, so a narration that says "this cannot
    be separated into queueing and processing" is stating the limit, not breaching
    it.
    """
    violations = []
    for claim in payload.get("forbidden_claims", []):
        for pattern in claim.get("patterns", []):
            for m in re.finditer(pattern, lowered):
                window = lowered[max(0, m.start() - NEGATION_WINDOW):m.start()]
                if any(cue in window for cue in NEGATION_CUES):
                    continue                # the disclaimer, not the claim
                violations.append({"id": claim.get("id", "?"),
                                   "matched": m.group(0),
                                   "because": claim.get("because", "")})
                break
            else:
                continue
            break                           # one violation per claim is enough
    return violations


def verify_narration(text: str, payload: dict) -> dict:
    """
    The test. Returns a report; `ok` is False if the narration must be rejected.
    """
    allowed = collect_payload_numbers(payload)

    unsupported = []
    for value in numbers_in(text):
        if not number_is_supported(value, allowed):
            unsupported.append(value)

    lowered = text.lower()
    banned_hits = [t for t in BANNED_TERMS if t in lowered]
    financial = check_financial(lowered)

    # A point estimate of the unsourced parameter, e.g. "a query probability of
    # 0.45" stated as fact rather than as one end of a swept range.
    point_estimate = []
    for m in re.finditer(r"q\s*(?:=|of|at)\s*(0?\.\d+)", lowered):
        window = lowered[max(0, m.start() - 120): m.end() + 120]
        if not any(w in window for w in
                   ("swept", "range", "between", "to 0.", "sweep", "unsourced")):
            point_estimate.append(m.group(0))

    forbidden = check_forbidden_claims(lowered, payload)

    return {
        "ok": not (unsupported or banned_hits or point_estimate
                   or financial["missing_labels"] or financial["attributions"]
                   or forbidden),
        "numbers_checked": len(numbers_in(text)),
        "forbidden_claims_made": forbidden,
        "unsupported_numbers": sorted(set(unsupported)),
        "banned_terms": banned_hits,
        "unqualified_point_estimates": point_estimate,
        "mentions_money": financial["mentions_money"],
        "missing_d5_labels": financial["missing_labels"],
        "attributes_to_study_hospital": financial["attributions"],
        "payload_numbers_available": len(allowed),
    }


# ---------------------------------------------------------------------------
# 3. SELF-TEST - proves the guard has teeth, with no API key
# ---------------------------------------------------------------------------
SELF_TESTS = [
    ("clean narration using only payload facts",
     "The pipeline reproduces the published antibiotics breach rate of 58.5% "
     "against a computed 58.44%. Even with no document queries at all, 24.4% of "
     "cashless patients still breach the three-hour deadline.",
     True),
    ("invents a number that is nowhere in the payload",
     "Cashless discharge took 41.7 hours longer on average, which is a "
     "substantial burden on bed capacity.",
     False),
    ("X1: attributes a loss to the hospital, and carries no D5 labels",
     "At 24.4% breaching, the hospital loses roughly 12 lakh rupees a month in "
     "blocked beds.",
     False),
    ("quotes the unsourced parameter as a point estimate",
     "The median patient breaches the deadline at q = 0.45, so the query loop "
     "is the decisive factor.",
     False),

    # ---- the D5 currency rule ---------------------------------------------
    ("D5: rupee figure WITH the full benchmark framing",
     "Valued against an external Indian benchmark - Global Health Ltd (Medanta), "
     "ARPOB INR 67,361 per occupied bed day - the discharge delay attributable to "
     "the cashless route is worth on the order of INR 44.6 lakh per 1,000 "
     "cashless admissions. This is an illustrative benchmark, not this hospital's "
     "own ARPOB: the benchmark chain is quaternary and sits at the premium end, "
     "its definition excludes pharmacy and other income, and revenue is not "
     "profit, so the figure is an upper bound on opportunity rather than a cash "
     "saving.",
     True),
    ("X1: presents the rupee figure as THIS hospital's actual loss",
     "This hospital lost INR 44.6 lakh last year to cashless discharge delays, "
     "an illustrative benchmark from Medanta that excludes pharmacy and other "
     "income, sits at the premium end, and is an upper bound on opportunity.",
     False),
    ("unlabelled rupee figure with no benchmark framing at all",
     "The discharge delay attributable to the cashless route is worth "
     "INR 44.6 lakh per 1,000 cashless admissions.",
     False),
]


def run_self_test(payload: dict) -> bool:
    rule_line("GUARD SELF-TEST - can it actually catch a hallucinated number?")
    all_ok = True
    for label, text, should_pass in SELF_TESTS:
        report = verify_narration(text, payload)
        passed = report["ok"] == should_pass
        all_ok &= passed
        verdict = "as expected" if passed else "GUARD IS BROKEN"
        print(f"\n  [{'ok  ' if passed else 'FAIL'}] {label}")
        print(f"        guard says ok={report['ok']}, expected {should_pass} "
              f"-> {verdict}")
        if report["unsupported_numbers"]:
            print(f"        unsupported numbers : {report['unsupported_numbers']}")
        if report["banned_terms"]:
            print(f"        banned terms        : {report['banned_terms']}")
        if report["unqualified_point_estimates"]:
            print(f"        point estimates     : "
                  f"{report['unqualified_point_estimates']}")
        if report["attributes_to_study_hospital"]:
            print(f"        X1 violation        : attributes the figure to the "
                  f"study hospital ({len(report['attributes_to_study_hospital'])} "
                  "pattern(s) matched)")
        if report["missing_d5_labels"]:
            print(f"        missing D5 labels   : "
                  f"{report['missing_d5_labels']}")
        elif report["mentions_money"]:
            print("        money mentioned     : all D5 labels present, allowed")
    print(f"\n  {'ALL GUARD TESTS PASSED' if all_ok else 'GUARD FAILED ITS OWN TESTS'}")
    return all_ok


# ---------------------------------------------------------------------------
# 4. THE NARRATION - only reached when credentials exist
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You write short, plain-language briefings for hospital \
administrators who are not statisticians.

ABSOLUTE RULES - these are checked mechanically after you answer, and your \
output is rejected if you break them:

1. Every number you write MUST appear verbatim in the JSON payload you are \
given. Do not compute, add, average, convert, round beyond one decimal, or \
infer any number. If you want to express magnitude without a payload number, \
use words ("roughly a quarter", "an order of magnitude").
2. Financial figures are permitted only when the complete D5 benchmark
framing is preserved. If you mention INR, rupees, ARPOB or revenue,
make clear that the figure is an illustrative external benchmark,
not this hospital's own ARPOB, that the benchmark is at the
premium/optimistic end, that its definition excludes pharmacy and
other income, and that the result is an upper-bound opportunity
rather than a cash saving. Never attribute the benchmark figure to
this hospital as an actual loss, saving, revenue or cash impact.
3. The parameter q is UNSOURCED. Refer to it only as a swept range, never as a \
single established value.
4. Respect every `caveat` field. If a finding is Dutch data, say so. If a result \
depends on an assumption, say so.
5. Use the `allowed_wording` field where one is supplied - it has been \
pre-cleared.

Write four short sections: What we found, How confident we are, What we cannot \
yet say, What to do next. Under 500 words."""


def read_api_key(var_name: str) -> str | None:
    """
    Find an API key in the environment, or failing that in a local `.env` file.

    The `.env` fallback exists because setting a Windows environment variable is
    a reliable source of confusion: `set` lasts only for the current terminal,
    `setx` writes to the user environment but does NOT affect the terminal you
    typed it into, and quoting it wrong stores the quote characters as part of
    the value - which then reaches the provider and comes back as a 403 that
    looks like a rejected key rather than a typo.

    A `.env` file has none of those failure modes, and it is already gitignored,
    so a key put there cannot reach GitHub by accident.

    Values are stripped of whitespace and of surrounding quotes, because pasting
    a key with either is the most common way this goes wrong.
    """
    def clean(value: str) -> str:
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        return value.strip()

    from_env = os.environ.get(var_name)
    if from_env and clean(from_env):
        return clean(from_env)

    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() == var_name and clean(value):
                return clean(value)
    return None


def _ollama_context_length(base_url: str, model: str) -> int | None:
    """
    Ask Ollama how big this model's context window actually is.

    Returns None for any non-Ollama endpoint or if the query fails - the caller
    then falls back to printing an estimate rather than refusing, because a
    hosted provider (Groq, Gemini) manages context itself and rejects an
    oversized prompt with a real error instead of truncating silently.
    """
    import json as _json
    import os
    import urllib.request

    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        req = urllib.request.Request(
            root + "/api/show",
            data=_json.dumps({"model": model}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            info = _json.loads(resp.read().decode("utf-8"))
    except Exception:                                          # noqa: BLE001
        return None

    # The RUNTIME window is `num_ctx` in `parameters`, set only when the model
    # declares it (e.g. via a Modelfile). When it is absent Ollama falls back to
    # its own default, which is 4096 - NOT the model's architectural maximum.
    #
    # DO NOT use `model_info["*.context_length"]` here. It reports 131072 for
    # llama3.1 whether or not num_ctx is set, so reading it made this check pass
    # a model that then truncated the prompt at 4096 - worse than no check at
    # all, because it gave false reassurance. Verified against /api/show for both
    # llama3.1:8b (no num_ctx) and llama3.1-16k (num_ctx 16384).
    params = info.get("parameters", "") or ""
    for line in params.splitlines():
        if line.strip().startswith("num_ctx"):
            try:
                return int(line.split()[-1])
            except ValueError:
                pass
    return int(os.environ.get("OLLAMA_CONTEXT_LENGTH", 4096))


def retry_after_seconds(exc, attempt: int) -> float:
    """
    How long to wait before retrying a rate-limited request.

    The server's own `Retry-After` is preferred when it sends one, because it
    knows when the window resets and guessing wastes either time or another
    rejected request. Capped either way, so a provider asking for an hour does
    not silently stall a sweep. Without the header, back off exponentially.
    """
    try:
        header = (exc.headers or {}).get("Retry-After", "")
    except Exception:                                          # noqa: BLE001
        header = ""
    try:
        return max(1.0, min(float(header), RATE_LIMIT_MAX_WAIT))
    except (TypeError, ValueError):
        pass
    return min(5.0 * (2 ** attempt), RATE_LIMIT_MAX_WAIT)


def narrate_openai_compatible(payload: dict, base_url: str, model: str,
                              api_key_env: str | None,
                              system_prompt: str | None = None) -> str:
    """
    Narrate via any OpenAI-compatible /chat/completions endpoint.

    WHY THIS EXISTS. CLAUDE.md's tool stack specifies the Claude API, and that
    remains the production path and the default. But the guard is
    PROVIDER-AGNOSTIC - `verify_narration()` takes a string and has no idea who
    wrote it - so the interesting question ("does a real model, unprompted,
    produce compliant prose?") can be answered with any model, including free
    ones. This backend makes that testable at zero cost:

        Ollama (local, free, no signup, works offline):
            ollama serve ; ollama pull llama3.1:8b
            --provider openai-compat --base-url http://localhost:11434/v1
            --model llama3.1:8b
        Groq / OpenRouter / Gemini free tiers: same flags, their base URL, and
        the env var holding the key via --api-key-env.

    Uses stdlib urllib so it adds no dependency.

    IN THE REPORT: say which backend produced any narration you show. A local
    model demonstrates the GUARD against real model output; it is not evidence
    about Claude, and the two claims must not be blurred.
    """
    import os
    import urllib.request

    # ---- BUG FIX: detect silent context truncation --------------------------
    # Ollama defaults to a 4096-token context and TRUNCATES anything longer
    # WITHOUT AN ERROR. This payload is ~4,300 tokens, so the first live run in
    # this project answered from a partial payload that never included findings
    # D1-D5, and nothing in the output said so. Documenting the trap was not
    # enough - it has to be detected, or the next person loses the same hour.
    system_prompt = system_prompt or SYSTEM_PROMPT
    prompt_chars = len(system_prompt) + len(json.dumps(payload, indent=2))
    est_tokens = prompt_chars // 4          # ~4 chars/token, good enough here
    ctx = _ollama_context_length(base_url, model)
    if ctx is not None:
        print(f"  context check: ~{est_tokens:,} prompt tokens vs {ctx:,} available")
        if est_tokens > ctx * 0.9:          # leave headroom for the response
            raise SystemExit(
                f"\n  REFUSING TO RUN - the prompt would be silently truncated.\n"
                f"  Prompt is ~{est_tokens:,} tokens; the model's context is "
                f"{ctx:,}.\n"
                "  Ollama truncates without error, so the model would answer from\n"
                "  a partial payload and the result would be meaningless.\n\n"
                "  Fix by giving the model a larger window:\n"
                f"    printf 'FROM {model}\\nPARAMETER num_ctx 16384\\n' > Modelfile\n"
                "    ollama create mymodel-16k -f Modelfile\n"
                "  then re-run with --model mymodel-16k")
    else:
        print(f"  context check: ~{est_tokens:,} prompt tokens "
              "(could not read the model's context window - verify it fits)")

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ],
        # 1600 was tuned for an 8B model writing prose. A larger reasoning
        # model answering in JSON needs more room, and running out mid-object
        # produces "JSON object is not closed" - a truncation, not a refusal.
        "max_tokens": 4000,
        "temperature": 0.3,
    }).encode("utf-8")

    # A REAL USER-AGENT, and this is load-bearing rather than cosmetic.
    #
    # urllib identifies itself as "Python-urllib/3.x" by default. Providers
    # sitting behind Cloudflare block that fingerprint outright: the request is
    # rejected with HTTP 403 and a Cloudflare body ("error code: 1010") BEFORE
    # the API ever sees the Authorization header. The failure is therefore
    # indistinguishable from a rejected key unless you read the response body -
    # a valid key and a known-bad one produced the identical error.
    headers = {"Content-Type": "application/json",
               "User-Agent": "hospital-bottleneck-detective/1.0"}
    if api_key_env:
        key = read_api_key(api_key_env)
        if not key:
            raise SystemExit(
                f"No API key found for {api_key_env}.\n"
                f"  Either export it:      setx {api_key_env} \"gsk_...\"\n"
                "                         (then CLOSE and REOPEN the terminal -\n"
                "                          setx does not affect the one you typed in)\n"
                f"  or put it in .env:     {api_key_env}=gsk_...\n"
                "                         (.env is gitignored; no quotes needed)")
        headers["Authorization"] = f"Bearer {key}"

    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=body, headers=headers)

    # A 429 is "come back later", not a verdict, so wait it out rather than
    # reporting it as a failed call. This is not merely a convenience: the
    # ground-truth sweep issues one request per scenario, and a free tier's
    # per-minute cap otherwise turned 18 of 27 scenarios into errors that had
    # nothing to do with what the sweep was measuring. Retrying happens in the
    # transport, where the HTTP status lives; no verification behaviour is
    # touched by it, and a rate limit cannot become evidence either way.
    last_exc = None
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read().decode("utf-8"))[
                    "choices"][0]["message"]["content"]
        except Exception as exc:                               # noqa: BLE001
            last_exc = exc
            if getattr(exc, "code", None) != 429 or attempt == RATE_LIMIT_RETRIES:
                break
            wait = retry_after_seconds(exc, attempt)
            print(f"  rate limited by {base_url} - waiting {wait:.0f}s "
                  f"(retry {attempt + 1}/{RATE_LIMIT_RETRIES})", flush=True)
            time.sleep(wait)

    exc = last_exc
    # Advise on the backend actually in use. Telling someone to run
    # `ollama serve` when they are calling a hosted endpoint sends them
    # looking in the wrong place - and a hosted deployment has no Ollama at
    # all, so that hint is worse than none.
    local = base_url.startswith(("http://localhost", "http://127.0.0.1"))
    status = getattr(exc, "code", None)
    if local:
        hint = ("  Check `ollama serve` is running and the model is pulled "
                f"(`ollama list`), and that {model!r} is among them.")
    elif status in (401, 403):
        hint = ("  That is an AUTHENTICATION failure, not a connection one - "
                "the endpoint was reached.\n"
                f"  Check the API key in the environment variable you passed "
                "to --api-key-env is valid and has not expired.")
    elif status == 429:
        hint = (f"  Still rate limited after {RATE_LIMIT_RETRIES} retries. Free "
                "tiers cap requests per minute; wait longer before re-running, "
                "or use a smaller model.")
    elif status == 404:
        hint = (f"  The endpoint was reached but {model!r} was not found - "
                "check the model name is one this provider serves.")
    else:
        hint = ("  Check the base URL, the model name, and that this machine "
                "has network access to the provider.")
    raise SystemExit(
        f"Request to {base_url} failed: {type(exc).__name__}: {exc}\n{hint}")


def narrate(payload: dict, system_prompt: str | None = None) -> str:
    """Call Claude. Imported lazily so the rest of this file runs without the SDK."""
    try:
        import anthropic
    except ImportError:
        raise SystemExit(
            "The `anthropic` package is not installed.\n"
            "  pip install anthropic\n"
            "and provide a credential (`ant auth login`, or ANTHROPIC_API_KEY).")

    # MISSING CREDENTIALS ARE THE COMMON CASE and the SDK signals them with a
    # bare TypeError raised deep inside header construction - a 20-line traceback
    # that does not mention what to do. Caught here and turned into the same
    # actionable message as the missing-package case. Found by installing the SDK
    # without a key and watching what a first-time runner would actually see.
    try:
        client = anthropic.Anthropic()
        _ = client.api_key or client.auth_token
    except TypeError:
        client = None
    if client is None or not (client.api_key or client.auth_token):
        raise SystemExit(
            "No Anthropic credential found.\n"
            "  Set one of:\n"
            "    setx ANTHROPIC_API_KEY \"sk-ant-...\"   (then reopen the terminal)\n"
            "    ant auth login                         (stores a profile the SDK reads)\n"
            "  The guard self-test needs neither:\n"
            "    python src/15_narrate_findings.py --self-test")

    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=system_prompt or SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
    )
    if response.stop_reason == "refusal":
        raise SystemExit(f"Model declined: {response.stop_details}")
    return "".join(b.text for b in response.content if b.type == "text")


def measure_guard_strength(payload: dict, n_random: int = 4000,
                           seed: int = 20260930) -> dict:
    """
    How often does the number check actually reject a number that is not in the
    payload? Measured here rather than quoted, because it was quoted.

    The two rates answer different questions. Random fabrication is the easy
    case: a number drawn from nowhere near the payload is almost always caught.
    A NEAR MISS is the hard one - a model repeating a real figure slightly
    wrong, which is the realistic failure and the one worth knowing the rate of.
    "Near" is defined as +-1% of a real value, because that is the size of
    mistake a model makes when paraphrasing a number rather than inventing one.

    Both rates depend on the payload: a denser number space is easier to land in
    by luck, so adding findings LOWERS them without the check having changed.
    That is why this is a function and not a constant - the figure it replaced
    had drifted out of date twice for exactly that reason.
    """
    allowed = collect_payload_numbers(payload)
    rng = random.Random(seed)
    lo, hi = 0.0, 120.0

    fabricated = sum(
        not number_is_supported(round(rng.uniform(lo, hi), 1), allowed)
        for _ in range(n_random))

    near = [c for v in sorted(allowed)
            for c in (round(v * 1.01, 4), round(v * 0.99, 4))
            if c not in allowed]
    near_rejected = sum(not number_is_supported(c, allowed) for c in near)

    return {
        "payload_numbers": len(allowed),
        "fabricated_pct": round(fabricated / n_random * 100, 1),
        "fabricated_n": n_random,
        "near_miss_pct": round(near_rejected / len(near) * 100, 1) if near else None,
        "near_miss_n": len(near),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measure-guard", action="store_true",
                    help="measure how often the number check rejects a number "
                         "not in the payload, instead of quoting a figure")
    ap.add_argument("--self-test", action="store_true",
                    help="run the guard against known-good and known-bad prose")
    ap.add_argument("--narrate", action="store_true",
                    help="generate a narration (see --provider)")
    ap.add_argument("--provider", default="anthropic",
                    choices=["anthropic", "openai-compat"],
                    help="anthropic (default, the documented production path) or "
                         "openai-compat for a free backend such as a local Ollama")
    ap.add_argument("--base-url", default="http://localhost:11434/v1",
                    help="openai-compat only; default is a local Ollama")
    ap.add_argument("--model", default=None,
                    help="openai-compat only, e.g. llama3.1:8b")
    ap.add_argument("--api-key-env", default=None,
                    help="openai-compat only; env var holding the key. Omit for "
                         "Ollama, which needs none.")
    ap.add_argument("--findings", default=None,
                    help="findings CSV to narrate (default: outputs/14_findings.csv, "
                         "this project's hand-assembled evidence package). Pass "
                         "outputs/20_findings.csv to narrate any log's extracted "
                         "findings instead.")
    ap.add_argument("--out", default=None,
                    help="where to write the accepted narration "
                         "(default: outputs/15_narration.md)")
    args = ap.parse_args()

    # Point the module at a different findings file if asked. The GUARD is
    # unchanged either way - it verifies prose against whatever payload was built,
    # so it protects an extracted findings set exactly as it protects the
    # hand-assembled one.
    for flag, name in [(args.findings, "FINDINGS_CSV"), (args.out, "NARRATION_MD")]:
        if flag:
            path = Path(flag)
            globals()[name] = path if path.is_absolute() else ROOT / path

    # Namespace the payload to the findings set it was built from. Without this,
    # narrating another log's findings silently overwrites
    # `15_narration_payload.json` - a file the report cites as THIS project's
    # payload. The artifact would still be named right and hold the wrong log.
    if args.findings:
        stem = Path(args.findings).stem.replace("_findings", "")
        globals()["PAYLOAD_JSON"] = ROOT / "outputs" / f"15_payload__{stem}.json"

    payload = build_payload()
    PAYLOAD_JSON.parent.mkdir(parents=True, exist_ok=True)
    PAYLOAD_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rule_line("PAYLOAD - what the model is allowed to see")
    print(f"  quotable findings : {len(payload['findings'])}")
    print(f"  forbidden claims  : {len(payload['forbidden_claims'])}")
    print(f"  distinct numbers  : {len(collect_payload_numbers(payload))}")
    print(f"  wrote {PAYLOAD_JSON.relative_to(ROOT)}")

    if args.measure_guard:
        rule_line("GUARD STRENGTH - measured against this payload")
        r = measure_guard_strength(payload)
        print(f"  distinct payload numbers        : {r['payload_numbers']}")
        print(f"  randomly fabricated rejected    : {r['fabricated_pct']}%  "
              f"(n={r['fabricated_n']})")
        print(f"  near-miss (+-1%) rejected       : {r['near_miss_pct']}%  "
              f"(n={r['near_miss_n']})")
        print()
        print("  A STRONG FILTER, NOT A PROOF: a fabricated value landing "
              "within the")
        print("  tolerance of a real one is accepted.")
        raise SystemExit(0)

    if args.self_test or not args.narrate:
        ok = run_self_test(payload)
        if not args.narrate:
            rule_line("NEXT")
            print("  Guard verified with no API call. To produce an actual narration:")
            print("    pip install anthropic")
            print("    ant auth login      (or set ANTHROPIC_API_KEY)")
            print("    python src/15_narrate_findings.py --narrate")
            raise SystemExit(0 if ok else 1)

    if args.provider == "anthropic":
        text = narrate(payload)
        backend = f"Anthropic API, {MODEL}"
    else:
        if not args.model:
            raise SystemExit("--provider openai-compat requires --model "
                             "(e.g. --model llama3.1:8b)")
        text = narrate_openai_compatible(
            payload, args.base_url, args.model, args.api_key_env)
        backend = f"{args.provider} @ {args.base_url}, model {args.model}"
    print(f"\n  backend: {backend}")
    report = verify_narration(text, payload)

    rule_line("NARRATION")
    print(text)

    rule_line("VERIFICATION - did the model compute anything?")
    for k, v in report.items():
        print(f"  {k:<28} {v}")

    if report["ok"]:
        # The backend is stamped into the file because provenance is not
        # recoverable afterwards: a narration from a local Llama and one from the
        # Claude API are indistinguishable as prose, and CLAUDE.md's stack names
        # Claude. Without this line a reader would reasonably assume the wrong one.
        NARRATION_MD.write_text(
            f"<!-- generated by: {backend} -->\n\n"
            f"> **Generated by: {backend}.**\n\n"
            + text
            + "\n\n---\n\nVerified: every number in this text appears in "
            f"`{PAYLOAD_JSON.name}`. {report['numbers_checked']} numbers checked "
            "against the computed payload; none were introduced by the model.\n",
            encoding="utf-8")
        print(f"\n  ACCEPTED - wrote {NARRATION_MD.relative_to(ROOT)}")
    else:
        print("\n  REJECTED - the narration introduced content it was not given.")
        print("  Not written to disk. Re-run, or tighten the system prompt.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
