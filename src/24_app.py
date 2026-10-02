"""
STEP 24 - THE UPLOAD APP. Bring your own event log.

HOW THIS DIFFERS FROM `16_dashboard.py`

`16` presents THIS project's findings. It reads artifacts the scripts already
wrote and never recomputes, so it cannot drift from the analysis.
It is a results poster.

`24` is a tool. You hand it a log it has never seen, it runs the pipeline, and it
shows you what that log supports - including, crucially, what it does NOT support.

THE ONE RULE THIS FILE OBEYS

    It calls the existing scripts. It does not reimplement any analysis.

Every number on screen is produced by `20_extract_findings.py` and interpreted by
`21_diagnose.py`, run as subprocesses exactly as they run from the command line.
A second implementation of the analysis would be a second source of truth, and the
copy is always the one that rots. If this app and the CLI ever disagree, that is a
bug in this file, not a difference of opinion.

WHAT AN UPLOAD TOOL NEEDS THAT THE CLI NEVER DID

  1. COLUMN MAPPING. A real hospital export has columns called `PatientID`,
     `Activity`, `Timestamp`, `Ward` - not `case:concept:name`. The app guesses,
     the user confirms, and a normalised copy is written before the pipeline sees
     it. Nothing downstream has to know this happened.

  2. TOLERANCE FOR WEAK LOGS. Plenty of exports carry only case, activity and
     timestamp. That is not an error, it is a weaker log: the analysis it supports
     still runs, and the claims it cannot support are forbidden.

WHAT IT SHOWS FIRST, AND WHY

  Not the bottleneck. The CAPABILITY GATE - what this log can and cannot answer.
  That ordering is the argument of the whole project: a finding presented without
  its limits is how a hedged result becomes a headline.

RUN IT

    conda activate hospital
    python -m streamlit run src/24_app.py
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "outputs" / "24_uploads"
PY = sys.executable

# The XES-standard names every script in this project expects.
STD = {
    "case": "case:concept:name",
    "activity": "concept:name",
    "timestamp": "time:timestamp",
    "resource": "org:group",
    "lifecycle": "lifecycle:transition",
}

# Words that hint at what a column holds, for the initial guess. The user always
# confirms - this only saves them clicks, it never decides anything.
HINTS = {
    "case": ["case", "patient", "episode", "encounter", "visit", "id", "trace"],
    "activity": ["activity", "event", "step", "task", "action", "concept:name"],
    "timestamp": ["time", "date", "stamp", "when", "datetime"],
    "resource": ["group", "dept", "department", "ward", "unit", "team",
                 "resource", "staff", "org"],
    "lifecycle": ["lifecycle", "transition", "status", "state"],
}

# Groq serves open models over an OpenAI-compatible endpoint, on a free tier with
# no card required. It matters here for one reason: Ollama runs on localhost and
# therefore does not exist on any host, so a deployed copy of this app needs a
# backend it can reach over the network.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Model availability on Groq changes, and differs per account - the llama-3.x
# names this originally listed are not served to every key. `check_api_key.py`
# prints what YOUR key can actually use; that list beats any list hardcoded here.
GROQ_MODELS = [
    "openai/gpt-oss-120b",          # ~15x the parameters of the local 8B default
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]

st.set_page_config(page_title="Process Bottleneck Detective",
                   page_icon="🏥", layout="wide")


# Session key under which a visitor's own Groq key is held on a shared host.
# Session state, not a module global: Streamlit shares globals across everyone
# connected to the same server process, so a global would hand one visitor's key
# to the next. It is never written to disk and never logged.
VISITOR_KEY = "visitor_groq_key"


def is_hosted() -> bool:
    """
    Are we running on a shared host rather than the author's machine?

    This decides two things that are fine locally and not fine in public: whose
    API key gets spent, and whether a stranger's uploaded file stays on disk.

    The detection is an ALLOWLIST of signals the known platforms set, so it
    defaults to LOCAL - an unrecognised host is treated as this machine. That is
    the permissive direction and it is a deliberate trade: closing it would
    demand a typed API key on every local run, for a risk that needs the owner
    to have put a key somewhere a deployed copy can reach. `.env` is gitignored,
    so a fresh deployment has none by default.

    Deploying anywhere other than the platforms named below, set
    BOTTLENECK_SHARED=1. It forces shared behaviour with no detection involved.
    """
    return bool(os.environ.get("BOTTLENECK_SHARED")   # explicit, any platform
                or os.environ.get("STREAMLIT_RUNTIME_ENV")
                or os.environ.get("HOSTNAME", "").startswith("streamlit")
                or os.environ.get("SPACE_ID"))          # Hugging Face Spaces


def groq_key_available() -> bool:
    """
    Is a Groq key reachable from any of the three places it can live?

    This delegates to `15_narrate_findings.read_api_key()` rather than repeating
    the lookup, and that is a bug fix rather than tidiness. The first version
    checked only the environment and Streamlit's secrets, while the analysis
    subprocess ALSO reads a local `.env` - so the app reported "no key found"
    while the very command it was about to run would have found one. Two
    resolution paths, one of them wrong: exactly the drift this project avoids
    everywhere else by having a single source of truth.

    The three places, and why each exists:
      - `.env`        easiest locally, and gitignored
      - environment   what `setx` writes
      - st.secrets    the only option on a host, where there is no shell; copied
                      into the environment because the subprocess inherits that
                      but knows nothing about Streamlit
    """
    # ON A SHARED HOST, ONLY THE VISITOR'S OWN KEY COUNTS.
    #
    # The sidebar already refuses to offer the owner's key to strangers, but it
    # refused by not looking - and this function looks, on every run, from both
    # branches. So an owner who ever pasted a key into Streamlit secrets would
    # have had it found here, exported, and spent by whoever opened the page,
    # with the sidebar still displaying "bring your own key". The sidebar's
    # comment states the rule; this enforces it.
    #
    # Nothing is inherited either: a key left in the environment by the platform
    # is cleared, so the only way to reach the model here is to type one in.
    if is_hosted():
        typed = st.session_state.get(VISITOR_KEY)
        if typed:
            os.environ["GROQ_API_KEY"] = typed
            return True
        os.environ.pop("GROQ_API_KEY", None)
        return False

    # Local: Streamlit secrets first, so the export happens before anything
    # else looks.
    try:
        key = st.secrets["GROQ_API_KEY"]
        if key:
            os.environ["GROQ_API_KEY"] = str(key)
    except Exception:                                          # noqa: BLE001
        pass

    spec = importlib.util.spec_from_file_location(
        "narrator", ROOT / "src" / "15_narrate_findings.py")
    narrator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(narrator)
    key = narrator.read_api_key("GROQ_API_KEY")
    if key:
        # The subprocess reads .env for itself, but exporting here means the
        # app and the subprocess can never disagree about whether a key exists.
        os.environ["GROQ_API_KEY"] = key
        return True
    return False


# ---------------------------------------------------------------------------
def guess_column(columns: list, role: str) -> int:
    """Index of the best-matching column for a role, or 0 if nothing matches."""
    for i, col in enumerate(columns):
        low = str(col).lower()
        if any(h in low for h in HINTS[role]):
            return i + 1                      # +1 because option 0 is "(none)"
    return 0


def run_script(args: list) -> subprocess.CompletedProcess:
    """Run a pipeline script exactly as the command line would."""
    return subprocess.run([PY, "-u", *args], cwd=ROOT, capture_output=True,
                          text=True)


def normalise(df: pd.DataFrame, mapping: dict, out_path: Path) -> tuple:
    """
    Rename the user's columns to the XES names and write a normalised copy.

    Returns (ok, message). The pipeline never learns that renaming happened,
    which is the point: one code path, whatever the upload looked like.
    """
    rename = {src: STD[role] for role, src in mapping.items() if src}
    slim = df[list(rename)].rename(columns=rename).copy()

    ts = STD["timestamp"]
    try:
        slim[ts] = pd.to_datetime(slim[ts], format="mixed", utc=True)
    except Exception as exc:                                   # noqa: BLE001
        return False, f"Could not read '{mapping['timestamp']}' as dates: {exc}"
    if slim[ts].isna().all():
        return False, f"Every value in '{mapping['timestamp']}' failed to parse."

    slim = slim.sort_values([STD["case"], ts], kind="stable")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slim.to_csv(out_path, index=False)
    return True, f"{len(slim):,} events, {slim[STD['case']].nunique():,} cases"


# ---------------------------------------------------------------------------
st.title("🏥 Process Bottleneck Detective")
st.caption(
    "Upload a process event log. Every number shown is computed by the same "
    "scripts that run from the command line - this page runs them, it does not "
    "reimplement them."
)

with st.sidebar:
    st.header("How it reads your file")
    st.markdown(
        "**One row per event.** The minimum is three columns:\n\n"
        "- **Case** — which patient/order this event belongs to\n"
        "- **Activity** — what happened\n"
        "- **Timestamp** — when\n\n"
        "A **department/resource** column unlocks handoff analysis. Without one, "
        "the analysis still runs and department claims are forbidden rather than "
        "guessed."
    )
    st.divider()
    st.header("AI interpretation")
    st.caption(
        "The model never computes anything. It receives the findings and writes "
        "reasoning around them; a guard then verifies it introduced no number of "
        "its own and made no forbidden claim."
    )
    # The local backends are hidden on a shared host, and that is a security
    # decision as much as a tidiness one. "Ollama URL" is a free-text field
    # passed to the HTTP client, so on a public deployment a visitor could point
    # it at any address and make THIS SERVER fetch it - a server-side request
    # forgery, useful for probing a cloud provider's internal metadata endpoints.
    # Ollama listens on localhost and cannot exist on a host anyway, so removing
    # the option costs nothing and closes the vector completely.
    # Claude is not offered here. The transport supports it and the
    # CLI can use it (--provider anthropic), but no credential has ever been
    # attached in this project, so the path is untested - and an option that
    # errors the moment anyone selects it is worse than an option that is absent.
    backends = ["None (analysis only)", "Groq (free tier)"]
    if not is_hosted():
        backends += ["Local (Ollama)"]
    provider = st.radio("Backend", backends, index=0)

    base_url, model = None, None
    if provider == "Local (Ollama)":
        base_url = st.text_input("Ollama URL", "http://localhost:11434/v1")
        model = st.text_input("Model", "llama3.1-16k")
        st.caption("Runs on this machine. Nothing leaves it — but Ollama does "
                   "not exist on a hosted server, so this option only works "
                   "locally.")

    elif provider == "Groq (free tier)":
        base_url = GROQ_BASE_URL
        model = st.selectbox("Model", GROQ_MODELS, index=0)
        # ON A SHARED HOST, THE VISITOR SUPPLIES THEIR OWN KEY - and it is the
        # only key groq_key_available() will look at there, so this is enforced
        # rather than merely offered. Locally, an exported GROQ_API_KEY, a
        # `.env`, or Streamlit secrets are all found automatically.
        #
        # If the owner's key sat in Streamlit secrets, every stranger who opened
        # the page would spend the owner's free quota, and one abusive visitor
        # could exhaust it for everyone. The analysis runs perfectly well with no
        # model at all, so the key is an optional extra rather than a gate.
        if is_hosted():
            typed = st.text_input(
                "Your Groq API key", type="password",
                help="Free at console.groq.com, no card. It is held for this "
                     "session only, never stored or logged.")
            if typed:
                st.session_state[VISITOR_KEY] = typed.strip()
            else:
                st.session_state.pop(VISITOR_KEY, None)
                st.info(
                    "**Bring your own key.** This is a shared deployment, so it "
                    "does not supply one - a free key from console.groq.com takes "
                    "a minute. The analysis above needs no key at all.")
        elif not groq_key_available():
            st.warning(
                "No `GROQ_API_KEY` found. Get a free key at console.groq.com "
                "(no card required), then put it in a `.env` file at the project "
                "root as `GROQ_API_KEY=gsk_...` (that file is gitignored).")
        else:
            st.caption("Key found. Note the findings are sent to Groq's "
                       "servers — fine for public or synthetic data, a "
                       "governance question for real patient logs.")



# ---- 1. upload ------------------------------------------------------------
st.subheader("1. Upload an event log")
upload = st.file_uploader("CSV, or an XES/XES.GZ process log",
                          type=["csv", "xes", "gz"])

if upload is None:
    st.info("Waiting for a file. A log with ~100+ cases gives the most to work "
            "with; anything smaller still runs but says less.")
    st.stop()

# Uploaded files are written to a TEMPORARY directory when hosted, so a
# stranger's event log is not left sitting on a shared server after they close
# the tab. Locally they stay under outputs/ where they are useful to re-run.
# The pipeline runs as a subprocess and needs a real path, so the file has to
# touch disk somewhere - the choice is only about how long it survives.
if is_hosted():
    UPLOADS = Path(tempfile.mkdtemp(prefix="bottleneck_"))
UPLOADS.mkdir(parents=True, exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")

# NEVER build a path from a client-supplied filename.
#
# A file called "../../evil.csv" would otherwise write outside the upload
# directory - on a public deployment that is an attacker overwriting files in
# the app. The uploaded name is used for DISPLAY only; the path is constructed
# entirely from a timestamp and an extension we recognise ourselves.
suffix = ".xes.gz" if upload.name.lower().endswith(".xes.gz") else (
    ".xes" if upload.name.lower().endswith(".xes") else ".csv")
safe_name = f"{stamp}_upload{suffix}"
raw_path = UPLOADS / safe_name
# Belt and braces: confirm the resolved path really is inside UPLOADS before
# writing a single byte. Cheap, and it catches anything the logic above missed.
if UPLOADS.resolve() not in raw_path.resolve().parents:
    st.error("Refused to write outside the upload directory.")
    st.stop()
raw_path.write_bytes(upload.getbuffer())

# The display name is escaped: a filename is attacker-controlled text and
# Streamlit renders markdown, so backticks or brackets in it would otherwise
# alter the page.
shown_name = re.sub(r"[^A-Za-z0-9._ -]", "", upload.name)[:60] or "uploaded file"

if upload.name.lower().endswith((".xes", ".gz")):
    # Already in the standard schema - no mapping needed or possible.
    st.success(f"Loaded `{shown_name}` — XES already uses the standard column "
               "names, so no mapping is needed.")
    log_for_pipeline = raw_path
else:
    df = pd.read_csv(raw_path)
    st.success(f"Loaded `{shown_name}` — {len(df):,} rows, "
               f"{len(df.columns)} columns")
    with st.expander("Preview the first rows"):
        st.dataframe(df.head(8), width="stretch")

    # ---- 2. column mapping ------------------------------------------------
    st.subheader("2. Tell it which column is which")
    st.caption("Guesses are from the column names. Correct anything that is wrong "
               "— everything downstream depends on these three.")

    options = ["(none)"] + list(df.columns)
    c1, c2, c3 = st.columns(3)
    mapping = {}
    with c1:
        mapping["case"] = st.selectbox("Case ID *", options,
                                       index=guess_column(list(df.columns), "case"))
    with c2:
        mapping["activity"] = st.selectbox(
            "Activity *", options, index=guess_column(list(df.columns), "activity"))
    with c3:
        mapping["timestamp"] = st.selectbox(
            "Timestamp *", options,
            index=guess_column(list(df.columns), "timestamp"))
    c4, c5 = st.columns(2)
    with c4:
        mapping["resource"] = st.selectbox(
            "Department / resource (optional)", options,
            index=guess_column(list(df.columns), "resource"))
    with c5:
        mapping["lifecycle"] = st.selectbox(
            "Lifecycle (optional)", options,
            index=guess_column(list(df.columns), "lifecycle"))

    mapping = {k: (v if v != "(none)" else None) for k, v in mapping.items()}
    if not all(mapping[k] for k in ("case", "activity", "timestamp")):
        st.warning("Case, Activity and Timestamp are all required.")
        st.stop()

    log_for_pipeline = UPLOADS / f"{stamp}_normalised.csv"
    ok, msg = normalise(df, mapping, log_for_pipeline)
    if not ok:
        st.error(msg)
        st.stop()
    st.caption(f"Normalised: {msg}")

    if not mapping["resource"]:
        st.info("**No department column mapped.** The analysis will run on "
                "activities only, and any claim about which team is responsible "
                "will be forbidden rather than guessed.")

# ---- 3. run ---------------------------------------------------------------
st.subheader("3. Analyse")
if not st.button("Run the pipeline", type="primary"):
    st.stop()

prefix = f"24_{stamp}"
with st.status("Running…", expanded=True) as status:
    st.write("Profiling the log and extracting findings…")
    proc = run_script(["src/20_extract_findings.py", "--log",
                       str(log_for_pipeline), "--prefix", prefix])
    if proc.returncode != 0:
        status.update(label="Analysis failed", state="error")
        st.error("The extractor could not read this log.")
        st.code((proc.stderr or proc.stdout)[-1500:])
        st.stop()
    status.update(label="Analysis complete", state="complete")

findings_path = ROOT / "outputs" / f"{prefix}_findings.json"
result = json.loads(findings_path.read_text(encoding="utf-8"))
caps = result["capabilities"]
quotable = [f for f in result["findings"] if not f["category"].startswith("5")]

# ---- 4. the capability gate, FIRST ---------------------------------------
st.divider()
st.subheader("What this log can and cannot answer")
st.caption("Shown before the findings on purpose. A result presented without its "
           "limits is how a hedged number becomes a headline.")

a, b, c, d = st.columns(4)
size = result["generated_from"]
a.metric("Events", f"{size['events']:,}")
b.metric("Cases", f"{size['cases']:,}")
c.metric("Findings available", len(quotable))
d.metric("Claims forbidden", len(result["forbidden_claims"]),
         help="Derived from what this log cannot support — not a fixed list")

left, right = st.columns(2)
with left:
    st.markdown("**Supported**")
    st.markdown(
        f"- Lifecycle values: `{caps['lifecycle_values'] or 'none'}`\n"
        f"- Separate waiting from processing: "
        f"**{'yes' if caps['can_separate_wait_from_work'] else 'no'}**\n"
        f"- Department column: `{caps['resource_column'] or 'none found'}`"
        + (f"\n- Department granularity: **{caps['resource_granularity']}** "
           f"({caps['same_resource_transition_share']*100:.2f}% of consecutive "
           f"events share an actor)" if caps.get("resource_granularity") else "")
        + (f"\n- Column completeness: "
           f"**{(1-caps['resource_missing_share'])*100:.1f}%** filled"
           if caps.get("resource_missing_share") is not None else "")
    )
with right:
    st.markdown("**Forbidden on this log**")
    if result["forbidden_claims"]:
        for blocked in result["forbidden_claims"]:
            st.markdown(f"- **{blocked['id']}** — must not claim "
                        f"{blocked['must_not_claim']}")
    else:
        st.markdown("_None._")

# ---- 5. findings ----------------------------------------------------------
st.divider()
st.subheader("Findings")
for f in quotable:
    with st.container(border=True):
        st.markdown(f"**{f['id']}** · {f['category']}")
        st.markdown(f["finding"])
        if f["caveat"]:
            st.caption(f"Caveat — {f['caveat']}")
        st.caption(f"Computed by `{f['evidence_source']}`")

with st.expander("Download"):
    st.download_button("Findings (JSON)", findings_path.read_bytes(),
                       file_name=f"{prefix}_findings.json", mime="application/json")
    csv_path = ROOT / "outputs" / f"{prefix}_findings.csv"
    if csv_path.exists():
        st.download_button("Findings (CSV)", csv_path.read_bytes(),
                           file_name=f"{prefix}_findings.csv", mime="text/csv")

# ---- 6. AI interpretation -------------------------------------------------
st.divider()
st.subheader("Interpretation and recommendations")

if provider == "None (analysis only)":
    st.info("No model selected. The findings above are complete on their own — "
            "the model adds interpretation, never numbers. Pick a backend in the "
            "sidebar to generate it.")
    st.stop()

args = ["src/21_diagnose.py", "--findings", str(findings_path), "--diagnose"]
if provider == "Local (Ollama)":
    args += ["--provider", "openai-compat", "--base-url", base_url,
             "--model", model]
elif provider == "Groq (free tier)":
    if not groq_key_available():
        st.error("No `GROQ_API_KEY` available — set it before running.")
        st.stop()
    # --api-key-env names the variable to read; the key itself is never passed
    # on the command line, where it would show up in process listings.
    args += ["--provider", "openai-compat", "--base-url", base_url,
             "--model", model, "--api-key-env", "GROQ_API_KEY"]
else:
    # unreachable: "None (analysis only)" returns before this point
    st.error("No backend selected.")
    st.stop()

with st.status("Asking the model, then verifying every claim…", expanded=True) as s:
    diag_proc = run_script(args)
    s.update(label="Verification complete",
             state="complete" if diag_proc.returncode == 0 else "error")

diag_path = ROOT / "outputs" / f"{prefix}_findings_diagnosis.json"
rejected = ROOT / "outputs" / f"{prefix}_findings_diagnosis_REJECTED.json"

# A FAILURE TO REACH THE MODEL IS NOT A VERIFICATION FAILURE, and showing one as
# the other is actively misleading. Selecting Ollama while it happens not to be
# running produced "Rejected - nothing was written: the model produced an answer
# that failed verification" - when no answer was ever produced.
#
# The REJECTED artifact is written only by the guard, so its presence is what
# distinguishes the two. Without it, the subprocess failed before any output
# existed to verify.
if diag_proc.returncode != 0 and not rejected.exists():
    st.error("**Could not reach the model.** Nothing was analysed by the AI layer.")
    st.markdown(
        "This is a connection or configuration problem, not a verification "
        "failure, the guard was never involved. The findings above are "
        "unaffected: they were computed without a model."
    )
    detail = (diag_proc.stdout or "") + (diag_proc.stderr or "")
    if "ollama" in detail.lower() or "11434" in detail:
        st.markdown(
            "Ollama does not appear to be running. Start it with `ollama serve`, "
            "check the model is pulled (`ollama list`), or switch the backend to "
            "**Groq** in the sidebar."
        )
    with st.expander("What the pipeline reported"):
        st.code(detail[-2000:] or "(no output)")
    st.stop()

if diag_proc.returncode != 0 or not diag_path.exists():
    st.error("**Rejected — nothing was written.**")
    st.markdown(
        "The model produced an answer that failed verification, so it was "
        "discarded. An unverified diagnosis is worse than none, because it reads "
        "exactly like a verified one."
    )
    if rejected.exists():
        report = json.loads(rejected.read_text(encoding="utf-8"))["verification"]
        for key in ("unsupported_numbers", "uncited_claims", "unknown_evidence_ids",
                    "untestable_hypotheses", "missing_sections"):
            if report.get(key):
                st.markdown(f"- **{key.replace('_', ' ')}**: `{report[key]}`")
        for v in report.get("forbidden_claims_made", []):
            st.markdown(f"- **forbidden claim {v['id']}** — matched "
                        f"“{v['matched']}”. {v['because']}")
    with st.expander("Raw output"):
        st.code(diag_proc.stdout[-2500:])
    st.stop()

diagnosis = json.loads(diag_path.read_text(encoding="utf-8"))
by_id = {f["id"]: f["finding"] for f in result["findings"]}
st.success("**Accepted.** Every number traces to a finding above, every claim "
           "cites one, and no forbidden claim was made.")

bottleneck = diagnosis.get("bottleneck", {})
st.markdown("#### Where the time goes")
st.markdown(bottleneck.get("summary", ""))
st.caption("Evidence: " + ", ".join(f"`{i}`" for i in
                                    bottleneck.get("evidence_ids", [])))

st.markdown("#### Probable drivers")
st.caption("Hypotheses, not conclusions — an event log records when things "
           "happened, not why. Each carries a way to test it.")
for drv in diagnosis.get("probable_drivers", []):
    with st.container(border=True):
        st.markdown(f"**{drv.get('hypothesis', '')}**  \n"
                    f"Confidence: {drv.get('confidence', '?')}")
        st.markdown(f"*How to test:* {drv.get('how_to_test', '')}")
        st.caption("Evidence: " + ", ".join(f"`{i}`" for i in
                                            drv.get("evidence_ids", [])))

st.markdown("#### Recommendations")
for rec in diagnosis.get("recommendations", []):
    with st.container(border=True):
        st.markdown(f"**{rec.get('action', '')}**  \n"
                    f"Effort: {rec.get('effort', '?')}")
        st.markdown(f"*Why this follows:* {rec.get('why_relevant', '')}")
        st.caption("Evidence: " + ", ".join(f"`{i}`" for i in
                                            rec.get("evidence_ids", [])))

with st.expander("The findings these cite"):
    cited = {i for k in ("bottleneck",) for i in
             (diagnosis.get(k) or {}).get("evidence_ids", [])}
    for key in ("probable_drivers", "recommendations"):
        for item in diagnosis.get(key, []):
            cited.update(item.get("evidence_ids", []))
    for fid in sorted(cited):
        st.markdown(f"- `{fid}` — {by_id.get(fid, '(unknown)')}")

md_path = ROOT / "outputs" / f"{prefix}_findings_diagnosis.md"
if md_path.exists():
    st.download_button("Download the diagnosis (Markdown)", md_path.read_bytes(),
                       file_name=md_path.name, mime="text/markdown")
