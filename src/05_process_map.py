"""
STEP 2d - The process map. Draw where the time goes.

A process map is the DIRECTLY-FOLLOWS GRAPH: nodes are activities, an edge means
"this activity was followed by that one". Each edge can be annotated with either
of two things, and they answer different questions:

  FREQUENCY   - how many patients took this path.     "What usually happens."
  PERFORMANCE - how long the clock ran on it.         "Where time is lost."

Same graph, different labels, often a very different-looking answer - because the
biggest delay is frequently on an edge that few patients take. That gap between
the two pictures is the argument for process mining in one image.

THREE THINGS THAT DECIDE WHETHER THIS MAP IS ACCURATE

  1. THE MAP IS ONLY AS GOOD AS THE LOG YOU FEED IT.
     Hand pm4py the raw log and the 47-day discharge-to-readmission edge walks
     straight back in and dominates the picture, exactly as it dominated the
     Step 2a ranking. So these maps are built on the EPISODE-CUT log from Step
     2b, with each episode of care treated as its own case.

  2. STRUCTURE COMES FROM FREQUENCY, COLOUR COMES FROM TIME.
     Filtering a PERFORMANCE graph by "keep the top 20%" keeps the SLOWEST
     edges, which are the rare ones - a map of anecdotes. So we filter the
     FREQUENCY graph to decide which edges to draw, then annotate the survivors
     with their median duration.

  3. FILTERING MUST BE DECLARED, NOT HIDDEN.
     This log has 846 variants across 1,050 patients, so the unfiltered map is
     spaghetti. Mannhardt & Blinde say exactly this about their own discovered
     model: it is "difficult to be used for communication with doctors and
     nurses". Filtering is therefore required, not optional - but presenting a
     filtered map without saying so is misleading. The full spaghetti is rendered
     too, and every filtered map states its coverage in the title.

Outputs (PNG, regenerated - they are gitignored):
  05_map_spaghetti.png              the unfiltered reality, for contrast
  05_map_frequency.png              what usually happens
  05_map_performance.png            where the time goes  <- the bottleneck map
  05_map_departments.png            the same, by DEPARTMENT code
  05_map_roles.png                  departments collapsed to roles <- the readable one
  05_powerbi_contrast.png           the bar chart vs the path-dependent finding
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # render to file, never try to open a window
import matplotlib.pyplot as plt
import pandas as pd
import pm4py

CASE_ID = "case:concept:name"
ACTIVITY = "concept:name"
TIMESTAMP = "time:timestamp"
GROUP = "org:group"

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "data" / "raw" / "Sepsis Cases - Event Log.xes.gz"
OUT = ROOT / "outputs"
OUT_PREFIX = "05"

# Fallbacks ONLY. The episode rule and the role map are now discovered from the
# incoming log by src/18_process_profile.py; these names are what we drop back to
# if profiling fails, and they are Sepsis-specific by construction.
FALLBACK_RELEASE_PREFIX = "Release"
FALLBACK_RETURN_ACTIVITY = "Return ER"
RESOURCE_CANDIDATES = ["org:group", "org:resource", "resource", "department"]

# Declared up front and printed on every map that uses it.
PATH_COVERAGE = 0.25    # keep the paths covering the busiest 25% of the graph
MIN_FIRES = 20          # a department pair must fire this often to be drawn


def rule_line(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def ensure_graphviz() -> None:
    """
    pm4py renders through Graphviz's `dot` executable, which must be on PATH.

    On Windows conda installs it to <env>/Library/bin, and that directory is NOT
    on PATH unless the environment was activated in this shell. Running the
    interpreter by its full path - which is exactly what an IDE or a scheduled
    task does - therefore fails with a confusing Graphviz error. We locate it
    next to the running interpreter and put it on PATH ourselves.
    """
    env_root = Path(sys.executable).parent
    for candidate in [env_root / "Library" / "bin", env_root / "bin", env_root]:
        if (candidate / "dot.exe").exists() or (candidate / "dot").exists():
            os.environ["PATH"] = str(candidate) + os.pathsep + os.environ["PATH"]
            print(f"Graphviz found at: {candidate}")
            return
    print(
        "WARNING: could not find `dot`. Map rendering will fail.\n"
        "  Fix with:  conda install -n hospital -c conda-forge graphviz python-graphviz"
    )


def load_profiler():
    """Import the profiler by path (module names cannot start with a digit)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "profiler", ROOT / "src" / "18_process_profile.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_episode_log() -> pd.DataFrame:
    """
    Load the log and apply the episode cut, then make each episode its own case
    so that pm4py never draws an edge across a discharge.

    The cut used to hard-code `Release*` / `Return ER`. It now ASKS the profiler
    which activities end and restart an episode, so a log from another hospital -
    whose discharge step is called something else entirely, or which has no
    readmission at all - gets the right treatment without being edited.
    """
    global GROUP

    if not LOG_PATH.exists():
        raise SystemExit(
            f"Event log not found at {LOG_PATH}\nRun:  python src/00_download_data.py"
        )

    print(f"Loading {LOG_PATH.name} ...")
    log = pm4py.read_xes(str(LOG_PATH))

    if GROUP not in log.columns:
        found = next((c for c in RESOURCE_CANDIDATES if c in log.columns), None)
        if found is None:
            raise SystemExit(
                f"No department/resource column found. Looked for: "
                f"{RESOURCE_CANDIDATES}\n  Columns present: {list(log.columns)[:12]}")
        print(f"  department column auto-detected: '{found}' (not '{GROUP}')")
        GROUP = found

    ev = log[[CASE_ID, ACTIVITY, TIMESTAMP, GROUP]].copy()
    ev[TIMESTAMP] = pd.to_datetime(ev[TIMESTAMP], utc=True, format="mixed")
    ev = ev.sort_values([CASE_ID, TIMESTAMP], kind="stable").reset_index(drop=True)

    try:
        ep = load_profiler().detect_episode_boundaries(
            ev, CASE_ID, ACTIVITY, TIMESTAMP)
        ends, restarts = ep["episode_end_activities"], ep["episode_restart_activities"]
        print(f"  episode ENDS discovered   : {ends or 'none'}")
        print(f"  episode RESTARTS discovered: {restarts or 'none'}")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  WARNING: profiling failed ({exc}); falling back to Sepsis names.")
        ends = [a for a in ev[ACTIVITY].unique()
                if str(a).startswith(FALLBACK_RELEASE_PREFIX)]
        restarts = [FALLBACK_RETURN_ACTIVITY]

    is_release = ev[ACTIVITY].isin(ends)
    is_return = ev[ACTIVITY].isin(restarts)
    prev_was_release = is_release.groupby(ev[CASE_ID]).shift(1).fillna(False)
    opens = is_return | prev_was_release
    episode = opens.groupby(ev[CASE_ID]).cumsum()

    # A new case identifier per episode. pm4py now cannot connect the last event
    # of one episode to the first of the next, which is precisely what we want:
    # the patient was at home in between.
    ev["episode_case"] = ev[CASE_ID].astype(str) + "#" + episode.astype(str)

    n_cases = ev[CASE_ID].nunique()
    n_eps = ev["episode_case"].nunique()
    print(f"{n_cases:,} patients -> {n_eps:,} episodes of care "
          f"({n_eps - n_cases:,} readmissions split off)")
    return ev


def as_pm4py_log(ev: pd.DataFrame, activity_col: str) -> pd.DataFrame:
    """
    Hand pm4py a frame using `episode_case` as the case and `activity_col` as the
    activity. Passing org:group as the activity is what produces the department
    map - the graph machinery does not care what the nodes are called.
    """
    out = ev[["episode_case", activity_col, TIMESTAMP]].copy()
    out.columns = [CASE_ID, ACTIVITY, TIMESTAMP]
    return pm4py.format_dataframe(
        out, case_id=CASE_ID, activity_key=ACTIVITY, timestamp_key=TIMESTAMP
    )


def prune_endpoints(dfg: dict, sa: dict, ea: dict) -> tuple[dict, dict]:
    """
    Keep only start/end activities that still appear in the (filtered) edge set.

    Needed because filtering removes edges but leaves the start/end dictionaries
    untouched. `Return ER` starts 294 episodes but its only outgoing edge fires 3
    times, so the filter drops that edge while `Return ER` stays listed as a start
    activity - and the renderer then looks up a node it never drew.
    """
    nodes = {a for edge in dfg for a in edge}
    return ({a: n for a, n in sa.items() if a in nodes},
            {a: n for a, n in ea.items() if a in nodes})


def draw_maps(ev: pd.DataFrame) -> None:
    rule_line("DRAWING THE ACTIVITY MAPS")
    log = as_pm4py_log(ev, ACTIVITY)

    # --- the structure: who follows whom, and how often ----------------------
    freq_dfg, sa, ea = pm4py.discover_dfg(log)
    print(f"  edges in the unfiltered graph : {len(freq_dfg):,}")

    # 1. The spaghetti, rendered so the filtered maps below can state
    #    what they are hiding.
    pm4py.save_vis_dfg(
        freq_dfg, sa, ea, str(OUT / f"{OUT_PREFIX}_map_spaghetti.png"),
        graph_title=f"UNFILTERED - all {len(freq_dfg)} paths. This is why filtering "
                    "is required, not optional.",
    )
    print(f"  wrote {OUT_PREFIX}_map_spaghetti.png")

    # 2. Filter on FREQUENCY to choose which edges are worth drawing.
    f_dfg, f_sa, f_ea = pm4py.filter_dfg_paths_percentage(
        freq_dfg, sa, ea, percentage=PATH_COVERAGE
    )
    f_sa, f_ea = prune_endpoints(f_dfg, f_sa, f_ea)
    kept = len(f_dfg) / len(freq_dfg) * 100
    print(f"  edges kept at {PATH_COVERAGE:.0%} coverage    : {len(f_dfg):,} "
          f"({kept:.0f}% of edges)")

    pm4py.save_vis_dfg(
        f_dfg, f_sa, f_ea, str(OUT / f"{OUT_PREFIX}_map_frequency.png"),
        graph_title=f"WHAT USUALLY HAPPENS - main paths only "
                    f"({len(f_dfg)} of {len(freq_dfg)} edges shown)",
    )
    print(f"  wrote {OUT_PREFIX}_map_frequency.png")

    # 3. The bottleneck map: same edges, annotated with MEDIAN duration.
    #    Median for the reason established in Step 2a - the long right tail.
    perf_dfg, p_sa, p_ea = pm4py.discover_performance_dfg(log)
    perf_kept = {edge: stats for edge, stats in perf_dfg.items() if edge in f_dfg}

    pm4py.save_vis_performance_dfg(
        perf_kept, f_sa, f_ea, str(OUT / f"{OUT_PREFIX}_map_performance.png"),
        aggregation_measure="median",
        graph_title=f"WHERE THE TIME GOES - median wait per path "
                    f"({len(perf_kept)} of {len(freq_dfg)} edges shown)",
    )
    print(f"  wrote {OUT_PREFIX}_map_performance.png")
    print(
        "\n  Compare 05_map_frequency.png with 05_map_performance.png. Identical\n"
        "  structure, different story: the busiest arrow and the slowest arrow are\n"
        "  not the same arrow. A frequency-only view sends you to fix the wrong one."
    )


def draw_department_map(ev: pd.DataFrame) -> None:
    """The Step 2b handoff finding, as a picture."""
    rule_line("DRAWING THE DEPARTMENT MAP (the handoff view)")
    log = as_pm4py_log(ev, GROUP)

    freq_dfg, sa, ea = pm4py.discover_dfg(log)
    # Here we filter by an absolute count rather than a percentage: 20 of the 26
    # department codes are individual wards with tiny volumes, and a percentage
    # filter would drop the lab-to-discharge handoff we most want to see.
    busy = {edge: n for edge, n in freq_dfg.items() if n >= MIN_FIRES}
    sa, ea = prune_endpoints(busy, sa, ea)
    print(f"  department pairs firing >= {MIN_FIRES}x : {len(busy):,} of {len(freq_dfg):,}")

    perf_dfg, _, _ = pm4py.discover_performance_dfg(log)
    perf_kept = {edge: stats for edge, stats in perf_dfg.items() if edge in busy}

    pm4py.save_vis_performance_dfg(
        perf_kept, sa, ea, str(OUT / f"{OUT_PREFIX}_map_departments.png"),
        aggregation_measure="median",
        graph_title="WHERE THE BATON IS DROPPED - median wait between departments "
                    f"(pairs firing at least {MIN_FIRES} times). "
                    "B=Laboratory, A=ER team, C=Triage, E=Discharge desk, rest=wards",
    )
    print(f"  wrote {OUT_PREFIX}_map_departments.png")
    print(
        "\n  This is the Step 2b table as a picture: look for the arrow from the\n"
        "  Laboratory (B) to the Discharge desk (E). That single handoff carries a\n"
        "  32-hour median across 637 discharges."
    )


def draw_role_map(ev: pd.DataFrame) -> None:
    """
    The department map is technically correct but unreadable: 20 of the 26 group
    codes are individual wards, and Step 2b already established they all behave
    the same way (each waits 18-35 h on the laboratory). Drawing them separately
    adds 20 nodes of noise and hides the story.

    So collapse each code to its ROLE. This used to be a hand-typed dictionary
    ({"B": "Laboratory", "A": "ER team", ...}) decoded once by cross-tabulating
    group against activity - correct for Sepsis, and worthless anywhere else,
    because another hospital's departments are not called A, B and C.

    It is now DERIVED: the profiler gives every department a behavioural
    signature (the activities covering 60% of its work) and merges departments
    whose signatures match. On Sepsis that independently rediscovers the typed
    map - the 18 wards collapse to one node on their own - and improves on it by
    separating the IC wards, which the typed version wrongly lumped in.

    The trade is accuracy for elegance: roles come out named "Leucocytes / CRP"
    rather than "Laboratory", because that is what the log can actually support.

    This is the paper's own complaint made constructive: a model that is correct
    but "difficult to be used for communication with doctors and nurses" has
    failed at its job. Aggregating to the level the audience thinks in is part of
    the analysis, not a cosmetic afterthought.
    """
    rule_line("DRAWING THE ROLE MAP (departments collapsed to what they DO)")

    ev = ev.copy()
    try:
        clusters = load_profiler().cluster_departments_by_behaviour(
            ev, ACTIVITY, GROUP)
    except Exception as exc:                                   # noqa: BLE001
        print(f"  WARNING: role clustering failed ({exc}); drawing raw departments.")
        clusters = {"available": False}

    if clusters.get("available"):
        mapping = clusters["mapping"]
        ev["role"] = ev[GROUP].astype(str).map(mapping).fillna(ev[GROUP].astype(str))
        print(f"  {clusters['n_departments']} departments -> "
              f"{clusters['n_roles']} behavioural roles")
    else:
        ev["role"] = ev[GROUP].astype(str)

    print("  " + ", ".join(f"{r}={n:,}" for r, n in
                           ev["role"].value_counts().items()))

    log = as_pm4py_log(ev, "role")
    freq_dfg, sa, ea = pm4py.discover_dfg(log)
    busy = {edge: n for edge, n in freq_dfg.items() if n >= MIN_FIRES}
    sa, ea = prune_endpoints(busy, sa, ea)

    perf_dfg, _, _ = pm4py.discover_performance_dfg(log)
    perf_kept = {edge: stats for edge, stats in perf_dfg.items() if edge in busy}

    pm4py.save_vis_performance_dfg(
        perf_kept, sa, ea, str(OUT / f"{OUT_PREFIX}_map_roles.png"),
        aggregation_measure="median",
        graph_title="THE HANDOFF MAP - median wait between hospital functions "
                    f"(pairs firing at least {MIN_FIRES} times)",
    )
    print(f"  wrote {OUT_PREFIX}_map_roles.png")
    print(
        "\n  This is the map to put in front of an administrator. The arrow from\n"
        "  Laboratory to Discharge desk is the 32-hour median across 637 discharges\n"
        "  that Step 2b identified as the largest single handoff in the hospital."
    )


def draw_powerbi_contrast(ev: pd.DataFrame) -> None:
    """
    The project's standing design rule, as one figure: a bar chart of average wait
    per activity next to the process-mining finding that the delay is
    path-dependent.

    Left panel  - what a dashboard shows you.
    Right panel - the same activity, split by which step the patient arrived from.
    """
    rule_line("DRAWING THE POWER BI CONTRAST")

    grouped = ev.groupby("episode_case", sort=False)
    ev = ev.copy()
    ev["next_activity"] = grouped[ACTIVITY].shift(-1)
    delta = grouped[TIMESTAMP].shift(-1) - ev[TIMESTAMP]
    ev["gap_h"] = delta.dt.total_seconds() / 3600.0
    gaps = ev.dropna(subset=["next_activity"])

    per_activity = gaps.groupby("next_activity")["gap_h"].median().sort_values()

    # Split the activity whose incoming paths disagree the most - the one the
    # dashboard's single average distorts worst.
    best, best_split, best_spread = None, None, 0.0
    for activity in per_activity.index:
        split = gaps[gaps["next_activity"] == activity].groupby(ACTIVITY)["gap_h"].agg(
            fires="size", median_h="median")
        split = split[split["fires"] >= MIN_FIRES].sort_values("median_h")
        if len(split) < 2:
            continue
        spread = split["median_h"].iloc[-1] - split["median_h"].iloc[0]
        if spread > best_spread:
            best, best_split, best_spread = activity, split, spread

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    ax1.barh(per_activity.index, per_activity.values, color="#7f8c8d")
    ax1.set_xlabel("median wait before activity (hours)")
    ax1.set_title("WHAT A DASHBOARD SHOWS YOU\none number per activity",
                  fontsize=11, fontweight="bold")
    ax1.axvline(0, color="black", linewidth=0.8)

    colours = ["#c0392b" if v == best_split["median_h"].max() else "#2980b9"
               for v in best_split["median_h"]]
    ax2.barh(best_split.index, best_split["median_h"], color=colours)
    dashboard_value = per_activity[best]
    ax2.axvline(dashboard_value, color="#7f8c8d", linestyle="--", linewidth=2,
                label=f"the dashboard's single number for\n'{best}': "
                      f"{dashboard_value:.1f} h")
    ax2.set_xlabel("median wait (hours)")
    ax2.set_title(f"WHAT PROCESS MINING SHOWS YOU\n'{best}' split by the step "
                  "the patient arrived from", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "Why not just build a Power BI dashboard?  The delay is path-dependent, "
        "so one average per activity hides it.",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / f"{OUT_PREFIX}_powerbi_contrast.png", dpi=150)
    plt.close(fig)

    print(f"  wrote {OUT_PREFIX}_powerbi_contrast.png")
    print(f"  dashboard reports '{best}' as {dashboard_value:.2f} h; arriving from "
          f"'{best_split.index[-1]}'\n  it is actually "
          f"{best_split['median_h'].iloc[-1]:.1f} h.")


def configure_from_args() -> None:
    """
    Let the maps be pointed at any event log. Defaults are unchanged, so every
    documented Sepsis command keeps working exactly as before.
    """
    global LOG_PATH, GROUP, OUT_PREFIX

    parser = argparse.ArgumentParser(
        description="Render process maps for any event log.")
    parser.add_argument("--log", default=None,
                        help="path to .xes/.xes.gz/.csv (default: the Sepsis log)")
    parser.add_argument("--group-col", default=None,
                        help="department column (default: auto-detect)")
    parser.add_argument("--prefix", default=None,
                        help="output filename prefix (default: 05)")
    args = parser.parse_args()

    if args.log:
        path = Path(args.log)
        LOG_PATH = path if path.is_absolute() else ROOT / path
    if args.group_col:
        GROUP = args.group_col
    if args.prefix:
        OUT_PREFIX = args.prefix


def main() -> None:
    configure_from_args()
    ensure_graphviz()
    OUT.mkdir(parents=True, exist_ok=True)
    ev = load_episode_log()

    draw_maps(ev)
    draw_department_map(ev)
    draw_role_map(ev)
    draw_powerbi_contrast(ev)

    rule_line("STEP 2d COMPLETE - STEP 2 IS DONE")
    print("Six figures in outputs/. The maps are built on the episode-cut log, so")
    print("no arrow in them measures a patient sitting at home.")
    print(
        "\nStep 2 delivered: waiting times (2a), department handoffs (2b), a\n"
        "validation gate against published numbers (2c), and the maps (2d).\n"
        "Next: Step 3 - the SimPy Indian generator, emitting start AND complete\n"
        "timestamps so the synthetic log can separate queueing from processing -\n"
        "the one thing this real log provably cannot do."
    )


if __name__ == "__main__":
    main()
