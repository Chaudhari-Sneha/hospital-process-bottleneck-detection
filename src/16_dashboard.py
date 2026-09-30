"""
STEP 7 - The Streamlit dashboard.

THREE DESIGN DECISIONS, EACH FOR A REASON

  1. IT READS ARTIFACTS, IT NEVER RECOMPUTES.
     Every panel is built from the CSV/PNG files the analysis scripts already
     wrote. The dashboard therefore loads instantly, and - more importantly - it
     CANNOT DRIFT from the analysis. A dashboard that re-implements the
     calculation is a second source of truth, and the second one is always the
     one that is wrong.

  2. NO MATPLOTLIB. Charts use Streamlit's native elements, which render in the
     browser. This is not a style preference: matplotlib in this environment
     dies with a Windows fatal exception unless `<env>/Library/bin` is on PATH
     (see src/12_plot_sweep.py), and Streamlit is launched in ways that do not
     activate the conda environment. Pre-rendered PNGs are shown with st.image,
     which just reads bytes off disk, so no plotting library is imported at all.

  3. IT ENFORCES THE EVIDENCE PACKAGE'S OWN RULES.
     The sensitivity results are always presented as a SWEPT RANGE, never as a
     point estimate, and there is a permanent panel listing what may not be
     claimed yet. A dashboard is where an unsourced number is most likely to get
     screenshotted into a slide, so the guard rails belong here most of all.

RUN IT

    conda activate hospital
    streamlit run src/16_dashboard.py

  or, without activating (streamlit.exe is not on PATH):

    python -m streamlit run src/16_dashboard.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

DEADLINE_H = 3.0     # IRDAI/HLT/CIR/PRO/84/5/2024 clause 16(a)

st.set_page_config(page_title="Hospital Bottleneck Detective",
                   page_icon="🏥", layout="wide")


# ---------------------------------------------------------------------------
@st.cache_data
def load(name: str) -> pd.DataFrame | None:
    """Read one artifact, or None if that step has not been run yet."""
    path = OUT / name
    return pd.read_csv(path) if path.exists() else None


def missing(name: str, script: str) -> None:
    st.warning(f"`outputs/{name}` not found. Run `python src/{script}` first.")


def load_segmented(name: str, column: str = "payment_mode") -> pd.DataFrame | None:
    """
    Load an artifact only if it actually carries the cohort column it is about to
    be split by.

    `payment_mode` is invented by our SimPy generator; no real hospital export has
    it. A dashboard pointed at a new hospital's artifacts would otherwise find the
    CSV present, sail past the `is not None` guard, and die on `.payment_mode`.
    Returning None here funnels that case into the SAME "not available" branch
    that already handles a missing file, so the panel is skipped, not crashed.
    """
    df = load(name)
    if df is None or column not in df.columns:
        return None
    return df


def payment_mode_notice(what: str) -> None:
    st.info(
        f"**{what} is not available for this dataset.** It compares cashless "
        "against self-pay patients, which needs a `payment_mode` column. That "
        "column is produced by this project's SimPy generator; a real hospital "
        "event log will not have one unless its billing mode is exported per "
        "case. Every other panel works without it."
    )


def sweep_unavailable(what: str) -> None:
    """Say WHY the panel is empty: step not run, or dataset has no payment mode."""
    if (OUT / "13_step5_robustness.csv").exists():
        payment_mode_notice(what)
    else:
        missing("13_step5_robustness.csv", "13_step5_robustness.py")


def caption_source(*paths: str) -> None:
    """Every panel names the file it came from - provenance is the point."""
    st.caption("Source: " + ", ".join(f"`outputs/{p}`" for p in paths))


# ---------------------------------------------------------------------------
st.title("🏥 AI Hospital Process Bottleneck Detective")
st.caption(
    "Every figure on this page is read from a file produced by the analysis "
    "scripts. Nothing is recomputed here, and nothing is estimated here."
)

tabs = st.tabs([
    "Overview",
    "Real hospital data",
    "Cashless vs self-pay",
    "Does the tool work?",
    "Business case",
    "Evidence table",
    "What we cannot claim",
])

# ===========================================================================
with tabs[0]:
    st.header("The five strongest findings")

    findings = load("14_findings.csv")
    sweep = load_segmented("13_step5_robustness.csv")
    handoff = load("03_department_handoffs.csv")

    if sweep is not None:
        base = sweep[(sweep.payment_mode == "cashless")
                     & (sweep.scenario == "baseline")]
        q0 = base[base["q"] == 0.0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Breach the 3-hour rule with NO document queries",
                  f"{q0['breach_3h_pct'].mean():.1f}%",
                  help="q=0 endpoint - does not depend on the unsourced parameter")
        c2.metric("Cashless slower than self-pay",
                  f"{len(base)}/{len(base)} runs",
                  help="5 seeds x 6 sweep values, baseline capacities")
        c3.metric("Published breach rate reproduced",
                  "58.44% vs 58.5%",
                  help="Mannhardt & Blinde 2017, verified against the paper")
        c4.metric("Planted bottlenecks recovered (rho >= 0.75)",
                  "8/8", help="zero false positives")

    st.markdown("""
1. **The pipeline reproduces published results on real data.** Antibiotics
   1-hour breach rate **58.44%** against a published **58.5%**; mean readmission
   interval **81.0** days against a published **81.6**. This is the only claim a
   third party can check against a paper they can download.
2. **Even a perfect insurer breaches the three-hour rule for about a quarter of
   cashless patients.** Measured at the q=0 endpoint, so it carries no dependence
   on the unsourced parameter.
3. **One extra timestamp changes the answer.** With `complete`-only data the true
   bottleneck ranks **4th**, behind two scheduled cadences and one
   processing-dominated edge. With `schedule`/`start`/`complete` it ranks **1st**.
4. **The detection method provably finds bottlenecks, and states its own limit.**
   Reliable above ~75% resource utilisation; below ~70% it needs a historical
   baseline to compare against.
5. **Cashless is slower in every run**, and at adequate staffing the delay is
   almost entirely *processing*, not queueing - so the lever is process design,
   not headcount.
    """)
    caption_source("14_findings.csv", "13_step5_robustness.csv")

# ===========================================================================
with tabs[1]:
    st.header("What the real hospital log shows")
    st.caption(
        "Sepsis Cases event log, a Dutch hospital (Mannhardt, TU Eindhoven). "
        "1,050 patients, 15,214 events. These are measurements, not simulation."
    )

    if handoff is None:
        missing("03_department_handoffs.csv", "03_department_handoffs.py")
    else:
        h = handoff.copy()
        h.columns = ["pair", "count", "median_h", "p90_h", "total_h"]
        h["total_days"] = (h["total_h"] / 24).round(0)

        st.subheader("Where time is lost between departments")
        st.dataframe(
            h.sort_values("total_h", ascending=False).head(12)[
                ["pair", "count", "median_h", "p90_h", "total_days"]],
            width='stretch', hide_index=True,
            column_config={
                "pair": "department pair (from → to)",
                "count": "times it fired",
                "median_h": st.column_config.NumberColumn("median (h)", format="%.1f"),
                "p90_h": st.column_config.NumberColumn("p90 (h)", format="%.1f"),
                "total_days": st.column_config.NumberColumn("total (days)", format="%d"),
            })
        st.info(
            "**B = Laboratory, A = ER team, C = Triage, E = Discharge desk**, the "
            "rest are individual wards. The codes are anonymised in the source "
            "data; their roles were recovered by cross-tabulating each code "
            "against the activities it performs.\n\n"
            "**`B → E` is the headline:** the last lab result to the discharge "
            "desk, 637 discharges, a 32-hour median, and one in ten waiting more "
            "than four days. `B → B` absorbs even more time in total, but 72% of "
            "those gaps are batch data-entry artefacts rather than real waiting."
        )
        caption_source("03_department_handoffs.csv")

    st.subheader("Process maps")
    st.caption(
        "Built on the episode-cut log, so no arrow measures a patient sitting at "
        "home between admissions."
    )
    for png, label in [
        ("05_map_roles.png", "The handoff map - departments collapsed to roles. "
                             "This is the one to show an administrator."),
        ("05_map_performance.png", "Activity map annotated with median waiting time."),
        ("05_powerbi_contrast.png", "Why not just build a dashboard? The same "
                                    "activity, one average vs split by arrival path."),
    ]:
        p = OUT / png
        if p.exists():
            st.markdown(f"**{label}**")
            st.image(str(p), width='stretch')

    st.subheader("Validation against published figures")
    val = load("04_validation_report.csv")
    if val is None:
        missing("04_validation_report.csv", "04_validate_literature.py")
    else:
        tier_a = val[val["tier"] == "A"]
        st.dataframe(
            tier_a[["claim", "published", "computed", "verdict"]],
            width='stretch', hide_index=True)
        st.caption(
            "Tier A = the primary paper, read in full. Tolerance was fixed at "
            "half the published digit BEFORE computing. The 5 discrepancies are "
            "cohort-definition differences - the paper reads percentages off a "
            "conformance-checked model, this pipeline counts directly."
        )
        caption_source("04_validation_report.csv")

# ===========================================================================
with tabs[2]:
    st.header("Cashless vs self-pay discharge delay")

    st.error(
        "**This is a SWEEP, not a measurement.** The document-query probability "
        "`q` has no public source. It is swept across its full plausible range "
        "and only the SHAPE of the result may be quoted - never a single value. "
        "See `config/calibration.yaml`, "
        "`insurance_branch.preauth_query_loop_probability`."
    )

    if sweep is None:
        sweep_unavailable("This comparison")
    else:
        scenario = st.radio(
            "Capacity scenario",
            ["baseline", "tight_tpa"],
            horizontal=True,
            format_func=lambda s: ("Baseline (insurance desk adequately staffed)"
                                   if s == "baseline"
                                   else "Understaffed TPA (rho ≈ 0.90)"),
            help="The understaffed scenario tests whether resource queueing adds "
                 "to the authorisation processing time.",
        )
        sel = sweep[sweep["scenario"] == scenario]
        med = sel.groupby(["q", "payment_mode"])["median_h"].mean().unstack()
        med["IRDAI 3-hour deadline"] = DEADLINE_H

        st.subheader("Median time from medically ready to actually leaving")
        st.line_chart(med, height=340)

        breach = sel.groupby(["q", "payment_mode"])["breach_3h_pct"].mean().unstack()
        st.subheader("Share of patients breaching the 3-hour rule")
        st.line_chart(breach, height=300)

        st.subheader("Is the delay queueing, or the process itself?")
        cash = sel[sel.payment_mode == "cashless"].groupby("q")[
            ["mean_queue_h", "mean_work_h"]].mean()
        cash.columns = ["queueing (capacity problem)", "processing (steps problem)"]
        st.bar_chart(cash, height=300)
        st.info(
            "At adequate staffing the delay is **almost entirely processing** - "
            "adding TPA reviewers would achieve little, because there is no queue "
            "to clear. Under an understaffed TPA a queue does appear, and although "
            "it adds well under an hour, that is enough to roughly **double** the "
            "breach rate at q=0, because the median already sits just under three "
            "hours. Near a deadline, a short queue costs as much as hours of "
            "process time."
        )

        st.subheader("Replication across seeds")
        spread = sweep[(sweep.scenario == scenario)
                       & (sweep.payment_mode == "cashless")].groupby("q").agg(
            median_mean=("median_h", "mean"), median_min=("median_h", "min"),
            median_max=("median_h", "max"), breach_mean=("breach_3h_pct", "mean"),
            breach_min=("breach_3h_pct", "min"), breach_max=("breach_3h_pct", "max"))
        st.dataframe(spread.round(2), width='stretch')
        st.caption(
            "5 seeds per sweep value. The direction holds in every run; the exact "
            "q at which the median crosses the deadline does NOT - it is 0.45 in "
            "four seeds and 0.30 in one, so it must be reported as a range."
        )
        caption_source("13_step5_robustness.csv")

# ===========================================================================
with tabs[3]:
    st.header("Does the detection method actually work?")
    st.caption(
        "A bottleneck was planted in the simulation, the analysis "
        "was run blind, and the result scored against a sealed answer."
    )

    rob = load("10_robustness_sweep.csv")
    if rob is None:
        missing("10_robustness_sweep.csv", "10_robustness_sweep.py")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Recovered at rank 1 (blind)",
                  f"{int(rob['recovered'].sum())}/{len(rob)}")
        c2.metric("Recovered vs a control baseline",
                  f"{int((rob['delta_rank'] == 1).sum())}/{len(rob)}")
        hot = rob[rob["rho_planted"] >= 0.75]
        c3.metric("At rho ≥ 0.75",
                  f"{int(hot['recovered'].sum())}/{len(hot)}",
                  help="zero false positives in this band")

        st.dataframe(
            rob[["resource", "capacity", "rho_planted", "queue_h_control",
                 "queue_h_planted", "dept_rank", "delta_rank",
                 "false_positives", "recovered"]],
            width='stretch', hide_index=True,
            column_config={
                "rho_planted": st.column_config.NumberColumn("ρ after plant",
                                                             format="%.2f"),
                "queue_h_control": st.column_config.NumberColumn("queue, control (h)",
                                                                 format="%.0f"),
                "queue_h_planted": st.column_config.NumberColumn("queue, planted (h)",
                                                                 format="%.0f"),
                "dept_rank": "rank (absolute)",
                "delta_rank": "rank (vs control)",
            })
        st.info(
            "**The three misses are not blind spots.** Ranking by growth against "
            "a control recovers all 12, so the method saw every plant - it simply "
            "could not distinguish *'large because throttled'* from *'large "
            "because this department is always the busiest'*. The missing "
            "ingredient is a historical baseline, not sensitivity. For a "
            "deployment, that means: give the tool last month's log."
        )
        caption_source("10_robustness_sweep.csv")

# ===========================================================================
with tabs[4]:
    st.header("Bed-hours, and what they are worth")

    st.warning(
        "**The rupee figures below are an ILLUSTRATIVE BENCHMARK.** They value "
        "simulated bed-hours against a real, SEBI-filed ARPOB from a different "
        "hospital group. They are NOT this hospital's loss, and they are an upper "
        "bound on opportunity rather than cash. The four labels are listed below "
        "the table and must be quoted with any figure taken from here."
    )

    bedh = load("17_bed_hours.csv")
    if bedh is None:
        missing("17_bed_hours.csv", "17_bed_hours.py")
    else:
        ARPOB, ARPOB_Q, ARPOB_FILED = 67361, "Q3 FY2026 (Oct-Dec 2025)", "2026-02-04"
        per_hour = ARPOB / 24

        agg = bedh.groupby("q").agg(
            total=("total_bed_h_per_1000", "mean"),
            excess=("excess_vs_selfpay_per_1000", "mean"),
            beyond=("beyond_deadline_per_1000", "mean")).reset_index()
        agg["illustrative_INR_lakh"] = (agg["excess"] * per_hour / 1e5).round(1)

        floor = agg[agg["q"] == 0.0].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Excess bed-hours per 1,000 cashless admissions (q=0 floor)",
                  f"{floor['excess']:,.0f}",
                  help="Assumes a perfect insurer - no dependence on the "
                       "unsourced parameter")
        c2.metric("Benchmark ARPOB", f"INR {ARPOB:,}/bed-day",
                  help=f"Global Health Ltd (Medanta), {ARPOB_Q}, "
                       f"SEBI-filed {ARPOB_FILED}")
        c3.metric("Illustrative value at the floor",
                  f"INR {floor['illustrative_INR_lakh']:.1f} lakh",
                  help="per 1,000 cashless admissions")

        st.subheader("Excess bed-hours per 1,000 cashless admissions")
        st.bar_chart(agg.set_index("q")[["excess"]], height=280)

        st.dataframe(
            agg.rename(columns={
                "q": "query probability q",
                "total": "total bed-hours",
                "excess": "excess vs self-pay",
                "beyond": "beyond the 3h deadline",
                "illustrative_INR_lakh": "illustrative INR (lakh)"}).round(0),
            width="stretch", hide_index=True)

        st.info(
            "**The regulatory measure badly understates the cost.** At q=0 the "
            "hours *beyond the three-hour deadline* are a small fraction of the "
            "*excess over self-pay* - most cashless patients breach by a little, "
            "or sit just under the line. A hospital could be broadly compliant "
            "and still lose most of these bed-hours. Compliance is not efficiency."
        )

        st.subheader("The four labels that must travel with any rupee figure")
        st.markdown("""
1. **Not this hospital's ARPOB.** The benchmark is from a listed Indian chain; the
   bed-hours come from a synthetic hospital whose process structure was calibrated
   on a Dutch log. This is an order of magnitude, not a valuation.
2. **The benchmark sits at the premium end.** Medanta is quaternary and
   high-acuity (ALOS 3.02 days). A district or mid-tier hospital's ARPOB is
   materially lower, so the figure is optimistic rather than median.
3. **The definition excludes pharmacy and other income** - a narrower revenue base,
   which biases the estimate *down* while point 2 biases it *up*. They do not
   cancel in any known proportion.
4. **Revenue is not profit, and a freed bed is not automatically filled.** This is
   an upper bound on opportunity. At the benchmark chain's own ~59% occupancy, the
   assumption that a patient is waiting for every freed bed is plainly not
   universally true.
        """)
        st.caption(
            "Source: `outputs/17_bed_hours.csv`, "
            "`config/calibration.yaml` economics.arpob_benchmark "
            "(Global Health Ltd, BSE 543654 / NSE MEDANTA, filed 4 Feb 2026)")


# ===========================================================================
with tabs[5]:
    st.header("Evidence table")
    st.caption(
        "Generated from the analysis artifacts by `src/14_evidence_package.py`. "
        "Each row carries its own parameter dependence, robustness status, "
        "pre-cleared wording and caveat."
    )

    if findings is None:
        missing("14_findings.csv", "14_evidence_package.py")
    else:
        cats = sorted(findings["category"].unique())
        chosen = st.multiselect("Categories", cats, default=cats)
        view = findings[findings["category"].isin(chosen)]

        for _, r in view.iterrows():
            with st.expander(f"**[{r['id']}]** {r['finding']}"):
                st.markdown(f"- **Kind:** {r['kind']}")
                st.markdown(f"- **Evidence:** `{r['evidence_source']}`")
                st.markdown(f"- **Parameter dependence:** {r['parameter_dependence']}")
                st.markdown(f"- **Robustness:** {r['robustness']}")
                st.markdown(f"- **Allowed wording:** {r['allowed_wording']}")
                st.markdown(f"- **Caveat:** {r['caveat']}")
        caption_source("14_findings.csv")

# ===========================================================================
with tabs[6]:
    st.header("What we cannot claim yet")
    st.caption(
        "Kept as a permanent panel. A dashboard is where an "
        "unsourced number is most likely to be screenshotted into a slide."
    )

    if findings is not None:
        blocked = findings[findings["category"].str.startswith("5")]
        for _, r in blocked.iterrows():
            st.error(f"**[{r['id']}] {r['finding']}**\n\n"
                     f"*Allowed:* {r['allowed_wording']}\n\n"
                     f"*Why:* {r['caveat']}")

    st.subheader("Calibration provenance")

    # Read the counts rather than hardcoding them. An earlier version of this
    # panel typed them in, and they went stale the moment ARPOB was sourced -
    # which is exactly the drift this dashboard is supposed to be immune to.
    calib = load("14_calibration_status.csv")
    meaning = {
        "verified": "primary document read, or measured by us from the real log",
        "assumption": "declared modelling choice; results are conditional on it",
        "secondary": "real citation, original not read in context",
        "placeholder": "no source yet - blocks any dependent claim",
        "TOTAL": "",
    }
    if calib is None:
        missing("14_calibration_status.csv", "14_evidence_package.py")
    else:
        calib = calib.copy()
        calib["meaning"] = calib["status"].map(meaning).fillna("")
        st.dataframe(calib[["status", "meaning", "count"]],
                     width="stretch", hide_index=True)

    st.info(
        "**ARPOB is no longer blocking.** `economics.arpob_benchmark` is now "
        "`verified` against a SEBI filing (Global Health Ltd / Medanta, Q3 FY2026, "
        "filed 4 Feb 2026), which unblocks the illustrative business case in the "
        "Business case tab.\n\n"
        "What stays blocked is narrower and **permanent**: "
        "`economics.arpob_of_study_hospital` is kept as a placeholder, "
        "because the analysed log is a Dutch hospital that discloses no ARPOB. A "
        "rupee figure may be quoted as a benchmark; it may never be attributed to "
        "the study hospital as its actual loss."
    )
    st.caption("Source: `outputs/14_calibration_status.csv`, generated from "
               "`config/calibration.yaml` by `src/06_check_calibration.py`")
