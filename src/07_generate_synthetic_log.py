"""
STEP 3b - The SimPy generator: a synthetic Indian hospital log that can separate
          WAITING time from PROCESSING time.

THE ONE IDEA THIS RESTS ON

  You do NOT tell SimPy "registration takes 95 minutes". You tell it two things:

      service time - how long a clerk actually takes with one patient
      capacity     - how many clerks there are

  Then patients arrive, and the QUEUE FORMS BY ITSELF. If patients arrive faster
  than the clerks can clear them, waiting time grows - and nobody specified it.
  It emerges from contention for a scarce resource.

  That is why this log can do what the real Sepsis log provably cannot. When a
  patient seizes a resource we write a `start` event; when the work finishes we
  write a `complete` event. So:

      complete - start            = PROCESSING time (the work itself)
      start - arrived at the step = WAITING time    (the queue)

  Step 1 proved the real log fuses these forever (every event is `complete`, there
  are no `start` events). Here we control the ground truth, so they come apart.
  This is the justification for building a simulator rather than only analysing
  public data.

THE SECOND STRUCTURAL IDEA - THE INSURANCE BRANCH HOLDS A BED

  A cashless patient becomes clinically fit for discharge, and then the file sits
  with the TPA. The bed stays occupied while a document loop runs. That loop is
  deliberately the same SHAPE as the lab loop measured in the real data - request,
  review, query, resubmit - because the project's hypothesis is that repeated
  document queries, not the regulatory clock, are the real cause of delay.

  The headline measurement is therefore:

      `Clinically Fit for Discharge`  ->  `Discharge`,  split by payment_mode

  which converts directly into bed-hours for the Step 8 business case.

OUTPUT SCHEMA - deliberately identical to the real log, so the Step 2 pipeline
runs on it unchanged:

  case:concept:name, concept:name, concept:instance, time:timestamp,
  lifecycle:transition (schedule/start/complete), org:group, payment_mode

EVERY PARAMETER comes from config/calibration.yaml. Nothing is hard-coded here,
so "where did that number come from?" is always answered by opening that file.
"""

import argparse
import importlib.util
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import simpy

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "synthetic"

# Load the calibration loader by path, because the module name starts with a digit
# and cannot be imported normally.
_spec = importlib.util.spec_from_file_location(
    "calib", ROOT / "src" / "06_check_calibration.py"
)
calib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calib)

# The simulation clock counts MINUTES from this instant. Like the real 4TU log,
# the absolute dates are arbitrary - only within-case durations carry meaning.
SIM_EPOCH = datetime(2025, 4, 1, 0, 0, 0)

# Which department owns each activity. Mirrors `org:group` in the real log, which
# is what made the Step 2b handoff analysis possible - so the synthetic log must
# carry the same thing or our own pipeline loses half its power.
GROUPS = {
    "Registration": "FrontDesk",
    "Triage": "TriageDesk",
    "Consultation": "Doctor",
    "Lab Test": "Laboratory",
    "Admission": "Ward",
    "Ward Round": "Ward",
    "Clinically Fit for Discharge": "Ward",
    "Preauth Request": "InsuranceDesk",
    "Preauth Review": "TPA",
    "Preauth Document Query": "TPA",
    "Preauth Approved": "TPA",
    "Discharge Auth Request": "InsuranceDesk",
    "Discharge Auth Review": "TPA",
    "Discharge Document Query": "TPA",
    "Discharge Authorized": "TPA",
    "Billing": "Billing",
    "Discharge": "DischargeDesk",
}


@dataclass
class EventLog:
    """Collects events. Each completed step produces TWO rows: start and complete."""
    rows: list = field(default_factory=list)
    _counter: dict = field(default_factory=dict)

    def record(self, case_id, activity, group, t_schedule, t_start, t_complete,
               payment_mode):
        # `concept:instance` is the XES standard attribute that ties a `start` to
        # ITS OWN `complete`. It is required here, not decorative: activities like
        # `Lab Test` and `Ward Round` repeat many times within one admission, so
        # pairing on (case, activity) alone would merge separate occurrences and
        # silently report one enormous processing time instead of several real ones.
        key = (case_id, activity)
        self._counter[key] = self._counter.get(key, 0) + 1
        instance = f"{activity}#{self._counter[key]}"

        # THE FULL XES LIFECYCLE, not just start/complete:
        #   schedule -> the patient JOINED THE QUEUE for this resource
        #   start    -> the resource was seized and work began
        #   complete -> the work finished
        # so   start - schedule = QUEUEING   and   complete - start = PROCESSING,
        # both read straight off the log with no inference. This matters: an
        # earlier version derived waiting from the gap between consecutive events
        # and wrongly counted the deliberate 24-hour ward-round interval as queue
        # time. A scheduled cadence is not a queue, and `schedule` is exactly the
        # standard attribute that tells them apart.
        for transition, minutes in (("schedule", t_schedule),
                                    ("start", t_start),
                                    ("complete", t_complete)):
            self.rows.append({
                "case:concept:name": case_id,
                "concept:name": activity,
                "concept:instance": instance,
                "time:timestamp": SIM_EPOCH + timedelta(minutes=float(minutes)),
                "lifecycle:transition": transition,
                "org:group": group,
                "payment_mode": payment_mode,
            })

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows).sort_values(
            ["case:concept:name", "time:timestamp"], kind="stable"
        ).reset_index(drop=True)


class Hospital:
    """
    The scarce resources. Capacity is what makes queues form, so these numbers ARE
    the experiment - any bottleneck the pipeline later finds is a consequence of
    them, which is why they are declared in the calibration file and reported
    alongside every result.
    """

    def __init__(self, env: simpy.Environment, capacity: dict):
        self.env = env
        self.registration = simpy.Resource(env, capacity["registration_clerks"])
        self.triage = simpy.Resource(env, capacity["triage_nurses"])
        self.doctors = simpy.Resource(env, capacity["doctors"])
        self.lab = simpy.Resource(env, capacity["lab_stations"])
        self.beds = simpy.Resource(env, capacity["beds"])
        self.insurance_desk = simpy.Resource(env, capacity["insurance_desk_staff"])
        self.tpa = simpy.Resource(env, capacity["tpa_reviewers"])


def triangular(rng: random.Random, spec: list) -> float:
    """
    Triangular sample from [shortest, most likely, longest].

    Triangular because it is the distribution you can actually elicit from someone
    who runs the desk - "best case, usual, bad day" - rather than one that needs
    fitting to data we do not have.
    """
    low, mode, high = spec
    return rng.triangular(low, high, mode)


def do_step(env, log, patient, activity, resource, minutes, rng=None):
    """
    One step of care through a scarce resource. This is where the whole
    waiting-vs-processing separation actually happens:

        t_request -> queueing starts (patient joins the line)
        t_start   -> resource seized. WAITING = t_start - t_request
        t_done    -> work finished.   PROCESSING = t_done - t_start

    We emit `start` at t_start and `complete` at t_done, exactly as the XES
    lifecycle standard intends, so any downstream tool can recover both.
    """
    t_request = env.now                 # joins the queue here
    with resource.request() as req:
        yield req                       # <- the queue. Nobody specified its length.
        t_start = env.now
        yield env.timeout(minutes)
        t_done = env.now
    log.record(patient.case_id, activity, GROUPS[activity], t_request, t_start,
               t_done, patient.payment_mode)


def instant(env, log, patient, activity):
    """
    A milestone with no duration - a state change, not work (e.g. 'Discharge').

    All three lifecycle timestamps are identical, so it contributes zero queueing
    and zero processing rather than polluting either measure.
    """
    now = env.now
    log.record(patient.case_id, activity, GROUPS[activity], now, now, now,
               patient.payment_mode)


@dataclass
class Patient:
    case_id: str
    payment_mode: str
    rng: random.Random = None       # see `common_random_numbers` in arrivals()


def authorization_cycle(env, log, patient, hosp, cfg, stage: str):
    """
    The insurance loop - the project's central hypothesis, made mechanical.

    The desk prepares a request, the TPA reviews it, and with probability
    `document_query_probability` the answer that comes back is not a decision but
    a QUERY for more documents - so the cycle repeats. The bed is held throughout.

    `stage` is "preauth" or "discharge"; the shape is identical, which is itself
    the point - the same failure mode appears twice in one admission.

    Activity names are looked up explicitly rather than built by string
    concatenation, so they cannot drift out of step with GROUPS.
    """
    svc = cfg["simulation_assumptions"]["service_times"]["value"]
    q_prob = cfg["sensitivity"]["document_query_probability"]["value"]
    max_loops = cfg["sensitivity"]["max_query_loops"]["value"]

    names = {
        "preauth": {
            "request": "Preauth Request",
            "review": "Preauth Review",
            "query": "Preauth Document Query",
            "done": "Preauth Approved",
        },
        "discharge": {
            "request": "Discharge Auth Request",
            "review": "Discharge Auth Review",
            "query": "Discharge Document Query",
            "done": "Discharge Authorized",
        },
    }[stage]
    request_activity = names["request"]
    review_activity = names["review"]
    query_activity = names["query"]
    done_activity = names["done"]

    loops = 0
    while True:
        yield from do_step(env, log, patient, request_activity, hosp.insurance_desk,
                           triangular(patient.rng, svc["insurance_desk_prepare_request"]))
        yield from do_step(env, log, patient, review_activity, hosp.tpa,
                           triangular(patient.rng, svc["tpa_review"]))

        if patient.rng.random() >= q_prob or loops >= max_loops:
            break

        # Not a decision - a request for more documents. Round we go again.
        loops += 1
        yield from do_step(env, log, patient, query_activity, hosp.tpa,
                           triangular(patient.rng, svc["tpa_document_query_rework"]))

    instant(env, log, patient, done_activity)


def patient_journey(env, log, patient, hosp, cfg):
    """One patient's whole pathway through the hospital."""
    a = cfg["simulation_assumptions"]
    svc = a["service_times"]["value"]

    yield from do_step(env, log, patient, "Registration", hosp.registration,
                       triangular(patient.rng, svc["registration"]))
    yield from do_step(env, log, patient, "Triage", hosp.triage,
                       triangular(patient.rng, svc["triage"]))
    yield from do_step(env, log, patient, "Consultation", hosp.doctors,
                       triangular(patient.rng, svc["consultation"]))

    if patient.rng.random() > a["probability_admitted"]["value"]:
        instant(env, log, patient, "Discharge")     # sent home from the ER
        return

    # Cashless patients need pre-authorisation before admission.
    if patient.payment_mode == "cashless":
        yield from authorization_cycle(env, log, patient, hosp, cfg, "preauth")

    # The bed is seized here and NOT released until the very end - including
    # through the discharge authorisation wait. That is the whole economic point:
    # an occupied bed cannot admit the next patient.
    # The wait for a BED is recorded explicitly, because bed-blocking is the
    # quantity the Step 8 business case monetises. Requesting the bed outside
    # do_step() (it is held far longer than one step) would otherwise leave this
    # queue invisible in the log - the most expensive wait in the hospital,
    # unmeasured.
    t_bed_requested = env.now
    bed = hosp.beds.request()
    yield bed
    log.record(patient.case_id, "Admission", GROUPS["Admission"],
               t_bed_requested, env.now, env.now, patient.payment_mode)

    los_days = triangular(patient.rng, a["length_of_stay_days"]["value"])
    round_interval = a["ward_round_interval"]["value"] * 60      # hours -> minutes
    elapsed = 0.0
    while elapsed < los_days * 24 * 60:
        yield env.timeout(round_interval)
        elapsed += round_interval
        instant(env, log, patient, "Ward Round")
        yield from do_step(env, log, patient, "Lab Test", hosp.lab,
                           triangular(patient.rng, svc["lab_test"]))

    # The clock that matters starts HERE: the patient is medically ready to leave.
    instant(env, log, patient, "Clinically Fit for Discharge")

    if patient.payment_mode == "cashless":
        yield from authorization_cycle(env, log, patient, hosp, cfg,
                                       "discharge")
    else:
        yield from do_step(env, log, patient, "Billing", hosp.insurance_desk,
                           triangular(patient.rng, svc["billing_self_pay"]))

    instant(env, log, patient, "Discharge")
    hosp.beds.release(bed)


def arrivals(env, log, hosp, cfg, rng, n_patients: int, seed: int,
             common_random_numbers: bool = False):
    """
    Poisson arrivals: exponential gaps with the configured mean.

    COMMON RANDOM NUMBERS (`common_random_numbers=True`). With a single shared
    RNG, changing a capacity changes the order in which processes interleave, so
    every patient draws DIFFERENT random numbers. Two runs would then differ for
    two reasons at once - the capacity change and the reshuffled randomness - and
    no controlled experiment is possible.

    Giving each patient a private RNG seeded from (seed, index) fixes their
    service times, payment mode, admission decision and query-loop outcomes
    regardless of scheduling. Only the QUEUEING differs between runs, which is
    exactly the variable under test. Used by the Step 4.5 planted-bottleneck
    experiment; off by default so the Step 4 production log stays reproducible.
    """
    a = cfg["simulation_assumptions"]
    mean_gap = a["arrival_interval_mean"]["value"]
    cashless_share = a["payment_mode_cashless_share"]["value"]

    for i in range(n_patients):
        yield env.timeout(rng.expovariate(1.0 / mean_gap))
        p_rng = random.Random(f"{seed}-{i}") if common_random_numbers else rng
        mode = "cashless" if p_rng.random() < cashless_share else "self_pay"
        patient = Patient(case_id=f"P{i + 1:05d}", payment_mode=mode, rng=p_rng)
        env.process(patient_journey(env, log, patient, hosp, cfg))


def check_stability(cfg: dict, verbose: bool = True) -> tuple:
    """
    Refuse to run an unstable queue.

    For every resource, rho = (arrival rate x mean service time) / capacity.
    If rho >= 1, work arrives faster than the resource can clear it and the queue
    grows without bound. The simulation then measures A BACKLOG BUILDING UP, not a
    hospital operating - and every duration it reports is a function of how long
    you ran it rather than of the process.

    This is the failure that produced an 88-hour bed wait in the first version of
    this model: beds carried an offered load of ~210 against a capacity of 60. It
    looked like a dramatic finding. It was an artefact.

    Mean of a triangular [low, mode, high] is (low + mode + high) / 3.
    """
    a = cfg["simulation_assumptions"]
    svc = a["service_times"]["value"]
    cap = a["capacity"]["value"]
    q = cfg["sensitivity"]["document_query_probability"]["value"]
    max_loops = cfg["sensitivity"]["max_query_loops"]["value"]

    mean = lambda spec: sum(spec) / 3.0

    rate = 1.0 / a["arrival_interval_mean"]["value"]      # patients per minute
    p_admit = a["probability_admitted"]["value"]
    cashless = a["payment_mode_cashless_share"]["value"]
    los_min = mean(a["length_of_stay_days"]["value"]) * 24 * 60
    tests_per_stay = los_min / (a["ward_round_interval"]["value"] * 60)

    # Expected submissions per authorisation cycle, capped at max_loops.
    submissions = sum(q ** k for k in range(max_loops + 1))
    queries = submissions - 1

    admitted = rate * p_admit
    # Pre-authorisation happens ONLY for patients who are actually admitted - see
    # patient_journey(), where the admit decision precedes the preauth call. An
    # earlier version of this check used `rate * cashless` for the preauth term,
    # i.e. every cashless arrival, which overstated the load on the insurance desk
    # and the TPA by roughly 2x.
    #
    # The error was conservative (it never let an unstable config through) but it
    # made the DECLARED rho for those two resources wrong, and the Step 4.5
    # robustness sweep caught it: a TPA plant at a supposed rho of 0.92 produced a
    # queue of exactly zero, which is impossible at 92% utilisation.
    #
    # Each admitted cashless patient goes through TWO authorisation cycles:
    # preauth on the way in, discharge authorisation on the way out.
    auth_cycles = admitted * cashless * 2

    loads = {
        "registration_clerks": rate * mean(svc["registration"]),
        "triage_nurses": rate * mean(svc["triage"]),
        "doctors": rate * mean(svc["consultation"]),
        "lab_stations": admitted * tests_per_stay * mean(svc["lab_test"]),
        "beds": admitted * los_min,
        "insurance_desk_staff": (
            auth_cycles * submissions * mean(svc["insurance_desk_prepare_request"])
            + admitted * (1 - cashless) * mean(svc["billing_self_pay"])),
        "tpa_reviewers": (
            auth_cycles * submissions * mean(svc["tpa_review"])
            + auth_cycles * queries * mean(svc["tpa_document_query_rework"])),
    }

    if verbose:
        print(f"\n{'=' * 78}\nQUEUEING STABILITY CHECK (rho must be < 1)\n{'=' * 78}")
    unstable, rhos = [], {}
    for name, load in loads.items():
        rho = load / cap[name]
        rhos[name] = rho
        flag = "OK" if rho < 1 else "UNSTABLE"
        if rho >= 1:
            unstable.append((name, rho, load))
        if verbose:
            print(f"  {name:<24} load {load:>8.1f}  capacity {cap[name]:>4}  "
                  f"rho {rho:>5.2f}  {flag}")
    # Return the computed loads as well, so a caller (the Step 4.5 robustness
    # sweep) can report rho for each plant without duplicating this formula.
    return unstable, {"loads": loads, "rhos": rhos, "capacity": dict(cap)}


def run(n_patients: int, seed: int, query_probability: float | None,
        capacity_override: dict | None = None,
        common_random_numbers: bool = False,
        verbose: bool = True) -> tuple:
    cfg = calib.load_calibration()

    # Allow the sensitivity parameter to be overridden per run. Step 5 sweeps it;
    # a single value must never be quoted as a finding.
    if query_probability is not None:
        cfg["sensitivity"]["document_query_probability"]["value"] = query_probability

    # Used by the Step 4.5 planted-bottleneck experiment: change ONE capacity and
    # leave everything else identical. The stability check below still runs, so a
    # plant that made the queue unbounded would be refused rather than measured.
    if capacity_override:
        cfg["simulation_assumptions"]["capacity"]["value"].update(capacity_override)

    unstable, _stability = check_stability(cfg, verbose=verbose)
    if unstable:
        names = ", ".join(f"{n} (rho={r:.2f})" for n, r, _ in unstable)
        raise SystemExit(
            f"\n  REFUSING TO RUN - unstable queue(s): {names}\n"
            "  The queue would grow without bound, so every duration this run\n"
            "  reported would be a function of run length, not of the process.\n"
            "  Raise the capacity in config/calibration.yaml, or lower the load.")

    rng = random.Random(seed)
    env = simpy.Environment()
    log = EventLog()
    hosp = Hospital(env, cfg["simulation_assumptions"]["capacity"]["value"])

    env.process(arrivals(env, log, hosp, cfg, rng, n_patients, seed,
                         common_random_numbers))
    env.run()
    return log.to_frame(), cfg


def summarise(df: pd.DataFrame) -> None:
    """
    Prove the log does the one thing it was built to do: separate waiting from
    processing. If this section is empty, the generator has failed at its purpose.
    """
    print(f"\n{'=' * 78}\nDOES THE LOG SEPARATE WAITING FROM PROCESSING?\n{'=' * 78}")

    # Pair each `start` with its own `complete` via concept:instance.
    keys = ["case:concept:name", "concept:instance", "concept:name"]
    starts = df[df["lifecycle:transition"] == "start"].set_index(keys)["time:timestamp"]
    ends = df[df["lifecycle:transition"] == "complete"].set_index(keys)["time:timestamp"]
    paired = pd.concat([starts.rename("start"), ends.rename("complete")], axis=1).dropna()
    paired["processing_min"] = (
        paired["complete"] - paired["start"]).dt.total_seconds() / 60

    work = paired[paired["processing_min"] > 0]
    print(f"  activity instances paired start->complete : {len(paired):,}")
    print(f"  of those, with real processing time       : {len(work):,}")
    print(f"  lifecycle values present                  : "
          f"{sorted(df['lifecycle:transition'].unique())}")
    print("\n  median PROCESSING time per activity (the work itself):")
    by_activity = work.groupby(level=2)["processing_min"].median().sort_values(
        ascending=False)
    for activity, minutes in by_activity.items():
        print(f"    {activity:<32} {minutes:>7.1f} min")

    # ---- THE CAPABILITY CLAIM, ACTUALLY DEMONSTRATED ------------------------
    # Queueing is now read directly off the log as start - schedule. No inference,
    # and a scheduled cadence (the 24h ward round) correctly contributes zero.
    sched = df[df["lifecycle:transition"] == "schedule"].set_index(
        keys)["time:timestamp"]
    paired["schedule"] = sched
    paired["waiting_min"] = (
        paired["start"] - paired["schedule"]).dt.total_seconds() / 60
    waits = paired[paired["waiting_min"] > 0]

    total_wait = paired["waiting_min"].sum()
    total_work = paired["processing_min"].sum()
    total = total_wait + total_work

    print(f"\n{'=' * 78}\nTHE THING THE REAL LOG CANNOT DO\n{'=' * 78}")
    print(f"  total QUEUEING time   : {total_wait / 60:>10,.0f} h  "
          f"({total_wait / total * 100:>4.1f}%)")
    print(f"  total PROCESSING time : {total_work / 60:>10,.0f} h  "
          f"({total_work / total * 100:>4.1f}%)")
    print(
        "\n  On the real Sepsis log this split is impossible: every event is\n"
        "  `complete`, so queueing and work are fused into one number forever\n"
        "  (proven in Step 1). Here it is a subtraction, because we control the\n"
        "  ground truth. THIS is what the simulator buys, and it is what lets a\n"
        "  recommendation say 'add a clerk' rather than 'change the procedure'."
    )

    print("\n  where the QUEUES actually form (median wait before starting):")
    q = waits.groupby(level=2)["waiting_min"].agg(
        n="size", median="median").sort_values("median", ascending=False)
    for activity, row in q.head(8).iterrows():
        print(f"    {activity:<32} {row['median']:>7.1f} min   (n={int(row['n']):,})")

    print(f"\n{'=' * 78}\nTHE HEADLINE MEASUREMENT - discharge cycle time by payment mode\n{'=' * 78}")
    print("  Clinically Fit for Discharge -> Discharge. This is bed time that no")
    print("  longer has any clinical purpose.\n")

    fit = df[(df["concept:name"] == "Clinically Fit for Discharge")
             & (df["lifecycle:transition"] == "complete")]
    out = df[(df["concept:name"] == "Discharge")
             & (df["lifecycle:transition"] == "complete")]
    merged = fit.merge(out, on="case:concept:name", suffixes=("_fit", "_out"))
    merged["hours"] = (merged["time:timestamp_out"]
                       - merged["time:timestamp_fit"]).dt.total_seconds() / 3600

    for mode, grp in merged.groupby("payment_mode_fit"):
        print(f"    {mode:<10} n={len(grp):>4}   median {grp['hours'].median():>6.1f} h"
              f"   p90 {grp['hours'].quantile(0.9):>6.1f} h")

    if merged["payment_mode_fit"].nunique() == 2:
        med = merged.groupby("payment_mode_fit")["hours"].median()
        print(f"\n    difference in medians: "
              f"{med.get('cashless', 0) - med.get('self_pay', 0):+.1f} h")
    print("\n  NOTE: conditional on an UNSOURCED document-query probability.")
    print("  Not quotable until Step 5 sweeps it. See config/calibration.yaml.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patients", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--query-probability", type=float, default=None,
                    help="override the sensitivity parameter for this run")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    df, cfg = run(args.patients, args.seed, args.query_probability)

    print(f"\n{'=' * 78}\nGENERATED\n{'=' * 78}")
    print(f"  patients        : {df['case:concept:name'].nunique():,}")
    print(f"  events          : {len(df):,}")
    print(f"  activities      : {df['concept:name'].nunique()}")
    print(f"  departments     : {sorted(df['org:group'].unique())}")
    per_patient = df.drop_duplicates("case:concept:name")["payment_mode"]
    print(f"  payment modes   : {per_patient.value_counts().to_dict()} (patients)")

    summarise(df)

    if args.out:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / args.out
        df.to_csv(path, index=False)

        # Every run ships the assumptions it was produced under. A log without its
        # manifest is not evidence of anything.
        manifest = {
            "patients": args.patients,
            "seed": args.seed,
            "document_query_probability":
                cfg["sensitivity"]["document_query_probability"]["value"],
            "capacity": cfg["simulation_assumptions"]["capacity"]["value"],
            "service_times": cfg["simulation_assumptions"]["service_times"]["value"],
            "warning": "Conditional on an unsourced sensitivity parameter. "
                       "Do not quote a single-run result.",
        }
        path.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\n  wrote {path.relative_to(ROOT)}")
        print(f"  wrote {path.with_suffix('.manifest.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
