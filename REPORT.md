# AI Hospital Process Bottleneck Detective

**An evidence-guarded system for locating, explaining and defending process
bottlenecks in hospital event data**

MBA (Systems & Digital Business) capstone report

---

## 1. Summary

Hospitals generate event logs as a by-product of running. Those logs record when
things happened, and therefore where time is lost, but reading them requires
process-mining skill that hospital administrators do not have, and the obvious
remedy, handing the log to a language model, produces fluent claims nobody can
check.

This project builds and validates a system that separates the two jobs:

> **Deterministic code computes every number. A language model interprets those
> numbers and is mechanically prevented from producing any of its own.**

The process mining itself is standard. The contribution is the verification
layer: the system states what a given log can and cannot support,
derives prohibitions from that, and enforces them on the model's output before
anything is written to disk. When the model reaches for a claim the data cannot
carry, the output is rejected and nothing is saved.

Three independent validations were run, each answering a different question:

| Validation | Question | Result |
|---|---|---|
| Published literature (§4) | Does the pipeline reproduce known figures? | 58.44% against a published **58.5%**, plus eight further figures matched; six discrepancies documented rather than hidden |
| Holdout log (§8) | Does it work on a hospital nobody looked at? | **18/18** checks on an unseen 451,359-event log |
| Ground truth (§9) | Are its answers *correct*? | **24/24** injected defects recovered; 19/19 diagnoses correct; **0/3 false positives** — originally 3/3, which is what forced §10 |

The third row is the most important. The system locates defects reliably, but
had no way to report that a process was healthy until this was measured and
addressed (Section 10).

---

## 2. The question

A hospital administrator asks four things, in order:

1. Where is the bottleneck?
2. What is the evidence?
3. What is probably driving it?
4. What should we do, and why is that the relevant thing to do?

Question 1 is arithmetic. Questions 3 and 4 are judgment. Question 2 is what makes
the other three usable, and it is the one most systems skip.

A business-intelligence dashboard answers question 1 badly: it reports average wait
per activity, which averages away the fact that delay is path-dependent. A
language model answers 3 and 4 fluently and cannot be trusted on 1 or 2, because
nothing stops it inventing the number it needs.

The design follows from that division:

```
event log  →  process mining        computes every number
           →  evidence extractor    states them as findings, each with its caveat
                                    and with the claims this log CANNOT support
           →  language model        interprets, hypothesises, recommends
           →  guard                 verifies the model introduced nothing
```

---

## 3. Data

Sepsis Cases, Event Log (Mannhardt, TU Eindhoven; DOI
`10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460`): **15,214 events, 1,050
patients, 16 activities** from a Dutch hospital's sepsis pathway. Not
redistributed; `src/00_download_data.py` refetches and md5-verifies it.

Two properties of this log shaped the entire project, and both were established by
inspection rather than assumed:

- `lifecycle:transition` is 100% filled and every value is `complete`. There
  are no `start` events. **Waiting time and processing time therefore cannot be
  separated in this data.** This is proven, not suspected, and it is why the
  simulator in §6 emits the full `schedule`/`start`/`complete` lifecycle, and why
  the system forbids any queueing-versus-processing claim about the real log.
- `org:group` is 100% filled, so every event carries a responsible department.
  This permits handoff analysis between departments, not merely between activities.

Lab tests dominate: Leucocytes 22.2% + CRP 21.4% + LacticAcid 9.6% = **53% of all
events**. Rework is visible in the spread of events per case: min 3, median 13,
mean 14.5, max 185.

4TU randomised the calendar dates but did not alter the time between events
within a case. All analysis therefore uses within-case durations only.

---

## 4. Validation against published findings

A pipeline that "resembles the literature" has not been validated. The gate
(`src/04_validate_literature.py`) reproduces *specific numbers* from Mannhardt &
Blinde (2017), whose full text was read rather than quoted second-hand.

| Claim | Published | Computed | Verdict |
|---|---|---|---|
| Antibiotics 1-hour rule violated | 58.5% | **58.44%** | near (0.1% off) |
| Mean days until return | 81.6 | **81.0** | near (0.7% off) |
| Distinct trace variants | 845 | **846** | near (0.1% off) |
| Trace uniqueness | 80% | **80.57%** | near (0.7% off) |
| Admitted to intensive care | 6.8% | **6.86%** | near (0.8% off) |
| Patients returning to ER | 27.8% | **28.0%** | near (0.7% off) |
| Cases / activities | 1,050 / 16 | 1,050 / 16 | exact |

**Five figures did not reproduce, and are reported as failures rather than
smoothed away:** mean triage-to-antibiotics time (1.86 h against a published 1.7),
the lactic-acid rule (1.28% against 0.7%), leaving ER without admission (22.9%
against 18.1%), normal care then directly to IC (2.29% against 3.6%), and the most
frequent variant (35 cases against 46).

All five turn on cohort definition, which patients count as eligible, and
each discrepancy is documented and traced to that choice. That the headline figure
reproduces to 0.1% while cohort-sensitive figures do not is itself the finding: it
localises the disagreement to a definitional choice rather than a computational
error.

---

## 5. Where the time actually goes

Two corrections were required before the real log gave a defensible answer. Each
is a trap the naive analysis falls into.

Correction 1, the episode cut. The top-ranked "bottleneck" was initially
`Release A → Return ER` at a median of 47 days: a discharged patient at home. That
is not hospital waiting time. Splitting each patient's trace into **episodes of
care** removed 296 gaps carrying 23,815 days. Only 2% of gaps were cut; they held
the majority of the elapsed time.

> *Lesson:* an event log records **elapsed** time. Only domain knowledge tells you
> which elapsed time the organisation is responsible for.

Correction 2, the ranking metric. The first ranking used `median × frequency`.
31% of all gaps are exactly zero, because lab results are entered in batches, which
drags the laboratory's median to zero, and anything multiplied by zero is zero.
The lab-to-lab transition scored 0.0 days in the ranking while actually
absorbing 2,711 days, 44.6% of all waiting time in the log. The metric silently
deleted the single largest block of time. Ranking now uses the actual sum.

The corrected picture distinguishes two questions that are not the same:

| | Step | Reading |
|---|---|---|
| **Largest absorber** | `CRP → Leucocytes` | 746.9 days across 1,445 occurrences, 12.3% of all waiting, but **median 0 min**. It dominates because it happens constantly, not because it is slow. |
| **Most disproportionate** | `Admission NC → Release A` | absorbs **7.4×** the time its share of transitions would give it (permutation test, p = 0.0025) |

These imply different interventions: the first is a batching and volume question,
the second a genuine constraint. Notably the second is the ward-to-discharge
transition, a discharge delay, which is the theme the Indian layer is built
around. A single ranked list cannot separate these; the system reports both.

**A process usually has more than one bottleneck, and reporting only the worst one
hides the rest.** Every step is tested against the same significance threshold, so
seven steps in this log absorb more time than their share of transitions accounts
for:

| Step | Disproportion | Total | *p* |
|---|---|---|---|
| `Admission NC → Release A` | **7.40×** | 379 days | 0.0025 |
| `CRP → CRP` | 4.53× | 630 days | 0.0025 |
| `Leucocytes → Release A` | 4.47× | 441 days | 0.0025 |
| `CRP → Release A` | 4.46× | 630 days | 0.0025 |
| `Leucocytes → Leucocytes` | 3.31× | 664 days | 0.015 |
| `Admission NC → CRP` | 3.20× | 517 days | 0.027 |
| `Admission NC → Leucocytes` | 3.11× | 557 days | 0.027 |

Read together these are not seven separate problems but two: three of the seven
end at `Release A`, patients waiting to be discharged, and the rest are
laboratory repeats and waits on laboratory results. That is a far more actionable
description than a single ranked row, and it is the same two themes the rest of
this report turns on.

The threshold is not per-step. Every step is compared against the distribution of
the *maximum* disproportion under permutation, which is the Westfall–Young max-T
procedure: under the null, the probability that *any* step clears the bar is 0.05,
the same rate a single-step test already carried. Reporting more bottlenecks
therefore costs no additional false positives, verified in §9.

At department level, 45% of waiting is internal to a department and **55% is
handoff** between departments, different management problems with different
owners. An activity-level ranking cannot tell them apart, which is the concrete
answer to *"why not just use a dashboard?"*

---

## 6. The Indian layer, and calibration discipline

The real log is Dutch and cannot answer the question this capstone is about:
insurance-driven discharge delay under IRDAI rules. A calibrated SimPy
simulator was therefore built, emitting the full `schedule`/`start`/`complete`
lifecycle so that queueing and processing *can* be separated, the thing the real
log provably cannot do.

A simulator running without crashing proves nothing about what it emitted, so
`src/08_validate_synthetic_log.py` checks the generated log against what it
claims to be before any analysis touches it: every activity instance has one
`schedule`, one `start` and one `complete`; those timestamps never go backwards;
each case runs registration to discharge; and `payment_mode` is constant within a
case, with cashless traces carrying authorisation activities and self-pay ones
billing, neither carrying the other's. §7 depends on that last check: the
comparison is only meaningful if the tag it splits by is sound.

Every parameter carries a `source:` field in `config/calibration.yaml`, so
*"where did this number come from?"* is answered by opening one file. The audit is
enforced and reported honestly:

| Provenance | Count |
|---|---|
| verified (primary source read) | **10** |
| assumption (stated, not sourced) | 9 |
| secondary (search-derived) | 1 |
| placeholder (deliberately unresolved) | 8 |
| **Total** | **28** |

Each of the 22 findings this produced is recorded in
[`outputs/14_evidence_package.md`](outputs/14_evidence_package.md) with the
parameters it depends on, its robustness status, the wording it licenses and the
caveat that must accompany it.

The regulatory anchors were read from the circular itself, not from secondary
reporting: IRDAI/HLT/CIR/PRO/84/5/2024, dated 29.05.2024, clause 15(b), a
cashless pre-authorisation decision within 1 hour; clause 16(a), final
discharge authorisation within 3 hours; clause 16(b), the insurer bearing
charges caused by breaching it. Secondary sources circulate a different circular
number that does not appear on the document; it is not cited here.

The unsourced parameter is never given a point value. The probability of a
document-query loop between hospital desk and TPA has no published figure, so every
result depending on it is reported as a sweep over its plausible range.

---

## 7. Headline finding: cashless versus self-pay discharge

Discharge cycle time, measured from *medically fit to leave* to *actually gone*,
across five random seeds at baseline staffing:

| Query-loop probability *q* | Cashless median | Self-pay median | Cashless breaching 3 h |
|---|---|---|---|
| 0.00 | 2.32 h | 0.79 h | **24.4%** |
| 0.15 | 2.51 h | 0.79 h | 35.2% |
| 0.30 | 2.85 h | 0.79 h | 46.5% |
| 0.45 | 3.46 h | 0.79 h | 55.9% |
| 0.60 | 8.36 h | 0.79 h | 68.3% |
| 0.75 | 15.27 h | 0.80 h | 82.3% |

Three things hold across the entire sweep, and are therefore **not artefacts of the
unsourced parameter**:

1. Cashless is slower at every value tested, from +1.5 h to +15.1 h.
2. **Even at q = 0, with a perfect insurer and no document queries at all,
   24.4% of cashless patients breach the 3-hour rule** (range 20.6–27.4% across
   five seeds). This is a floor, not an estimate.
3. Self-pay is flat at ~0.79 h regardless of *q*, which is the control that shows
   the effect is insurance-driven rather than a general capacity problem.

The median crosses the regulatory 3-hour line somewhere around q ≈ 0.30–0.45,
which varies by seed, reported as a range precisely because a single seed
suggested a sharper answer than the data supports.

At adequate staffing the delay is almost entirely processing, not queueing.
That distinction is the actionable part: the lever is removing steps from the
discharge path, not adding staff to it.

---

## 8. Does it work on a hospital nobody looked at?

Everything above was built while looking at one log. Code written that way encodes
the log it was written against, and **you cannot detect that by re-running it on
the same log**, it will keep passing. So a second log was held out entirely: the
Hospital Billing event log, 451,359 events across 100,000 cases, a different
hospital and a different process, untouched until the generalisation work.

The pipeline was made structural rather than lexical: episode boundaries,
department roles, resource semantics and cohort splits are now discovered from
whatever log arrives (`src/18_process_profile.py`), not named. On Sepsis the
discovered department roles independently reproduce the previously hand-typed map, the 18 identical wards collapse on their own, and improve on it by separating the
intensive-care wards, which the typed version wrongly merged.

Run once, unaided, the holdout exposed a genuine defect: the pipeline reported
100.0% of waiting as "handoff" and nothing warned.

| | Sepsis | Hospital Billing |
|---|---|---|
| resource column | `org:group`, **26 departments** | `org:resource`, **1,150 individuals** |
| consecutive events sharing an actor | **50.24%** | **0.01%** |
| column completeness | 100% | **55.2%** |

With 1,150 individuals, consecutive events are almost never the same person, so
everything registers as a handoff. The number was confident and meaningless. The
profiler now measures both properties and the analysis withholds the split
rather than printing it.

The deeper lesson is the second row of that table. The project's own notes
recorded, as an established fact, that `org:group` is 100% filled in the Sepsis
log. That was true, was never written into the code, and was relied upon anyway.
The holdout is 44.8% empty, and the analysis crashed on it. Hard-coded *names* are easy to find; hard-coded *properties* are
invisible, because nothing in the code names them.

Final holdout result: 18/18, including the sharpest assertion, *no
Sepsis-only activity name appears in any artifact built from the holdout*. That
assertion failed three times before passing, each time catching a different
hard-coded remnant.

---

## 9. Are the answers correct?

Accepting an unseen log is not the same as being right about it. There is no ground
truth for a real hospital, so it was manufactured: **27 randomised process
structures**, each with one defect injected and hidden, the pipeline run unchanged,
and the output compared against the buried answer.

Three metrics, deliberately never merged. Conflating them is how this kind of
result gets oversold. The full output is committed as
[`outputs/22_report.md`](outputs/22_report.md), with one row per scenario in
[`outputs/22_results.csv`](outputs/22_results.csv) giving the defect injected
beside what the pipeline said — so every figure below can be recounted rather
than taken on trust:

| Metric | Result |
|---|---|
| **Pipeline recovery**, did the deterministic analysis rank the injected defect first, before any model saw it? | **24/24 (100%)** |
| **Grounding validity**, did the output survive the guard? | 19/27 accepted, **8 abstentions** |
| **Diagnostic correctness**, did it name the real defect? | **19/19** of accepted diagnoses |
| **False positives**, did a healthy process get a diagnosis? | **0/3** controls |

Recovery held at 100% across all four defect types (delay, rework loop, routing,
handoff), both noise levels, all three process sizes, and both deceptive framings,
and at every effect size tested, down to a delay of half the background gap —
[`outputs/22_detection_floor.csv`](outputs/22_detection_floor.csv) has the
per-magnitude counts.

The deceptive cases were verified to actually deceive, since a trap that does not
trap proves nothing:

- Rare slow path: a decoy with median 219 h on 17 occurrences against the real
  defect's 26 h on 290. A median-ranked pipeline picks the decoy; a total-ranked
  one does not.
- Zero median: the real defect ranks 10th by `median × count` (it scores
  0.0) and 1st by total, at 24× the next edge. This is the §5 lab-batching
  failure reproduced deliberately, with the answer known in advance.

The eight abstentions split three ways. Four were the model reaching for
queueing-versus-processing language on `complete`-only data, which those logs
cannot support. One quoted a number that was not in its evidence. The remaining
three were the no-defect controls, examined in §10. In every case nothing was
written.

### A defect this harness could not have found

Every scenario above injects exactly one defect. That makes *"reports one
bottleneck"* and *"reports the bottleneck"* indistinguishable, and a later test
with two planted bottlenecks exposed what the difference was hiding: the
second was never reported, at any strength. On a 250-case log the secondary held
1,450 hours against the primary's 1,923, 75% of it, and was invisible.

The cause was not ranking, thresholds or the significance test, all of which
behaved correctly. Both findings were argmax reductions: the code computed a
disproportion ratio for every step and discarded all but the maximum, one line
after computing it. A secondary bottleneck was *structurally unreportable*.

The fix reuses the existing null rather than relaxing anything, see §5, and the
sweep was rerun to confirm it invents nothing:

| Variant | What it injects | Mean steps reported |
|---|---|---|
| `plain` | one defect only | **1.00** |
| `zero_median` | one defect only | **1.00** |
| `decoy_median` | defect **+ a rare slow path** | 1.33 |
| `competing` | defect **+ a second weaker anomaly** | **1.67** |

Every additional detection falls on a variant that deliberately contains a second
anomaly; the two single-defect variants report exactly one each. Recovery stayed
24/24 and false positives on the no-defect controls stayed 0/3.

The harness then gained the scenario it had been missing. A `two_defects`
variant plants a second bottleneck at 75% of the primary, the same ratio as the
log that exposed the fault, on an edge sharing no activity with the first.
Secondary recovery is scored in its own column, never folded into the headline,
because averaging the two would let *"found the primary, dropped the secondary"*
read as a partial pass. Current result: 5/5.

A regression test that passes proves nothing unless it would fail when the bug
returns, so that was checked by temporarily reinstating the fault:

| | Secondary recovery | The **old** headline metric |
|---|---|---|
| fault reinstated | **0/5** | 24/24 (100%), a clean pass |
| fault fixed | **5/5** | 24/24 (100%) |

The old metric is identical in both rows. Every figure this harness reported
before was blind to the defect, which is exactly how it survived so long; the new
column is the only thing distinguishing the two runs, and when it fails it names
the missed edges rather than moving a percentage.

The methodological point generalises beyond this project: **a validation suite
cannot find a failure its scenarios do not contain.** Twenty-seven randomised
processes, four defect types, two deceptive framings and six effect sizes all
missed this, because none of them had two of anything.

---

## 10. The limitation this surfaced, and the fix

**All three no-defect control processes produced a confident diagnosis naming a
bottleneck.** This was not a bug but a design gap: the pipeline reports the
*relative* worst step, and some step is always worst. **Ranking is not significance
testing.**

The obvious remedy, *"flag it if the top step is 3× the median"*, means choosing
a threshold that makes one's own controls pass, on data generated by a script one
also wrote. A permutation test removes the choice:

> **Null hypothesis:** waiting time is *exchangeable* across transitions. No step
> is special; a step that fires often simply accumulates more time.
>
> Shuffle which transition each observed gap belongs to, holding every step's
> count fixed, and recompute. Four hundred times. That is what the data would look
> like by luck alone.

| Population | Disproportion | Verdict |
|---|---|---|
| controls (no defect) | 1.02–1.05× | no signal; a "no bottleneck" prohibition is issued |
| injected defects | 2.19–12.97× | material, and it names the injected step |
| real Sepsis log | **7.4×, p = 0.0025** | material |

The first version of this test was wrong, and the real log caught it. It used
*largest total* as the statistic, but the largest total goes to the highest-count
step under both the observed data and the null, so it compared a frequent step with
itself and returned ≈1.0 by construction. On synthetic data it appeared perfect,
because the injected defects were both frequent *and* slow. Only the Sepsis log,
where those properties come apart, exposed it: the test declared a hospital with a
7.4× disproportionate discharge step to have no bottleneck at all.

> *Lesson:* a test that passes on data you generated can still be measuring the
> wrong quantity.

### Re-running the controls after the fix

All three controls are now found immaterial (*p* = 0.58, 1.00, 0.76), all three
receive the `X-NOBOTTLENECK` prohibition, and none produces an accepted
diagnosis: **0/3**, against 3/3 before.

The prohibition is not what stopped them. It matches phrases containing the word
*bottleneck*, and on all three controls the model avoided that word, writing
instead that waiting time was *concentrated in* a particular step — the claim
the prohibition exists to prevent. The diagnoses were rejected by a different
check: `probable_drivers` and `recommendations` came back empty, so the output
was incomplete.

The prohibition was not tightened afterwards. A guard adjusted until the results
improve measures the adjustment rather than the system, and that constraint was
set before the sweep ran. The behaviour is recorded here instead, as a concrete
case of the limit stated in §11: matching a phrasing is not the same as
constraining a claim.

---

## 10b. Making the interpretation useful

A guarded output that says nothing is of little use. Early recommendations read:

> *"Review and simplify the process for the 'CRP → Leucocytes' step."*

That is circular, it restates the question. But the cause was not model weakness.
The prompt asked for a fix while supplying no vocabulary of fixes, so with
measurements and no domain knowledge, "review and analyse X" is the only safe
thing a model can say.

Two changes followed, and the second is the more interesting one.

An intervention catalogue. Six mechanisms, batching, rare path, rework,
coordination gap, constraint, returns, each stating the evidence signature that
suggests it and the levers that follow. Plus a constraint that falls out of the
capability gate: *if the log cannot separate queueing from processing, you cannot
know whether more staff would help, so do not recommend capacity on the basis of
queueing.*

The signature classification moved into Python. The catalogue alone caused a
new fault: the model called a step with 1,445 occurrences a "rare slow path",
whose signature explicitly requires *few* occurrences. That was a design error,
not a model limitation. Deciding *"is this median near zero?"* or *"does this edge
cross departments?"* is arithmetic, and this project's central rule is that
arithmetic happens in code. Asking the model to evaluate a numeric condition had
put a computation back into the prompt, exactly what the architecture exists to
prevent. Each step is now classified deterministically and the finding states the
result, so the model selects a lever for a signature already decided.

The effect on the same log, with the same evidence:

| | Output |
|---|---|
| Before | "Review and simplify the process for 'CRP → Leucocytes'." |
| After | Three distinct problems: shift the batch release to continuous forwarding; assign a single owner per inter-department handoff; introduce upstream validation to cut repeat tests. |

Every claim cites finding ids, and the groupings verify: every finding the
model filed under "handoff" is classified `handoff`, and both filed under "rework"
are the two self-loops. There is no misgrouping, it was not asked to infer the
grouping, so it could not get it wrong.

It is not exhaustive, and the report should not pretend otherwise: five steps are
classified `handoff` and the model cited four of them. Omitting one is a
completeness gap, not an error, and the full list remains in the `materiality`
block of the findings file for anyone who wants it.

Model capacity is a separate axis from prompt design. A local 8-billion
parameter model reported only one of the three signatures present, across four
prompt revisions. A 120-billion parameter model covered all three. The catalogue
and the deterministic signatures improve *any* model's output; covering every
distinct signature turned out to need capacity, and a free hosted tier supplied it.

---

## 11. What the system guarantees, and what it does not

The guard is measured rather than asserted, by
`python src/15_narrate_findings.py --measure-guard`. Against the evidence payload
it rejects **95.0%** of randomly fabricated numbers (n=4,000) and **66.4%** of
near misses, defined as a real figure perturbed by ±1% (n=152). Both rates are
properties of the payload as much as of the check: a denser number space is
easier to land in by luck, so adding findings lowers them without the guard
changing. Both figures were previously quoted from a comment and had drifted out
of date, which is why they are now computed on demand. The categorical checks are
deterministic string matches and fire every time. Both guard suites pass their
self-tests (7/7 and 7/7).

A live local model run on the unseen holdout log was rejected for attempting
the internal-versus-handoff split, the exact claim that log cannot support. That
prohibition was never written by hand. The log's own properties produced it:

> holdout revealed a person-level column → profiler learned to detect it →
> extractor turned it into a prohibition → guard caught a real model making the claim

**The limit is architectural and is stated plainly because it is the honest answer
to the obvious question.** On a control process the model wrote:

> *"Time is lost in the transition between ACT00 and ACT01, which accounts for
> **57.7%** of all measured waiting time."*

The evidence says: *"The top 5 of 10 steps account for 57.7%."* The model took a
real number describing five steps and bound it to one transition. Every
component is in the evidence; the combination is false.

Numbers are checked against a set. Phrases are checked against patterns. **Neither
check examines the binding between a quantity and the entity it is predicated of.**
Catching that requires semantic parsing of the claim, not string matching.

> **The system guarantees provenance, not inference.** Every figure can be traced
> to the code that computed it. Whether the sentence around it reasons correctly is
> not something this mechanism can establish, and claiming otherwise would be
> precisely the overselling the project exists to avoid.

A related boundary: the guard proves output is *grounded*, not that it is
*insightful*. The local 8-billion-parameter model cleared every check while
producing near-circular reasoning and ignoring the most actionable finding
available to it. That is a statement about the model, not the architecture, the
same payload sent to a stronger model meets the same guard.

---

## 12. Business case

Converted to the unit an administrator manages, bed-hours, at the q = 0
floor, so the figure carries no dependence on the unsourced parameter:

- 1,588 excess bed-hours per 1,000 cashless admissions, against the self-pay
  counterfactual
- 134 bed-hours per 1,000 fall beyond the 3-hour regulatory deadline

Valued against a primary-source Indian benchmark, Global Health Ltd (Medanta),
ARPOB ₹67,361 per occupied bed-day, Q3 FY2026, from the BSE/NSE filing dated
2026-02-04, that is on the order of ₹44.6 lakh per 1,000 cashless admissions.

This figure is an illustrative external benchmark and nothing else. The
bed-hours are simulated; the ARPOB belongs to a different company in a different
country from the analysed log, which is Dutch and discloses none. It is an upper
bound on opportunity, not a cash saving, and the system permanently forbids
phrasing it as the study hospital's own loss, a prohibition the guard enforces
regardless of how carefully the surrounding paragraph is worded.

---

## 13. Conclusion

The system locates bottlenecks in hospital event data reliably, 24/24 on known
ground truth, across defect types, noise levels, deceptive framings and effect
sizes down to half the background gap, and it explains them in language an
administrator can act on, with every figure traceable to the code that produced it.

Its three validations answer three different questions, and the project's method
was to keep them apart: reproducing published figures says the arithmetic is right;
the holdout says the code is not secretly memorising one hospital; the ground-truth
harness says the answers are correct. None of the three substitutes for the others.

Nor is any of them sufficient. All three passed while the system was silently
reporting only one bottleneck per process, because no scenario in any of them
contained two, a defect found later by a single test that did (§9). Validation
bounds the failures you thought to look for; it does not certify the absence of
the ones you did not.

What it does not do is equally clear, and stated deliberately: it cannot
separate queueing from processing on a log without start timestamps; it cannot
measure department handoffs when the resource column names individuals; it could
not, until measured and fixed, say that a process is healthy; and it cannot verify
that a correctly-sourced number has been attached to the right thing.

Each of those limits is enforced in code rather than noted in prose. That is the
argument of this project: **the value of an AI layer over operational data is not
what it can say, but what it can be prevented from saying.**

---

## Appendix, reproducing this work

```bash
conda activate hospital
python src/00_download_data.py          # fetch and verify the Sepsis log
python src/03_department_handoffs.py    # where the time goes, by department
python src/04_validate_literature.py    # the published-figures gate
python src/20_extract_findings.py       # structured findings from any log
python src/21_diagnose.py --self-test   # diagnosis guard, no model needed
python src/15_narrate_findings.py --measure-guard   # guard strength, measured
python src/19_holdout_test.py           # generalisation: 18/18 on an unseen log
python src/22_ground_truth.py --no-model --tag demo  # ground-truth sweep
#   --tag leaves the committed results intact; add --api-key-env GROQ_API_KEY
#   to include the model layer
python -m streamlit run src/16_dashboard.py
```

Findings, validation reports and rejected model runs are written to `outputs/`.
Every figure quoted in this report is reproducible from the commands above;
the validation artifacts are committed under `outputs/`.
