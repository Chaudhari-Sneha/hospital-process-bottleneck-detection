# Step 22 - Ground-truth validation

_27 randomised process structures, each with one known injected defect (or none). The pipeline was run unchanged; the guard was not modified._

## 1. Pipeline recovery (deterministic, before any model)

**24/24 (100%)** of injected defects were ranked first by the analysis itself.

| variant | recovered | n |
|---|---|---|
| competing | 5 | 5 |
| decoy_median | 5 | 5 |
| plain | 4 | 4 |
| two_defects | 5 | 5 |
| zero_median | 5 | 5 |

| defect type | recovered | n |
|---|---|---|
| edge_delay | 6 | 6 |
| handoff | 6 | 6 |
| routing | 6 | 6 |
| self_loop | 6 | 6 |

### Secondary bottlenecks

**5/5** scenarios with a SECOND planted bottleneck had it reported as well as the first.

This variant exists because every other scenario injects exactly one defect, which makes *reporting one bottleneck* and *reporting the bottleneck* score identically. The pipeline reported only its argmax for a long time and passed everything above while doing it.

## 2. Grounding validity (is the output verifiable?)

**19/27** diagnoses passed the guard. **8** were rejected and nothing was written.

| scenario | violation |
|---|---|
| c1 | missing_sections |
| c2 | missing_sections |
| c3 | missing_sections |
| s05 | forbidden:X-LIFECYCLE |
| s07 | forbidden:X-LIFECYCLE |
| s12 | forbidden:X-LIFECYCLE |
| s15 | unsupported_numbers |
| s19 | forbidden:X-LIFECYCLE |

A rejection is an ABSTENTION, not a wrong answer. It is counted separately from correctness throughout.

## 3. Diagnostic correctness (did it name the real defect?)

**Read this metric carefully.** By design, the model does not search for the bottleneck - the deterministic pipeline ranks it and hands it over as finding `T1`. So this measures whether the model pointed at the right finding among the several it was given, not whether it located the bottleneck itself. Finding it is the pipeline's job (section 1); that separation is the architecture, not an accident. A high score here is evidence the model does not wander off the evidence - not evidence that it is analytically clever.

Of the **19** accepted diagnoses on defective processes, **19 (100%)** named the injected defect.

## 4. False positives (controls with NO defect)

3 control processes had no defect injected. **0** still produced an accepted diagnosis naming a bottleneck.

The permutation test found no material bottleneck in **3/3** of them (p = 0.5835, 1, 0.7581), and the capability gate emitted the `X-NOBOTTLENECK` prohibition on **3/3**. The analysis therefore distinguishes a healthy process from a defective one, which it could not do when this sweep was first run.

The prohibition is not what stopped the output. It matches phrases containing the word *bottleneck*, and the model avoided that word while writing that waiting time was *concentrated in* a particular step - the claim the prohibition exists to prevent. The diagnoses were rejected by a separate check, for returning empty `probable_drivers` and `recommendations`. Matching a phrasing is not the same as constraining a claim; §11 of REPORT.md states that limit in full.

## 5. Ground truth was hidden

27/27 scenarios verified to contain no ground-truth string in the findings handed to the model.
