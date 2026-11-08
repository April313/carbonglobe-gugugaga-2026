# What a Forest-Carbon Emulator Can and Cannot Learn from One Forcing Record: Lessons from the CarbonGlobe AI Emulation Challenge

**Team:** GuGuGaGa (single-member team)  
**Author:** Yangbojun Yang  
**Draft:** 2026-11-07. Section 7.6 awaits one running experiment (C3).

---

## Abstract

We took part in the IEEE BigData Cup 2026 AI Emulation Challenge, which asks for year-40 forest height and aboveground biomass (AGB) under two future emission scenarios, for 6,171 sites and 15 seed forest ages, emulating the Ecosystem Demography (ED) model. We report three kinds of results.

First, an input issue: the provided initial states appear to be an affine transform of physical values. Undoing it, with no change to the models, moved our public score from 0.143 to 0.127. A later split test supports the transform for height only weakly, and leaves the AGB scale unresolved.

Second, a model: gradient-boosted trees predicting the 40-year change from the initial states of all 15 seed ages at a site, trained on 3,373 sites, scored 0.091. A closed-form blend of earlier submissions fitted to public scores reached 0.076. That blend depends on the public leaderboard and may not generalize.

Third, and in our view most useful, a set of controlled experiments on historical data. A zero-parameter baseline that reads the answer off the site's own age curve transfers to the future scenarios better than any model we fitted. Models trained on early years of the historical run do not beat it on later years, and adding 15 times more training sites did not improve the public score. Abrupt biomass collapses, which carry most of the remaining error, are highly predictable in ranking but not in overall rate across periods; a model that separates collapse from growth extrapolates better in time within the historical run, yet scored worse on the public leaderboard. We argue that the main obstacle is that the training data contain one forcing record per site, so responses to scenario-level change cannot be identified offline.

---

## 1. Task

For each of 6,171 sites, two scenarios (SSP1-2.6, SSP5-8.5) and 15 seed ages (1 to 500 years), predict the December state of year 40 for canopy height and AGB. Inputs are the initial state (7 variables) at all 15 ages and 40 years of monthly forcing (136 drivers). The score is

`0.5 × (MSE_height + MSE_AGB)`,

with each variable divided by its global mean (10.90645 and 3.39567). The public leaderboard uses about 30% of the 185,130 rows; the final ranking uses the rest.

Training data come from the CarbonGlobe benchmark [1]: 54,152 sites, 40 years of historical forcing and calibrated ED outputs.

---

## 2. Data, validation and what we used

### 2.1 Site pools

| Pool | Sites | Used for |
|---|---:|---|
| Original `train` + `val` | 3,373 | S1 and all submitted single models before 2 Nov |
| All historical except protected | 49,204 | G1, G2 and all controlled experiments in Section 6 |
| Protected confirmation | 4,948 | Not used for training. 3,924 used once to describe G1 |
| Fresh confirmation (subset of protected) | 1,024 | Reserved as a final hold-out |

**Disclosure.** The 49,204-site pool includes sites of the original benchmark `test` split. The competition data page lists the whole CarbonGlobe dataset as training data and we found no rule excluding that split, but the organizers have not confirmed it. All results that use this pool are marked. The 1,024 fresh sites were loaded into memory once by a run that was stopped before any evaluation; nothing derived from them was computed or written.

### 2.2 Validation

- **Spatial (V1):** 5-fold grouped by 10° blocks. From 2 Nov we also drop training sites within 4 grid cells (Chebyshev, 0.5° cells) of any validation site. Without this buffer, dense pools leak through neighbouring cells.
- **Temporal (V2):** train on an early window of the historical run, evaluate on a later window, combined with the spatial folds.
- **Public leaderboard (V6):** from 5 Nov, each upload was a pre-registered hypothesis test with a decision rule fixed before upload. Two public scores are treated as different only if they differ by at least 0.002, because scores are rounded to three decimals.

One detail matters for interpreting spatial validation: the 3,373 original sites lie on a regular lattice about 4 cells apart. With 10° blocks, the median distance from a validation site to the nearest site in another fold is still 4 cells, the same as random site-wise folds. Block folds alone therefore do not make this pool harder.

---

## 3. The initial states are transformed

### 3.1 Evidence

| Check | Value |
|---|---:|
| Historical maximum height (model cap) | 35.00 m |
| SSP maximum height, both scenarios | 29.77 |
| Slope from 0 → −0.9009 and 35 → 29.77 | 0.8763 |
| Third anchor: minimum seedling height after inversion | 1.50 m (historical minimum 1.50 m) |

A shared vector appears in 8,177 rows, almost all at old ages, consistent with the image of a zero (dead) state. We therefore model `test_y0 = a · true + c`, with `c` the shared vector, `a = 0.876341` for height. For AGB we use `a = 0.876`; this value comes from a public notebook's leaderboard calibration [2], and our own allometric estimates ranged from 0.78 to 0.88.

Three early submissions of the form `y40 = α · y0` gave an optimum `α ≈ 1.114`. If the inputs were compressed by 0.876 and the answers were in physical units, the expected optimum is about 0.970/0.876 ≈ 1.107; if the answers were also transformed, about 0.97. This supported physical units for the answers.

None of this has been confirmed by the organizers.

### 3.2 Effect

Re-running the then-best recipe on inverted inputs (clipping height to 35 m) moved the public score from 0.143 to 0.127.

### 3.3 A later split test

On 5 Nov we uploaded the zero-parameter baseline of Section 5 three ways:

| File | Height | AGB | Public |
|---|---|---|---:|
| V6-3 | inverted | inverted | 0.127 |
| V6-4 | raw | raw | 0.095 |
| V6-5 | inverted | raw | 0.093 |

Because the score is the mean of two column MSEs, these three scores separate the columns. Inverting height lowered its MSE by about 0.004 (±0.002 from rounding); inverting AGB raised its MSE by about 0.068. The pre-registered rule counts this as support for the height inversion, but the score sits exactly on the threshold, so the support is weak.

The AGB result is confounded: on historical data this baseline already overestimates AGB by an amount close to the raw-minus-inverted difference (for example −0.229 against −0.228 at age 150, scaled). We cannot tell whether the AGB gap comes from the baseline's own bias or from a wrong AGB slope. **The AGB scale remains the main open question about the inputs.**

---

## 4. Submitted models

### 4.1 S1: change plus site age profile

| Item | Setting |
|---|---|
| Target | 40-year change `Δ = y40 − y0`, one model per variable |
| Features (46) | own state (3); height and AGB of all 15 seed ages at the site (30); distance to the site maximum (1); climate summaries (12) |
| Model | scikit-learn `HistGradientBoostingRegressor`, learning rate 0.05, 63 leaves, `min_samples_leaf` 50, `max_features` 0.8 |
| Iterations | height 1242, AGB 1175, median of fold optima |
| Ensemble | 3 seeds |
| Training sites | 3,373 |

The change target and the use of the site's age profile were suggested by a public notebook [2]; the code is our own.

Spatial 5-fold block OOF (5 seeds): height 0.0305, AGB 0.0643, combined 0.0474. Persistence scores 0.292. Public score: **0.091**.

### 4.2 Public-score blending (F2, F3, P1)

For a base prediction `b` and earlier submissions `p_i` with public scores `S_i`, write `d_i = p_i − b`, `G_ij = mean(d_i d_j)`, and `g_i = (S_i − S_b − G_ii)/2`. The public MSE of `b + Σ c_i d_i` is then a known quadratic in `c`, minimized by solving `G c = −g` with eigenvalue truncation (`tol = 1e-3`).

| Blend | Directions | Kept | Σ\|c\| | Predicted | Public |
|---|---:|---:|---:|---:|---:|
| F2 | 13 earlier submissions | 8 | 1.33 | 0.0815 | 0.082 |
| F3 | 17 (adds four new models) | 12 | 1.76 | 0.0763 | 0.077 |

P1 adds one fixed-shape direction to F3: height of tall stands lowered by 6% (weighted by a logistic function of initial height around 20 m). Public score **0.076**. A second direction (young-stand growth +14%) scored 0.078; their joint optimum was predicted at 0.0757, above a threshold fixed in advance, so no further blend was produced.

**These blends fit weights to the public leaderboard.** They exclude all components trained on original-test labels, but they are by construction tuned to about 30% of the test rows and may not hold on the private part.

### 4.3 Public scores

| Submission | ID | Public | Fitted to public scores | Training sites |
|---|---|---:|---|---:|
| Earlier best (includes one component trained on original-test labels) | 57144224 | 0.143 | no | see note |
| A2: earlier recipe on inverted inputs | 57859955 | 0.127 | no | see note |
| **S1** | 57860559 | **0.091** | no | 3,373 |
| F2 | 57861338 | 0.082 | yes | 3,373 |
| F3 | 57891074 | 0.077 | yes | 3,373 |
| **P1** | 57896151 | **0.076** | yes | 3,373 |
| G1: as S1, 49,204 sites | 58007331 | 0.090 | no | 49,204 |
| G2: as G1, no climate | 58007424 | 0.122 | no | 49,204 |
| S1-type, inner-CV iterations | 58132734 | 0.090 | no | 49,204 |
| C3 collapse mixture | 58132761 | 0.096 | no | 49,204 |

Note: one component of the earlier best and of A2 was trained on 20,480 sites, 17,107 of them outside the original `train`/`val` split.

[Final selection: to be confirmed by the team.]

---

## 5. A zero-parameter baseline

For a row with seed age `a`, the site provides its state at all 15 seed ages. A stand starting at age `a` reaches age `a + 40` after 40 years, so we linearly interpolate the site's initial-state curve at `a + 40` (capped at age 500). We call this `B`. It has no fitted parameters.

On historical data `B` scores 0.176, against 0.292 for persistence and 0.047 for S1. It is very accurate for young stands (age 1: height 0.027, AGB 0.044, against 0.93 and 0.90 for persistence) and no better than persistence for old stands.

On the public leaderboard, inverted-input `B` scored 0.127. Its public-to-historical ratio is about 0.72; for S1 the ratio is about 1.9. Unfitted predictors lose less, or even gain, when moved to the future scenarios; fitted models lose more. An independent entry reported the same pattern across several systems [3].

---

## 6. Controlled experiments on historical data

All experiments in this section use the 49,204-site pool (Section 2.1) and spatial blocks with a 4-cell buffer. Each had a decision rule written before it ran.

### 6.1 Where S1's error is

We flagged a row as a collapse (C20) if the scaled variable drops by at least 0.2 in a single year within the 40-year window. On the 3,373-site OOF of S1:

| Variable | Collapse rows | Share of S1 squared error |
|---|---:|---:|
| Height | 6.8% | 31.7% |
| AGB | 12.5% | 70.3% |

Combined share 57.9% (block bootstrap 95% interval 49.4%–64.8%). For AGB at ages ≥ 150, 18% of rows hold 75% of the error. Height collapses rarely recover within 15 years and are not synchronous across seed ages of the same site.

### 6.2 An early-run transient

Using `B` with windows of equal length but different start years:

| Window length | Start year 0 | Start year 10 | Start year 20 |
|---:|---:|---:|---:|
| 10 years | 0.0624 | 0.0235 | 0.0197 |
| 20 years | 0.1126 | 0.0511 | 0.0408 |

The first decade of the historical run is two to three times harder to extrapolate than later ones.

### 6.3 Baseline plus residual (C1): rejected

We trained the same trees on `y40 − B` instead of `Δ`. For ages ≤ 50, under spatial validation C1 (0.0266) matched the no-climate change model (0.0264) and lost to S1-type with climate (0.0164). The pre-registered rule required beating both, so the hypothesis was rejected. Our rule mixed two factors (target and climate features), which we note as a design flaw; we kept the verdict.

### 6.4 Temporal extrapolation

Train on an early window, evaluate on years 25–40, all windows 15 years, iterations chosen inside the training window only:

| Training window | S1-type | No-climate |
|---|---:|---:|
| 0 → 15 | 0.0513 | 0.1296 |
| 5 → 20 | 0.0340 | 0.0485 |
| 10 → 25 | 0.0323 | 0.0440 |
| 25 → 40 (same window) | 0.0192 | 0.0259 |
| `B`, no training | 0.0310 | |

- The best temporally extrapolated model ties `B` (difference +0.0013, interval −0.0061 to +0.0076). It does not beat it.
- Most of the gap appears when the first five years are included in training.
- The same models learn the late window well when trained on it, so the issue is not capacity.
- Replacing each model's mean change per age with the true mean of the evaluation window (a diagnostic using evaluation labels) only reduced the 0→15 score from 0.0513 to 0.0445. The windows differ in how sites relate to each other, not just in their average drift.

An earlier version of this experiment, with iterations chosen on the evaluation window, found that almost every model was best at the first checkpoint (25 iterations): the more it learned from years 0–20, the worse it did on years 20–40.

### 6.5 More training sites

G1 (S1's recipe on 49,204 sites) scored 0.0145 on 3,924 held-out sites, against 0.0331 for S1. Those held-out sites were not spatially buffered, and 99.95% of them were within one cell of a G1 training site, so part of this gain is interpolation. On the public leaderboard G1 scored 0.090 against S1's 0.091, which our pre-registered rule counts as indistinguishable. The no-climate version scored 0.122 against 0.103 for its 3,373-site counterpart.

### 6.6 Collapses are predictable in rank

We classified AGB collapse (C20) within 40 years:

| Model | AUC, all | AUC, age ≥ 150 | Brier skill vs age-only |
|---|---:|---:|---:|
| Age only | 0.663 | 0.516 | 0 |
| Age + initial state, logistic | 0.835 | — | 0.087 |
| Trees, no climate | 0.961 | 0.957 | 0.518 |
| Trees, with climate | 0.974 | 0.972 | 0.605 |

Within each age group the trees still separate collapsing from non-collapsing rows (AUC 0.95–0.97). Under temporal extrapolation (train 0–20, evaluate 20–40), the climate model keeps AUC 0.948, while the no-climate model drops to 0.818 at ages ≥ 150.

The overall rate is not stable: AGB collapses fall from 9.6% of rows in years 0–20 to 6.3% in years 20–40, and models trained on the early window predict 4.6% to 10.2% depending on the model. **Who collapses transfers across periods; how many collapse does not.**

For AGB collapse rows, the final value is about 0.70 of the initial value (median).

### 6.7 Collapse mixture model (C3)

We predicted, for each variable, the probability `p` of a collapse within the window and two conditional changes, `μ0` for non-collapsing rows and `μ1` for collapsing rows, each with the same trees and 46 features as S1. The prediction is `y0 + (1 − p)·μ0 + p·μ1`. Iterations were chosen inside the training sites and window only.

Ages ≥ 100, both variables:

| Validation | S1-type | C3 | Difference (95% interval) | `B` |
|---|---:|---:|---|---:|
| Spatial (V1) | 0.0537 | 0.0531 | −0.0006 (−0.0023 to +0.0010) | 0.2464 |
| Temporal (V2) | 0.0704 | 0.0577 | −0.0127 (−0.0185 to −0.0074) | 0.0501 |

The pre-registered rule (C3 below S1-type in both) counts this as support. Spatially the two tie; temporally C3 is about 18% better, consistent with the finding that collapse ranking transfers across periods.

C3 still does not beat the zero-parameter baseline under temporal extrapolation (all ages: 0.0470 against 0.0385 for `B`). Replacing `p` with the true collapse labels gives 0.0406, still above `B`. The remaining gap therefore lies in the conditional end states, not in knowing who collapses.

Scaling `p` down by 25% improved the temporal score (0.0460), consistent with the lower collapse rate in later years. We report this as a sensitivity only; it was not used to choose anything.

**SSP test (pre-registered).** We retrained both models on all 49,204 sites with identical settings and uploaded them once each. S1-type scored 0.090 and C3 scored 0.096. The pre-registered rule counts this as no support: the temporal advantage seen inside the historical run did not carry over to the future scenarios, and C3 was worse. Its predictions were higher on average (height +0.036, AGB +0.016, scaled).

## 7. Discussion

**One forcing record per site.** Each training site has a single 40-year historical forcing. There is no contrast between scenarios anywhere in the training data, so the response to scenario-level change is not identifiable offline. Climate levels are partly covered by other sites: about 72%–74% of SSP sites have an analogue among the 3,373 original historical sites, within the 95th percentile of historical nearest-neighbour distances, in the last decade's climate (inputs only). CO₂-like drivers, however, take the same value at all sites in a given year and are collinear with calendar time and with the early-run transient. Better models cannot recover what the data do not vary.

**Fitting harder transfers worse.** The pattern is consistent across our experiments: the unfitted baseline transfers best, fitted corrections transfer at best neutrally in time (Section 6.4), and more sites help locally but not on the public leaderboard (Section 6.5). We read this as a property of the data, not of tree models in particular, though we have not tested sequence models under the same protocol in this round.

**What does transfer.** The ranking of collapse risk survives a change of period, and climate features help it survive. This suggests separating "which sites" from "how many", and treating the overall collapse rate under a new scenario as an explicit unknown rather than something learned from history.

**Local validation does not rank models for the test set.** We confirm this for our systems and add one mechanism: temporal non-stationarity inside the historical run.

---

## 8. Limitations

- The input transform, the answer units and the AGB slope are inferred, not confirmed. The AGB scale is unresolved (Section 3.3).
- Blends F2, F3 and P1 are fitted to public scores.
- Results in Section 6 use original-test labels whose use has not been confirmed.
- Climate-column meanings come from a public notebook.
- ED is a simulator. Our statements concern the emulation of ED, not real forests.
- All temporal tests are inside the historical run; none uses scenario data with labels.
- Section 6.4 confounds "excluding early years" with "being closer in time to the evaluation window".

**Procedure changed after seeing results.** Two acceptance criteria were changed after results were seen: (1) a replay check was re-run with the same batch geometry as the full run, keeping the 5e-5 threshold, after we found that small-batch differences predated the input fix; (2) the blend's clipping stop condition was changed from "more than 2% of rows clipped" to "clipping energy ≤ 1e-4", because the original rule failed even for S1, whose AGB is zero in 12% of rows.

---

## 9. Reproducibility

- Environment: Python 3.14, scikit-learn 1.9.1, numpy 2.3.5, scipy 1.18.1, pandas 3.0.1, joblib 1.6.0. CPU only.
- S1 reproduces from the official data in about 2.5 minutes; we re-ran the package in a clean directory and matched the submitted file's SHA256 (`23216519…0824aba`).
- F2 reproduces from the 13 earlier submission files and their public scores; the SHA256 matched (`43b2d88b…152dab1`). F3 and P1 follow the same procedure with the inputs listed in Section 4.2.
- Experiments in Section 6 store code, out-of-fold predictions and SHA256 records per run.

[Package location and licence: to be decided by the team.]

---

## Use of AI tools

The author is solely responsible for this report and all submissions. Following common publication policies, AI tools are not listed as authors; their use is disclosed here.

| Tool | Role |
|---|---|
| Claude (Anthropic), via Claude Code | Reviewing results, designing experiments and pre-registered decision rules, read-only data checks, and drafting this report |
| GPT models (OpenAI), via Codex | Implementing scripts, running training jobs and performing the leaderboard uploads |
| Gemini models (Google), via Antigravity | Implementing and running most of the training and validation experiments |
| DeepSeek models, via DeepSeek Harness | Discussing method choices in October 2026 |

All AI-produced code, numbers and text were checked against saved run records before use. Section 6 results were independently recomputed from saved predictions where noted in the project logs. Remaining errors are the author's.

## Acknowledgements and sources

We thank the organizers and the CarbonGlobe authors [1]. We used ideas, not code, from two public works: the notebook by Lucas Bruzzone [2] (Apache 2.0), for the input transform form, the AGB slope and the change-plus-profile design; and the entry by Francisco Angulo de Lafuente [3] (MIT), whose measurements on transfer ratios prompted Sections 5 and 6.4.

## References

[1] Z. Wang et al., "CarbonGlobe: A Global-Scale, Multi-Decade Dataset and Benchmark for Carbon Forecasting in Forest Ecosystems," NeurIPS 2025.  
[2] L. Bruzzone, "CarbonGlobe: from 0.142 to 0.060," Kaggle notebook, 2026.  
[3] F. Angulo de Lafuente, "carbonglobe-emulator-2026," GitHub repository and technical report, 2026.
