# GameFi: Opportunities, Challenges and Prospects

**A UTAUT2-based study of technology acceptance in GameFi platforms — N = 516**

MSc dissertation research · University of Western Macedonia · School of Economic Sciences ·
Department of Management Science and Technology · MSc *"Electronic Business and Digital Marketing"* · Kozani, 2025

**Author:** Georgios Dikos · **Supervisor:** Prof. Ioannis Antoniadis

---

## What this repository is

The complete quantitative pipeline behind the dissertation: the survey dataset, every Python
script used to analyse it, and the raw console output of each run. Nothing is summarised away —
if a number appears in the thesis, the script that produced it and the output it printed are
both here.

The study extends **UTAUT2** (Venkatesh, Thong & Xu, 2012) with four GameFi-specific constructs —
Economic Motivation, Risk Perception, Trust in Technology, and Regulatory & Compliance Risks —
and tests the resulting model on 516 GameFi users, predominantly from Southeast Asia and
Latin America.

> **A note on the folder layout.** Everything lives under `Users/Velze/MSC UTAUT2/`. That path is
> a leftover from the original local project directory. It is kept deliberately: the dissertation
> text contains permanent hyperlinks into these exact paths, and renaming them would break every
> citation in the submitted document. Use the navigation table below rather than the folder tree.

---

## Navigation

| Section | What's in it |
|---|---|
| **[Questionnaire (PDF)](Users/Velze/MSC%20UTAUT2/questionnaire%20GameFi_%20Opportunities,%20Challenges%20and%20Prospects.%20-%20Google%20Forms.pdf)** | The full survey instrument as administered |
| **[Reliability — Cronbach's Alpha](Users/Velze/MSC%20UTAUT2/Cronbach's%20Alpha)** | α for all 13 scales, item-total correlations, α-if-deleted, per-construct data files |
| **[Factor Analysis](Users/Velze/MSC%20UTAUT2/Factor%20Analysis)** | KMO, Bartlett's test, EFA, scree plot, loadings heatmap |
| **[PLS-SEM Analysis](Users/Velze/MSC%20UTAUT2/PLS-SEM%20Analysis%20test%202)** | Data cleaning, measurement model, structural models v1/v2a/v2b, moderation analyses |
| **[Cluster Analysis](Users/Velze/MSC%20UTAUT2/Cluster%20Analysis)** | k-means segmentation, elbow and silhouette diagnostics, cluster profiles |
| **[Cluster Analysis — Thesis version](Users/Velze/MSC%20UTAUT2/Thesis%20version)** | The k-means run reported in the dissertation (Ch. 4.3). See *Two clustering solutions* below |
| **[Demographics](Users/Velze/MSC%20UTAUT2/14-Demographic)** | Age, gender, education, income, experience, region — charts and scripts |
| **Constructs 1–13** | Folders `1-Performance-Expectancy-(PE)` through `13.-Use-Behavior-(UB)` — per-question distributions, charts and item wording |

The cleaned dataset is at
**[`PLS-SEM Analysis test 2/utaut2_cleaned_data.xlsx`](Users/Velze/MSC%20UTAUT2/PLS-SEM%20Analysis%20test%202/utaut2_cleaned_data.xlsx)**
(516 × 50). The raw export is `dataset-utaut2.xlsx` in the same folder.

---

## Key findings

| | |
|---|---|
| **Sample** | 516 valid responses (534-row export; the final 18 rows are frequency tallies, not respondents) |
| **Model** | 12 latent constructs · 42 Likert indicators · 17 structural paths tested |
| **R² Behavioural Intention** | **0.712** (adj. 0.706) |
| **R² Use Behaviour** | 0.037 |
| **Strongest predictor of BI** | **Habit** (β = +0.355, f² = 0.216) |
| **Then** | Price Value (β = +0.249) · Social Influence (β = +0.150) |
| **Measurement quality** | α ≥ 0.80 on 12 of 13 scales · KMO = 0.949 · Bartlett's p < 0.001 |

### Three results worth arguing about

**1. Habit beats economics.** Economic Motivation records the *highest mean of any construct*
(M = 4.13) yet contributes essentially nothing once the other predictors are in the model
(β = −0.006, ns). Users say earning is what matters; their intention is actually driven by
routine and perceived value for money. Wanting to earn is near-universal in this sample, so it
does not discriminate between who intends to keep playing and who does not.

> **A note on the dissertation abstract.** The abstract describes Economic Motivation as the
> strongest predictor of intention and use. That is a description of the *descriptive* result:
> EM is the highest-rated construct in the sample (M = 4.13, above all eleven others). It is not
> a statement about predictive weight. In the structural model reported in Chapter 4.4 — and in
> `path_coefficients.xlsx` in this repository — EM → BI is β = −0.006 (ns), and the strongest
> predictor of intention is Habit at β = +0.355. EM is not the strongest predictor of use either
> (r = +0.15 with usage frequency, against +0.24 for Behavioural Intention and +0.23 for Habit).
> Where the abstract and the structural model differ, **Chapter 4.4 and the outputs in this
> repository are the authoritative account.**

**2. Trust runs the wrong way — and that is the interesting part.** Trust in Technology has a
*negative* direct effect on intention (β = −0.115, p < 0.01) and Risk Perception a *positive* one
(β = +0.098). Both contradict the standard hypotheses. The moderation analysis explains why:

> **Trust × (Risk Perception → Behavioural Intention): ΔR² = 0.042, t(512) = −6.14, p < 0.001**
>
> Low-trust users: risk → intention slope = **+0.659**
> High-trust users: risk → intention slope = **+0.280**

Risk *raises* intention, and trust dampens that effect. This fits a speculative-participation
reading rather than a safety-seeking one: in this population, perceived risk appears to function
partly as perceived upside.

**3. Experience flattens the learning curve, and little else.** Of five moderations tested by
GameFi experience, only one holds: Experience × (Effort Expectancy → BI), β = −0.325, ΔR² = 0.034.
Ease of use matters to newcomers and stops mattering once people are regular users. Social
influence, facilitating conditions and habit effects do *not* vary by experience level.

---

## The dataset

Fully anonymous. No names, emails, IP addresses, wallet addresses, timestamps or free-text
responses are included — those fields were stripped at export. Collected by online questionnaire
under GDPR-compliant consent.

**42 Likert items (1 = Strongly Disagree … 5 = Strongly Agree) across 12 constructs:**

| Construct | Code | Items | Cronbach's α | Mean (SD) |
|---|---|---|---|---|
| Performance Expectancy | PE | 5 | 0.930 | 3.92 (0.83) |
| Effort Expectancy | EE | 4 | 0.886 | 3.38 (1.07) |
| Social Influence | SI | 3 | 0.804 | 3.69 (0.91) |
| Facilitating Conditions | FC | 4 | 0.836 | 4.03 (0.85) |
| Hedonic Motivation | HM | 4 | 0.890 | 3.86 (0.87) |
| Price Value | PV | 3 | 0.887 | 4.02 (0.87) |
| Habit | HB | 4 | 0.924 | 3.92 (0.95) |
| Behavioural Intention | BI | 3 | 0.912 | 4.09 (0.95) |
| Economic Motivation † | EM | 3 | 0.882 | 4.13 (0.94) |
| Risk Perception † | RP | 4 | 0.907 | 4.02 (0.91) |
| Trust in Technology † | TT | 3 | 0.809 | 3.43 (1.19) |
| Regulatory & Compliance Risks † | RC | 2 | 0.895 | 3.31 (1.25) |

† GameFi-specific extension to standard UTAUT2.

Plus **Use Behaviour** (UB — usage frequency and weekly hours, 2 items, α = 0.666) and six
demographic variables.

**Sample composition.** 89.9% are current GameFi users, 75.2% of them weekly or more. 91.7% are
aged 18–34 and 76.6% earn under $1,000/month. 44.2% are from Southeast Asia and 19.2% from Latin
America and the Caribbean. 75.2% male.

This is a self-selected sample of *active users in emerging markets*, not a general population.
Every finding above should be read inside that frame.

---

## Reproducing the analysis

```bash
git clone https://github.com/Jojohorororo/MSC-UTAUT2-Research-GameFi.git
cd "MSC-UTAUT2-Research-GameFi/Users/Velze/MSC UTAUT2"
pip install pandas numpy openpyxl scipy statsmodels scikit-learn factor-analyzer matplotlib seaborn
```

Each analysis folder is self-contained — it carries its own copy of the data file its scripts
read, so `cd` into a folder and run. Suggested order:

```bash
cd "PLS-SEM Analysis test 2"  && python test.py           # cleaning: 534 → 516 rows
                                 python part2.py          # measurement + structural model v1
                                 python v2a_analysis.py   # v2a: adds RC → BI
                                 python v2b_analysis.py   # v2b: GameFi inter-construct paths
                                 python final.py          # experience moderations
                                 python tt_rp_moderation.py
cd "../Factor Analysis"       && python test1.py
cd "../Cluster Analysis"      && python "cluster analysis.py"
cd "../Cronbach's Alpha"      && python "1.Cronbach's Alpha Performance-expectancy-(PE).py"
```

Every script writes its results next to itself. The `.txt` files beside each script are the
console output from the original runs, kept for comparison.

Tested on Python 3.10 and 3.11. The headline figures above were independently re-derived from
`utaut2_cleaned_data.xlsx`, and the reliability and cluster scripts re-run from a clean clone
reproduce their committed outputs exactly.

**One compatibility note:** the 13 scripts in `Cronbach's Alpha` contain a backslash inside an
f-string expression, which is a syntax error before Python 3.12. On older versions, either run
them on 3.12+ or change the line

```python
print(f"\n{'Item':<6} {'Corrected Item-Total':<20} {'Cronbach\'s α if':<15}")
```

to

```python
print("\n{:<6} {:<20} {:<15}".format('Item', 'Corrected Item-Total', "Cronbach's α if"))
```

Everything else runs on 3.10+ unchanged.

---

## Methodological notes and limitations

Stated plainly, because they affect how the results should be read:

- **PLS-SEM is implemented manually** — standardised OLS regression per structural equation, not
  SmartPLS or a dedicated PLS library. Significance is assessed analytically rather than by
  bootstrapping. Path coefficients are reliable; confidence intervals for indirect effects would
  need a bootstrap of 5,000+ resamples.
- **AVE falls below the 0.50 threshold for SI (0.447) and FC (0.467).** Convergent validity for
  these two constructs is marginal and their coefficients deserve more caution than the rest.
- **Use Behaviour is weakly measured.** Two self-reported items, α = 0.666, R² = 0.037. The
  behavioural half of the model is the weakest part of the study: intention is well explained,
  actual use is not.
- **The cluster solution is exploratory.** k = 4 was chosen for interpretability as much as fit —
  the silhouette optimum is k = 2, and at k = 4 the score is 0.271. All 12 constructs differ
  across the four groups at p < 0.001, but they are soft regions rather than sharply separated
  segments. See *Two clustering solutions* below.
- **Cross-sectional and self-selected.** No causal claims are made or supported.

---

## Two clustering solutions

This repository contains **two** k-means runs on the same data, in two folders. This is
deliberate, and worth understanding before citing either.

| | `Thesis version/` | `Cluster Analysis/` |
|---|---|---|
| Preprocessing | raw 1–5 construct means | z-scored constructs |
| Risk-Aware Skeptics | **n = 67** (13.0%) | n = 243 (47.1%) |
| Disengaged Users | **n = 81** (15.7%) | n = 22 (4.3%) |
| Pragmatic Adopters | **n = 192** (37.2%) | n = 171 (33.1%) |
| Confident Enthusiasts | **n = 176** (34.1%) | n = 80 (15.5%) |
| Silhouette (k = 4) | **0.271** | 0.228 |
| Reported in the dissertation | **yes** (Ch. 4.3) | no |

**The `Thesis version/` run is the one reported in the dissertation.** The two scripts are
identical apart from a single step: whether the twelve construct scores are z-scored before
k-means.

### Why that one step changes the answer

The constructs do not have equal variance on the 1–5 scale. Regulatory & Compliance Risks
(SD = 1.25) and Trust in Technology (SD = 1.19) spread respondents out considerably more than
Performance Expectancy (SD = 0.83) or Facilitating Conditions (SD = 0.85) do. Because k-means
minimises squared Euclidean distance, it implicitly weights each dimension by its variance — so
in the raw space, trust and regulatory concern drive the partition.

Z-scoring removes that weighting and forces every construct to count equally. The effect is
measurable in the results: Trust in Technology goes from being the **strongest** separator
between clusters (a 3.06-point spread from 1.40 to 4.47 on the 1–5 scale) to the **weakest**
(2.13). Once trust stops differentiating, the solution collapses into a general
high / medium / low agreement gradient, and the four segments are distinguished mainly by how
positively respondents answered overall.

Both are legitimate preprocessing choices, and neither is a mistake. Which one is appropriate
depends on the question. This study asks whether GameFi users divide into groups with
*qualitatively different adoption logics* — so the informative solution is the one that lets
naturally high-variance constructs separate people, rather than the one that flattens them into
a single satisfaction dimension.

### Why the raw-space solution is the one that answers the research question

Its distinguishing segment is **Risk-Aware Skeptics (n = 67)**: Trust in Technology = **1.40**,
Risk Perception = 4.73, Behavioural Intention = **4.83**. These are people with almost no trust
in the technology, high awareness of its risks, and the *highest* intention to keep using it in
the entire sample.

That group is the empirical anchor for the study's central structural finding — the negative
TT → BI path and the significant Trust × Risk interaction reported in Chapter 4.4. In the
z-scored solution this segment does not appear at all; its trust signal is averaged away, and
no cluster has a trust score below 1.52 or above 3.64. The raw-space solution also separates
better on silhouette (0.271 vs 0.228).

### Reproducing each

```bash
cd "Users/Velze/MSC UTAUT2/Thesis version"
python "cluster analysis (thesis version).py"      # the dissertation's solution

cd "../Cluster Analysis"
python "cluster analysis.py"                        # the z-scored variant
```

### Two caveats, stated plainly

- **Cluster sizes drift slightly.** The dissertation reports 67 / 81 / 182 / 186; re-running the
  script today gives 67 / 81 / 192 / 176. The two distinctive segments reproduce exactly, in size
  and in all twelve construct means to two decimal places. The two larger, more similar groups
  exchange about six cases — expected behaviour for k-means on weakly separated clusters, and
  sensitive to the scikit-learn version.
- **The F-statistics in the dissertation's Σχήμα 18 do not reproduce.** The table reports F values
  between 38.67 and 72.45; this script produces 92–447. Every construct is significant at
  p < 0.001 in both, so the substantive conclusion — that the four groups differ on all twelve
  constructs — holds either way, but the exact F values in that table should not be relied on.
  Use `Thesis version/cluster_analysis_results.xlsx` (sheet `ANOVA_Results`) for the figures this
  code actually produces.

---

## Use, reuse and contact

**You are welcome to use this research.** Reuse the dataset, the scripts, the figures or the
findings for your own work — academic, commercial or otherwise. No permission needed.
Attribution is appreciated:

> Dikos, G. (2025). *GameFi: Opportunities, Challenges and Prospects.* MSc dissertation,
> University of Western Macedonia, Department of Management Science and Technology.

The full dissertation (125 pages, Greek with English abstract) is in this repository:
**[GameFi - Opportunities, Challenges and Prospects.pdf](GameFi%20-%20Opportunities,%20Challenges%20and%20Prospects.pdf)**

If you have questions about the methodology, or would like to discuss the findings, reach out
on LinkedIn:

**→ [linkedin.com/in/george-dikos-6b371a287](https://www.linkedin.com/in/george-dikos-6b371a287)**

I am glad to hear from researchers building on this, and from anyone who finds something in the
data I missed.

---

## Licence

Released under [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE) - free to
share and adapt, including commercially, with attribution.
