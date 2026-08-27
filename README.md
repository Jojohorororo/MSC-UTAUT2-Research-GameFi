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
- **The cluster solution is exploratory, and its labels are unreliable.** k = 4 was chosen for
  interpretability, not fit — the silhouette score at k = 4 is 0.228, well under the 0.5
  separation threshold the script itself plots, and the silhouette optimum is k = 2. The four
  groups do differ on all 12 constructs at p < 0.001, but they are soft regions rather than clean
  segments. Separately, the descriptive names assigned in `cluster analysis.py` come from a
  rule-based fallback and do not all match their profiles — Cluster 2 (n = 243) is labelled
  *Risk-Aware Skeptics* while actually scoring highest on nearly every construct. **Read the
  numbers in `cluster_analysis_results.xlsx` (sheet `Cluster_Means`), not the labels.**
- **Cross-sectional and self-selected.** No causal claims are made or supported.

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
