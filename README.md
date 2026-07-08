# 🏛️ Boston Housing — Statistical Hypothesis Testing

## 1. Project Overview

This project applies classical statistical hypothesis testing to the Boston Housing dataset, moving beyond simple descriptive statistics into formal inferential analysis. Four distinct statistical techniques are demonstrated — a non-parametric group comparison, a one-way ANOVA, a correlation significance test, and a linear regression — each addressing a specific question about what drives housing values in the dataset.

This is a focused, statistics-first notebook rather than a full ML pipeline: the goal is to rigorously test relationships in the data using formal hypothesis tests (with explicit null hypotheses, test statistics, and p-values), not to build a predictive model.

---

## 2. Dataset

| Attribute | Value |
|---|---|
| Source | OpenML (`fetch_openml(name='boston', version=1)`) |
| Rows | 506 |
| Columns | 14 (13 features + `MEDV` target) |

**Feature reference:**

| Column | Meaning |
|---|---|
| `CRIM` | Per-capita crime rate by town |
| `ZN` | Proportion of residential land zoned for large lots |
| `INDUS` | Proportion of non-retail business acres per town |
| `CHAS` | Charles River dummy variable (1 if tract bounds the river, 0 otherwise) |
| `NOX` | Nitric oxide concentration (air pollution proxy) |
| `RM` | Average number of rooms per dwelling |
| `AGE` | Proportion of owner-occupied units built before 1940 |
| `DIS` | Weighted distance to five Boston employment centers |
| `RAD` | Index of accessibility to radial highways |
| `TAX` | Full-value property tax rate |
| `PTRATIO` | Pupil-teacher ratio by town |
| `B` | A demographic-derived statistic from the original 1978 study |
| `LSTAT` | Percentage of lower-status population |
| `MEDV` | **Target** — median value of owner-occupied homes ($1,000s) |

> **A note on this dataset's status:** `load_boston()` was removed from scikit-learn starting in version 1.2, because the `B` column encodes a racially-derived statistic from the original 1978 study design that the scikit-learn maintainers determined was not appropriate to distribute as a convenience-loaded "toy dataset" without that context. This notebook works around the removal by pulling the same data from OpenML instead. It's included here for statistical-methods practice; if this project were extended toward an actual predictive model, this is a relevant ethical consideration worth being aware of and addressing explicitly (e.g. by excluding or carefully justifying the use of `B`).

---

## 3. Methodology

Four independent hypothesis tests were run, each targeting a different type of relationship:

```
Load Data (OpenML)
     │
     ├──► Mann-Whitney U Test  — MEDV by CHAS (river-adjacent vs. not)
     │
     ├──► One-Way ANOVA        — MEDV across AGE groups (binned)
     │
     ├──► Pearson Correlation  — NOX vs. INDUS
     │
     └──► OLS Linear Regression — MEDV ~ DIS
```

---

## 4. Test 1 — Mann-Whitney U Test: MEDV by Charles River Adjacency

**Question:** Do homes near the Charles River (`CHAS = 1`) have significantly different median values than homes that aren't (`CHAS = 0`)?

**Method:** Mann-Whitney U — a non-parametric test appropriate when comparing two independent groups without assuming a normal distribution.

**Result as run:**
```
CHAS=0 count: 0
CHAS=1 count: 0
Mann-Whitney U statistic: nan
P-value: nan
```

**⚠️ Data quality issue identified:** Both group counts returned 0, which should be immediately suspicious — the full dataset has 506 rows, and `CHAS` is a binary indicator, so at least *some* rows should match `CHAS == 1` and `CHAS == 0`. This happened because OpenML's `boston` dataset loads `CHAS` as a **categorical/string** column (e.g. values stored as `'0'` and `'1'` as text, or as a pandas `category` dtype) rather than as an integer — so the comparison `df['CHAS'] == 1` silently matches nothing instead of raising an error. **The fix** is to cast the column explicitly before filtering:
```python
df['CHAS'] = df['CHAS'].astype(int)
```
This is flagged here rather than silently corrected, because it's a genuinely useful catch: a comparison that returns `NaN` without erroring can easily be mistaken for a valid "no difference found" result if the group-count sanity check hadn't been printed alongside it.

---

## 5. Test 2 — One-Way ANOVA: MEDV Across Building-Age Groups

**Question:** Does the median home value differ significantly across neighborhoods with different building ages?

**Method:** Properties were binned into three `AGE` groups — `0–50`, `51–70`, `71–100` (percent of units built before 1940) — and compared using a one-way ANOVA.

**Result:**
```
F-statistic: 34.26
P-value: 1.12e-14
→ Reject H0: At least one AGE group differs
```

**Interpretation:** The p-value is far below the conventional 0.05 threshold, providing strong statistical evidence that median home value is not the same across all building-age groups — older housing stock is associated with a significantly different (based on the underlying data, lower) median value.

---

## 6. Test 3 — Pearson Correlation: NOX vs. INDUS

**Question:** Is air pollution (`NOX`) related to the proportion of industrial land (`INDUS`) in a given area?

**Result:**
```
Pearson correlation coefficient: 0.764
P-value: 7.91e-98
→ Reject H0: Significant correlation exists
```

**Interpretation:** A correlation coefficient of 0.76 indicates a strong positive linear relationship, and the extremely small p-value confirms this is not due to chance. This result is intuitive and serves as a sanity check on the data: areas with more industrial land use do, in fact, show meaningfully higher air pollution levels.

---

## 7. Test 4 — OLS Linear Regression: MEDV ~ DIS

**Question:** Does distance to employment centers (`DIS`) predict home value (`MEDV`)?

**Result (via `statsmodels.OLS`):**

| Metric | Value |
|---|---|
| R-squared | 0.062 |
| F-statistic | 33.58 (p = 1.21e-08) |
| `DIS` coefficient | +1.0916 (p < 0.001, 95% CI: [0.722, 1.462]) |
| `const` (intercept) | 18.39 |

**Interpretation:** The relationship between distance-to-employment-centers and home value is **statistically significant** (the coefficient is reliably different from zero) but **practically weak** — `DIS` alone explains only about 6.2% of the variance in `MEDV` (R² = 0.062). The positive coefficient indicates that, on average, homes farther from Boston's employment centers tend to have *higher* median values in this dataset — plausibly reflecting that areas immediately surrounding dense employment centers were more industrial/lower-value, while further-out residential areas commanded higher prices. This finding illustrates an important distinction: statistical significance (a real, non-zero effect) does not imply the predictor is practically sufficient on its own — a real predictive model would need `DIS` combined with other features (like `RM`, `LSTAT`, `NOX`) to explain a meaningful share of price variation.

---

## 8. Key Insights

- **Building age is strongly associated with value** — the ANOVA result (p < 0.001) is the strongest, cleanest finding in this analysis
- **Industrial land use and air pollution move together** — as expected, and confirmed with high statistical confidence (r = 0.76)
- **Distance to employment centers alone is a weak predictor** — statistically real, but explains only ~6% of price variation; a useful reminder that significance and predictive power are two different things
- **Always sanity-check group sizes before interpreting a hypothesis test** — the CHAS/Mann-Whitney result is a concrete example of how a silent type-mismatch can produce a technically "valid-looking" but meaningless `NaN` result

---

## 9. Repository Structure

```
Boston_Analysis.ipynb/
├── README.md
└── Boston_Analysis.ipynb    ← Statistical hypothesis testing notebook
```

---

## 10. Setup & Installation

```bash
pip install pandas scikit-learn scipy statsmodels
jupyter notebook Boston_Analysis.ipynb
```

No external dataset file is needed — the data is fetched directly from OpenML at runtime.

---

## 11. Tech Stack

**Data handling:** Python, pandas, scikit-learn (`fetch_openml`)
**Statistical testing:** SciPy (`mannwhitneyu`, `f_oneway`, `pearsonr`), statsmodels (`OLS`)
**Environment:** Jupyter Notebook

---

## 12. Scope & Limitations

- **The `CHAS` dtype issue (Section 4) should be fixed before drawing any conclusion about river adjacency** — as currently run, that test produced no usable result
- **This notebook tests individual relationships in isolation**, not a combined multivariate model — the OLS regression uses only one predictor (`DIS`); a fuller model would include multiple features simultaneously (and check for multicollinearity between them, e.g. `NOX` and `INDUS`, which are already known to be correlated per Section 6)
- **Ethical note on the `B` column** — see the callout in Section 2. Any extension of this work into a predictive model should explicitly address this rather than using the feature set unchanged
- **No train/test split or predictive evaluation** — this notebook is scoped to statistical inference on the full dataset, not out-of-sample prediction
