# Steel Plate Fault Classification - Five-Model Comparison

A comparison of five supervised classifiers on the UCI *Steel Plates Faults* dataset,
with a Streamlit front-end that scores an uploaded test CSV against the saved models.

- **Live app:** https://<your-app>.streamlit.app
- **Repository:** https://github.com/abhinayy00/steel-plate-fault-classifier

---

## a. Problem statement

Given 27 geometric and luminosity measurements taken from an automated inspection
system on a steel plate surface defect, predict which of **seven mutually-exclusive
fault types** the defect belongs to.

This is a multi-class classification problem. It is operationally useful because the
fault type determines the corrective action on the rolling line, and it is
statistically interesting because the classes are heavily imbalanced (12.2:1 between
the most and least common) and one class, `Other_Faults`, is an explicit catch-all
rather than a coherent physical category. That combination means **accuracy alone is
misleading**, which is why all six required metrics are reported.

## b. Dataset description

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository, *Steel Plates Faults* (id 198) |
| URL | https://archive.ics.uci.edu/dataset/198/steel+plates+faults |
| DOI | 10.24432/C5J88N |
| Instances | **1941** (requirement: >= 500) |
| Predictors | **27**, all numeric (requirement: >= 12) |
| Target | `Fault` - 7 mutually-exclusive classes |
| Missing values | None |
| Raw file SHA-256 | `e33d095557276bb3...` |

The requirement arithmetic: 27 >= 12 and 1941 >= 500, so the
dataset clears both minimums with margin.

**A note on the target.** The UCI file does not ship a single label column. It
provides the label as **seven binary indicator columns**, exactly one of which is set
per row. These are collapsed into one categorical `Fault` column with `idxmax`, after
asserting that every row sums to 1. The seven indicator columns are then removed from
the feature matrix - leaving even one of them in `X` makes the task trivially
separable and every model reports close to 100% accuracy.

### Class distribution

| Fault class  | Instances | Share |
|--------------|-----------|-------|
| Other_Faults | 673       | 34.7% |
| Bumps        | 402       | 20.7% |
| K_Scratch    | 391       | 20.1% |
| Z_Scratch    | 190       | 9.8%  |
| Pastry       | 158       | 8.1%  |
| Stains       | 72        | 3.7%  |
| Dirtiness    | 55        | 2.8%  |

### Preprocessing and split

| Step | Choice | Reason |
|---|---|---|
| Split | Stratified 70/30, `random_state=5465` | 70/30 rather than the usual 80/20 puts ~17 `Dirtiness` and ~22 `Stains` rows in the test set instead of ~11 and ~15, which makes the macro-averaged metrics materially less noisy. Seed derived from the BITS ID. |
| Train / test | 1358 / 583 rows | Every class is present in the test split, which multi-class AUC requires. |
| Imputation | Median | Defensive: the training data has no gaps, but an uploaded CSV may. |
| Scaling | `StandardScaler`, one shared preprocessor | Mandatory for KNN, SVM and Logistic Regression. Harmless for the Decision Tree (invariant under monotone rescaling) and for GaussianNB (per-feature mean and variance absorb it), so all five models see an identical matrix and the comparison stays like-for-like. |

## c. GitHub repository link

https://github.com/abhinayy00/steel-plate-fault-classifier

## d. Models used

All five classifiers are trained on the same 1358-row training split and
evaluated on the same 583-row held-out test split. Each is persisted as a
complete `Pipeline` (imputer -> scaler -> estimator), so the deployed app applies the
identical transform at inference and never refits.

| Model | Key settings |
|---|---|
| Logistic Regression | `solver='lbfgs'`, `C=1.0`, `max_iter=3000` |
| Decision Tree | `criterion='gini'`, `max_depth=8`, `min_samples_leaf=10` |
| K-Nearest Neighbors | `n_neighbors=7` (5-fold macro-F1 sweep on train), `weights='distance'` |
| Naive Bayes | `GaussianNB(var_smoothing=1e-9)` - Gaussian, not Multinomial, because all 27 predictors are continuous and standardisation produces negative values |
| Support Vector Machine | `kernel='rbf'`, `C=1.0`, `gamma='scale'`, `probability=True` |

### Metric definitions used

The task is multi-class, so the averaged metrics need an explicit convention:

- **Accuracy** - `accuracy_score`
- **AUC** - `roc_auc_score(..., multi_class='ovr', average='macro', labels=classes)`
- **Precision / Recall / F1** - `average='macro'`, `zero_division=0`
- **MCC** - `matthews_corrcoef`, natively multi-class

`macro` is used rather than `micro` because in single-label multi-class problems
micro-precision, micro-recall and micro-F1 are all identical to accuracy, which would
waste three of the six required columns. It is used rather than `weighted` because at
12.2:1 imbalance a weighted average tracks accuracy and hides exactly the rare-class
failures this dataset is interesting for.

### Comparison table

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression    | 0.6998   | 0.9251 | 0.6956    | 0.7118 | 0.7010 | 0.6159 |
| Decision Tree          | 0.6844   | 0.9053 | 0.7005    | 0.7110 | 0.7045 | 0.5944 |
| K-Nearest Neighbors    | 0.7153   | 0.9177 | 0.7043    | 0.7442 | 0.7189 | 0.6413 |
| Naive Bayes (Gaussian) | 0.5866   | 0.9057 | 0.6291    | 0.6727 | 0.6092 | 0.5275 |
| Support Vector Machine | 0.7341   | 0.9415 | 0.7602    | 0.7192 | 0.7276 | 0.6575 |

### Per-class recall (the evidence behind the observations)

| ML Model Name          | Bumps | Dirtiness | K_Scratch | Other_Faults | Pastry | Stains | Z_Scratch |
|------------------------|-------|-----------|-----------|--------------|--------|--------|-----------|
| Logistic Regression    | 0.603 | 0.588     | 0.940     | 0.614        | 0.468  | 0.909  | 0.860     |
| Decision Tree          | 0.512 | 0.706     | 0.923     | 0.634        | 0.468  | 0.909  | 0.825     |
| K-Nearest Neighbors    | 0.686 | 0.765     | 0.949     | 0.599        | 0.362  | 0.955  | 0.895     |
| Naive Bayes (Gaussian) | 0.835 | 0.647     | 0.906     | 0.213        | 0.574  | 0.955  | 0.579     |
| Support Vector Machine | 0.719 | 0.588     | 0.932     | 0.683        | 0.298  | 0.955  | 0.860     |

### Training cost

| ML Model Name          | Fit time (s) | Artifact size (KB) |
|------------------------|--------------|--------------------|
| Logistic Regression    | 0.10         | 4.00               |
| Decision Tree          | 0.02         | 6.30               |
| K-Nearest Neighbors    | 0.01         | 177.70             |
| Naive Bayes (Gaussian) | 0.01         | 5.30               |
| Support Vector Machine | 0.19         | 136.40             |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Mid-table on every metric (accuracy 0.6998, macro-F1 0.7010). A single linear boundary per class is a poor fit for a feature space where several predictors are algebraic transforms of one another, but L2 regularisation keeps it stable and it is the second-fastest to fit. Notably it is beaten on macro-F1 by the Decision Tree despite a *higher* accuracy, because it leans toward the majority `Other_Faults` class. |
| Decision Tree | Lowest accuracy of the four non-Bayesian models (0.6844) yet a higher macro-F1 (0.7045) than Logistic Regression - axis-aligned splits handle the rare classes relatively well. It is the weakest model on `Bumps` (recall 0.512). Depth was capped at 8 with a minimum leaf size of 10; unconstrained it memorises the 1,358 training rows and reports a meaningless perfect training score. |
| K-Nearest Neighbors | Best recall of any model (0.7442) and the strongest on the rarest class, `Dirtiness` (0.765). A local, non-parametric decision rule suits a feature space where classes form tight clusters. Two practical costs: the saved artifact is 178 KB because it stores the entire training matrix, and k had to be tuned - 5-fold macro-F1 on the training split peaked at k=7, though the optimum is flat (k=3 scores 0.7678 against 0.7705). |
| Naive Bayes (Gaussian) | Last on accuracy (0.5866) but its AUC (0.9057) matches the Decision Tree's (0.9053) - it *ranks* samples well and *decides* badly. The per-class recall table shows exactly why: it scores only 0.213 on `Other_Faults` while every other model manages 0.60-0.68, yet it is the best model on `Bumps` (0.835) and ties for best on `Stains` (0.955). `Other_Faults` is a heterogeneous catch-all with no single Gaussian shape, and the conditional-independence assumption is additionally violated because `Pixels_Areas`, `LogOfAreas` and `SigmoidOfAreas` are three transforms of one quantity. Since `Other_Faults` is 34.7% of the data, accuracy collapses while ranking quality survives. |
| Support Vector Machine | Best on four of the six metrics - accuracy 0.7341, AUC 0.9415, precision 0.7602, macro-F1 0.7276, MCC 0.6575 - and the best model on `Other_Faults` (recall 0.683), which is what drives its accuracy lead. An RBF kernel on standardised features handles the non-linear, overlapping boundaries that defeat the linear model. It is the slowest to fit (0.19 s) because `probability=True` adds internal Platt calibration, but that cost is paid once at training and never in the app. |
| **Overall, Winner for your dataset?** | **Support Vector Machine**, by the rule fixed before any number was seen: rank on macro-F1, tie-break on MCC, then one-vs-rest macro AUC. It leads on macro-F1 (0.7276), MCC (0.6575) and AUC (0.9415) simultaneously, so the ranking does not depend on which of the three is used. The honest caveat is that `Dirtiness` has only 17 test rows and `Stains` 22, so macro-averaged metrics carry real sampling noise and the gap to K-Nearest Neighbors (macro-F1 0.7189) is narrower than three decimal places suggest. |


## e. Streamlit app features

| Requirement | Implementation |
|---|---|
| Dataset upload (CSV) | `st.file_uploader` in the sidebar, plus a toggle to score the bundled `test_data.csv` so the app is usable in one click. Schema validation reports *every* problem at once - missing target, missing or extra features, non-numeric cells, unknown class labels - and shows the expected header. |
| Model selection dropdown | `st.selectbox` populated from `model/metadata.json`; no model names are hardcoded in `app.py`. |
| Evaluation metrics | All six metrics as `st.metric` tiles, plus a "Compare all five" tab scoring every model on the uploaded file with the best value in each column highlighted. |
| Confusion matrix / report | Both: a labelled confusion matrix with a colour gradient, and the full per-class `classification_report`. Rendered as DataFrames rather than matplotlib, which keeps the deployed dependency set to six packages. |

The app **loads saved models and never retrains** - its only estimator calls are
`.predict()` and `.predict_proba()`. Because preprocessing lives *inside* each
persisted pipeline, fitting on uploaded data is structurally impossible rather than
merely avoided. `smoke_test.py` asserts this.

## f. Reproducing the results

```bash
pip install -r requirements.txt
python src/make_split.py     # fetch UCI 198, collapse the target, write the splits
python train_models.py       # train all five, save artifacts, publish metrics
python build_readme.py       # regenerate this file from those artifacts
python smoke_test.py         # verify artifacts reproduce the published numbers
streamlit run app.py
```

Trained with scikit-learn 1.9.1 on Python 3.14.6.
`requirements.txt` pins exact versions: a scikit-learn mismatch between the machine
that writes the pickles and the machine that reads them raises only a warning by
default, so the model would load and score subtly wrongly in silence. The app
escalates that warning to an error and shows a provenance panel comparing training
and runtime versions.

## g. BITS Virtual Lab execution

See `reports/screenshots/bits_lab_execution.png`.

## h. References

1. M. Buscema, S. Terzi, W. Tastle. *Steel Plates Faults Data Set*. UCI Machine
   Learning Repository, 2010. DOI 10.24432/C5J88N.
2. F. Pedregosa et al. *Scikit-learn: Machine Learning in Python*. JMLR 12, 2011.
3. D. Chicco, G. Jurman. *The advantages of the Matthews correlation coefficient
   (MCC) over F1 score and accuracy in binary classification evaluation*.
   BMC Genomics 21, 2020.
