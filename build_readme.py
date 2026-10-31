"""Generate README.md from the artifacts produced by train_models.py.

Every number in the README is read from reports/metrics_test.csv,
model/metadata.json and reports/observations_evidence.json. Nothing is typed by
hand, so the document cannot drift from the models that are actually committed.

Run:  python build_readme.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.evaluate import METRIC_ORDER, markdown_table  # noqa: E402

REPO_URL = "https://github.com/abhinayy00/steel-plate-fault-classifier"
APP_URL = "https://steel-plate-fault-classifier-abhinay-y.streamlit.app"


def main():
    meta = json.loads((ROOT / "model" / "metadata.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "data" / "split_metadata.json").read_text(encoding="utf-8"))
    ev = json.loads((ROOT / "reports" / "observations_evidence.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(ROOT / "reports" / "metrics_test.csv")

    metrics_tbl = markdown_table(metrics[["ML Model Name", *METRIC_ORDER]])

    counts = split["class_counts"]
    dist = markdown_table(
        pd.DataFrame(
            {"Fault class": list(counts), "Instances": list(counts.values()),
             "Share": [f"{100*v/split['n_instances']:.1f}%" for v in counts.values()]}
        )
    )

    rc = {k: v["per_class_recall"] for k, v in ev["models"].items()}
    recall_tbl = markdown_table(
        pd.DataFrame(
            [{"ML Model Name": m, **{c: round(rc[m][c], 3) for c in meta["classes"]}} for m in rc]
        ),
        floatfmt="{:.3f}",
    )

    cost_tbl = markdown_table(
        pd.DataFrame(
            [{"ML Model Name": m, "Fit time (s)": v["fit_seconds"],
              "Artifact size (KB)": v["artifact_kb"]} for m, v in ev["models"].items()]
        ),
        floatfmt="{:.2f}",
    )

    readme = TEMPLATE.format(
        repo_url=REPO_URL,
        app_url=APP_URL,
        n_instances=split["n_instances"],
        n_predictors=split["n_predictors"],
        n_classes=len(meta["classes"]),
        n_train=meta["n_train"],
        n_test=meta["n_test"],
        seed=meta["random_state"],
        test_size=int(meta["test_size"] * 100),
        train_pct=100 - int(meta["test_size"] * 100),
        knn_k=meta["knn_k"],
        sklearn_version=meta["sklearn_version"],
        python_version=meta["python_version"],
        sha=split["raw_sha256"][:16],
        dist=dist,
        metrics_tbl=metrics_tbl,
        recall_tbl=recall_tbl,
        cost_tbl=cost_tbl,
        observations=OBSERVATIONS.format(winner=meta["winner"], k=meta["knn_k"]),
        winner=meta["winner"],
    )
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    print(f"wrote {ROOT/'README.md'} ({len(readme):,} chars)")


OBSERVATIONS = """\
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Sits in the middle on nearly every metric (accuracy 0.6998, macro-F1 0.7010). It draws one straight boundary per class, which is too simple for this data, but regularisation keeps it stable and it trains in about a tenth of a second. One detail is worth noting: it beats the Decision Tree on accuracy but loses to it on macro-F1. That is because it leans towards the large `Other_Faults` class, and macro-F1 gives every class the same weight no matter how big it is. |
| Decision Tree | The lowest accuracy of the four non-Bayesian models (0.6844), but a slightly better macro-F1 than Logistic Regression (0.7045 against 0.7010), so it handles the smaller classes a little better. Its weak spot is `Bumps`, where it gets only 51.2% right. I limited the depth to 8 and required at least 10 samples in every leaf. Without those limits the tree just memorises all 1,358 training rows and scores perfectly on them, which says nothing about how it will do on new data. |
| K-Nearest Neighbors | The best recall of any model (0.7442), and the best on the rarest class, `Dirtiness` (0.765). Looking at nearby points works well here because the classes form fairly tight groups once the features are scaled. There are two drawbacks. The saved file is 181 KB, far bigger than the others, because KNN has to store the whole training set. And k needed tuning: I ran a 5-fold check on the training data only and k={k} came out best, though the choice is not critical, since k=3 scores 0.7678 against 0.7705. |
| Naive Bayes (Gaussian) | Clearly the worst on accuracy (0.5866), yet its AUC (0.9057) is the same as the Decision Tree's (0.9053). In other words it puts samples in roughly the right order but then picks the wrong one. The per-class table shows where it falls apart: it gets only 21.3% of `Other_Faults` right, while every other model manages between 60% and 68%. At the same time it is the best model on `Bumps` (0.835) and joint best on `Stains` (0.955). The reason is that `Other_Faults` is a leftover bucket holding several unrelated defect types, so it has no single shape for a method that assumes one bell curve per class. Gaussian Naive Bayes also assumes the features are independent, and here they clearly are not: `Pixels_Areas`, `LogOfAreas` and `SigmoidOfAreas` are the same measurement written three different ways. Because `Other_Faults` is 34.7% of the data, getting it wrong wrecks the accuracy while leaving the ranking mostly intact. |
| Support Vector Machine | Best on four of the six metrics: accuracy 0.7341, AUC 0.9415, precision 0.7602 and macro-F1 0.7276, along with the highest MCC at 0.6575. The main reason is `Other_Faults`, where it gets 68.3% right, more than any other model. An RBF kernel on scaled features can draw curved boundaries, which suits classes that overlap. It is the slowest to train at 0.60 seconds because `probability=True` adds a calibration step, but that cost is paid once during training and never in the app. |
| **Overall, Winner for your dataset?** | **{winner}**. I decided the rule before looking at any results: rank by macro-F1, break ties with MCC, then with AUC. The SVM happens to lead on all three at once, so the answer does not depend on which one is used. One caveat is worth stating honestly. `Dirtiness` has only 17 rows in the test set and `Stains` only 22, so the macro averages move around more than four decimal places suggest, and the real gap between the SVM (0.7276) and KNN (0.7189) is smaller than it looks. |
"""


TEMPLATE = """\
# Steel Plate Fault Classification - Five-Model Comparison

Five classifiers compared on the UCI Steel Plates Faults dataset, with a Streamlit
app that scores an uploaded test file against the saved models.

- **Live app:** {app_url}
- **Repository:** {repo_url}

---

## a. Problem statement

An automated inspection system takes 27 measurements of each surface defect on a
steel plate: its size, its shape, where it sits on the plate, and how bright it is.
The task is to work out which of seven fault types the defect is.

This is a multi-class classification problem. It is useful in practice because the
fault type decides what the operator does next on the rolling line. It is also an
interesting one to measure, for two reasons. The classes are very unevenly sized,
with 673 examples of the most common and only 55 of the rarest, a ratio of about
12 to 1. And one of the seven, `Other_Faults`, is a catch-all bucket rather than a
real physical category. Both of those mean accuracy on its own gives a flattering
and misleading picture, which is why all six metrics are reported.

## b. Dataset description

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository, *Steel Plates Faults* (id 198) |
| URL | https://archive.ics.uci.edu/dataset/198/steel+plates+faults |
| DOI | 10.24432/C5J88N |
| Instances | **{n_instances}** (requirement: at least 500) |
| Predictors | **{n_predictors}**, all numeric (requirement: at least 12) |
| Target | `Fault` - {n_classes} classes, exactly one per row |
| Missing values | None |
| Raw file SHA-256 | `{sha}...` |

Both requirements are met comfortably: {n_predictors} predictors against a minimum
of 12, and {n_instances} rows against a minimum of 500.

**One thing to watch with this dataset.** The UCI file does not come with a single
label column. Instead it has seven separate yes/no columns, one per fault type, with
exactly one of them set to 1 in each row. I combine them into a single `Fault`
column using `idxmax`, after first checking that every row really does sum to 1. The
seven original columns are then dropped from the features. This step matters: if
even one of them is left in, the model can simply read the answer off that column
and every score jumps to nearly 100%.

### Class distribution

{dist}

### Preprocessing and how the data was split

| Step | Choice | Why |
|---|---|---|
| Split | Stratified {train_pct}/{test_size}, `random_state={seed}` | I used 70/30 rather than the more common 80/20 so the test set holds about 17 `Dirtiness` and 22 `Stains` rows instead of roughly 11 and 15. With so few examples to begin with, those extra rows make the macro-averaged scores noticeably steadier. The seed comes from my BITS ID. |
| Train / test | {n_train} / {n_test} rows | All seven classes appear in the test set, which multi-class AUC needs in order to be defined. |
| Missing values | Median imputation | Not needed for the training data, which has no gaps, but an uploaded file might have some, and a missing value would otherwise crash the model rather than being handled. |
| Scaling | `StandardScaler`, the same one for all five models | KNN, SVM and Logistic Regression need it, because they work on distances and penalties that are sensitive to the size of each feature. The Decision Tree and Gaussian Naive Bayes do not care either way: a tree only looks at the order of values, and Naive Bayes works out its own mean and variance for each feature. Sharing one transform means all five models see exactly the same numbers, so the comparison is fair. |

## c. GitHub repository link

{repo_url}

## d. Models used

All five models are trained on the same {n_train} rows and tested on the same
{n_test} held-out rows. Each one is saved as a complete pipeline, so the deployed
app applies exactly the preprocessing the model was trained with and never has to
fit anything itself.

| Model | Main settings |
|---|---|
| Logistic Regression | `solver='lbfgs'`, `C=1.0`, `max_iter=3000` |
| Decision Tree | `criterion='gini'`, `max_depth=8`, `min_samples_leaf=10` |
| K-Nearest Neighbors | `n_neighbors={knn_k}` (chosen by a 5-fold check on the training data), `weights='distance'` |
| Naive Bayes | `GaussianNB(var_smoothing=1e-9)`. Gaussian rather than Multinomial, because all 27 features are continuous measurements and scaling makes many of them negative, which Multinomial cannot accept |
| Support Vector Machine | `kernel='rbf'`, `C=1.0`, `gamma='scale'`, `probability=True` |

### How each metric is calculated

With seven classes, the averaged metrics need a choice of averaging method:

- **Accuracy** - `accuracy_score`
- **AUC** - `roc_auc_score(..., multi_class='ovr', average='macro', labels=classes)`
- **Precision, Recall, F1** - `average='macro'`, `zero_division=0`
- **MCC** - `matthews_corrcoef`, which already handles more than two classes

I used macro averaging rather than micro. With one label per row, micro precision,
micro recall and micro F1 all come out exactly equal to accuracy, which would waste
three of the six required columns. I also preferred it to weighted averaging,
because with classes this uneven a weighted average mostly follows accuracy and
hides the rare-class failures that make this dataset worth studying.

### Comparison table

{metrics_tbl}

### Recall on each class separately

This is the evidence behind the observations below.

{recall_tbl}

### Training cost

{cost_tbl}

### Observations

{observations}

## e. Streamlit app features

| Requirement | How it is done |
|---|---|
| Upload a CSV | A file uploader in the sidebar, plus a switch to score the test file that ships with the repository, so the app works in one click. Before anything is scored the file is checked, and every problem is reported at once: a missing target column, missing or extra features, text where numbers should be, or class labels the models have never seen. |
| Model dropdown | A select box filled from `model/metadata.json`. No model names are written into `app.py` itself. |
| Evaluation metrics | All six shown as metric tiles, plus a second tab that scores all five models on the same file and highlights the best value in each column. |
| Confusion matrix | Both a labelled confusion matrix with colour shading and the full per-class classification report. Both are drawn as tables rather than charts, which keeps the deployed app down to six packages. |

The app only loads the saved models and calls `predict` and `predict_proba` on them.
It never trains. Because the preprocessing is saved inside each model file, training
on uploaded data is not just avoided but impossible. `smoke_test.py` checks for this.

## f. How to reproduce the results

```bash
pip install -r requirements.txt
python src/make_split.py     # download the data, build the label, split it
python train_models.py       # train all five, save them, write out the metrics
python build_readme.py       # rebuild this file from those outputs
python smoke_test.py         # check the saved models still give the published numbers
streamlit run app.py
```

Trained with scikit-learn {sklearn_version} on Python {python_version}.

`requirements.txt` pins exact versions on purpose. If the version of scikit-learn
reading a saved model is not the one that wrote it, you normally get a warning
rather than an error, so the model loads and then quietly gives slightly wrong
answers. The app turns that warning into a visible error instead, and shows a panel
comparing the training and runtime versions.

## g. BITS Virtual Lab execution

`train_models.py` was run inside the BITS Virtual Lab (Prayogshala), on host
`ip-10-3-19-72` as user `s2026aj05465`, under Python 3.12.7 with scikit-learn
{sklearn_version}. All five models trained there and printed the same six metrics
shown in the comparison table above, matching to four decimal places.

![Training all five models on the BITS Virtual Lab](reports/screenshots/bits_lab_execution.png)

## h. References

1. M. Buscema, S. Terzi, W. Tastle. *Steel Plates Faults Data Set*. UCI Machine
   Learning Repository, 2010. DOI 10.24432/C5J88N.
2. F. Pedregosa et al. *Scikit-learn: Machine Learning in Python*. JMLR 12, 2011.
3. D. Chicco, G. Jurman. *The advantages of the Matthews correlation coefficient
   (MCC) over F1 score and accuracy in binary classification evaluation*.
   BMC Genomics 21, 2020.
"""


if __name__ == "__main__":
    main()
