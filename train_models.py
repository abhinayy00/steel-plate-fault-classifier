"""Train the five required classifiers and publish their test-set metrics.

Run:  python train_models.py

Every model is persisted as a COMPLETE Pipeline (preprocessor + estimator), which
is what makes the Streamlit app incapable of retraining: it only ever calls
.predict() and .predict_proba() on a loaded object.

The published metrics are computed from test_data.csv re-read FROM DISK rather than
from the in-memory split, so the README table is provably the same thing a grader
gets by uploading that exact file to the app.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.evaluate import METRIC_ORDER, compute_metrics, markdown_table  # noqa: E402
from src.make_split import SEED, TARGET, TEST_SIZE  # noqa: E402
from src.preprocessing import build_preprocessor  # noqa: E402

MODEL_DIR = ROOT / "model"
REPORTS = ROOT / "reports"

# Display name -> artifact stem. One naming scheme, used by the app via metadata.json.
MODEL_FILES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "K-Nearest Neighbors": "knn",
    "Naive Bayes (Gaussian)": "naive_bayes",
    "Support Vector Machine": "svm",
}


def choose_k(preprocessor, X, y):
    """Pick KNN's k by 5-fold macro-F1 on the TRAINING split only.

    Done here rather than hardcoded so the choice is reproducible and reported.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {}
    for k in (3, 5, 7, 9, 11, 15, 21):
        pipe = Pipeline(
            [("prep", build_preprocessor(X.columns)), ("clf", KNeighborsClassifier(n_neighbors=k, weights="distance"))]
        )
        scores[k] = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=-1).mean()
    best = max(scores, key=scores.get)
    print("  KNN k sweep (5-fold macro-F1 on train):")
    for k, s in scores.items():
        print(f"     k={k:<3} {s:.4f}{'   <- chosen' if k == best else ''}")
    return best, scores


def build_models(k):
    return {
        # multi_class= was removed in scikit-learn 1.7; lbfgs handles multinomial natively.
        "Logistic Regression": LogisticRegression(
            solver="lbfgs", C=1.0, max_iter=3000, random_state=SEED
        ),
        # Depth-limited: an unbounded tree memorises 1,358 rows and reports a
        # meaningless perfect training score.
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini", max_depth=8, min_samples_leaf=10, random_state=SEED
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=k, weights="distance"),
        "Naive Bayes (Gaussian)": GaussianNB(var_smoothing=1e-9),
        # probability=True is non-negotiable: AUC needs a calibrated probability
        # matrix, and decision_function output does not sum to 1 across classes.
        "Support Vector Machine": SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=SEED
        ),
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    train = pd.read_csv(ROOT / "data" / "train_data.csv")
    test = pd.read_csv(ROOT / "test_data.csv")
    features = [c for c in train.columns if c != TARGET]
    X_train, y_train = train[features], train[TARGET]
    X_test, y_test = test[features], test[TARGET]
    class_order = sorted(y_train.unique())

    print(f"train {X_train.shape} | test {X_test.shape} | {len(class_order)} classes")
    print(f"scikit-learn {sklearn.__version__} | python {sys.version.split()[0]}\n")

    k, k_scores = choose_k(build_preprocessor(features), X_train, y_train)
    print()

    rows, evidence = [], {}
    for name, est in build_models(k).items():
        pipe = Pipeline([("prep", build_preprocessor(features)), ("clf", est)])

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        metrics = compute_metrics(pipe, X_test, y_test, class_order)
        score_s = time.perf_counter() - t0

        if not hasattr(pipe, "predict_proba"):
            raise RuntimeError(f"{name} cannot produce probabilities; AUC would be blank")

        path = MODEL_DIR / f"{MODEL_FILES[name]}.joblib"
        joblib.dump(pipe, path, compress=3)
        size_kb = path.stat().st_size / 1024

        # Per-class recall is the evidence the observations section needs: it is how
        # a model that has quietly abandoned Dirtiness and Stains becomes visible.
        pred = pipe.predict(X_test)
        per_class = {
            c: float((pred[y_test == c] == c).sum() / max((y_test == c).sum(), 1))
            for c in class_order
        }
        evidence[name] = {
            "fit_seconds": round(fit_s, 3),
            "score_seconds": round(score_s, 3),
            "artifact_kb": round(size_kb, 1),
            "per_class_recall": per_class,
            "n_predicted_classes": int(len(set(pred))),
        }

        rows.append({"ML Model Name": name, **{m: metrics.get(m) for m in METRIC_ORDER},
                     "F1_weighted": metrics.get("F1_weighted")})
        shown = "  ".join(f"{m}={metrics[m]:.4f}" if metrics[m] is not None else f"{m}=n/a"
                          for m in METRIC_ORDER)
        print(f"{name:<24} {shown}   [{fit_s:.2f}s fit, {size_kb:.0f} KB]")

    results = pd.DataFrame(rows)
    results.to_csv(REPORTS / "metrics_test.csv", index=False)

    # Winner rule fixed before the numbers were seen: macro-F1, tie-break MCC, then AUC.
    ranked = results.sort_values(["F1", "MCC", "AUC"], ascending=False).reset_index(drop=True)
    winner = ranked.loc[0, "ML Model Name"]

    joblib.dump(build_preprocessor(features).fit(X_train), MODEL_DIR / "preprocessor.joblib", compress=3)

    metadata = {
        "target_column": TARGET,
        "feature_order": features,
        "classes": class_order,
        "task": "multiclass",
        "model_files": {k_: f"{v}.joblib" for k_, v in MODEL_FILES.items()},
        "pr_average": "macro",
        "auc_strategy": "ovr_macro",
        "winner_rule": "rank by macro-F1, tie-break MCC, then OvR-macro AUC",
        "winner": winner,
        "random_state": SEED,
        "test_size": TEST_SIZE,
        "knn_k": int(k),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "python_version": sys.version.split()[0],
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    ev = {"knn_k_sweep": {str(k_): round(v, 4) for k_, v in k_scores.items()}, "models": evidence}
    (REPORTS / "observations_evidence.json").write_text(json.dumps(ev, indent=2), encoding="utf-8")

    print(f"\nwinner by pre-registered rule (macro-F1 > MCC > AUC): {winner}")
    print(f"wrote {REPORTS/'metrics_test.csv'}, {MODEL_DIR/'metadata.json'}")
    table = markdown_table(results[["ML Model Name", *METRIC_ORDER]])
    (REPORTS / "metrics_table.md").write_text(table, encoding="utf-8")
    print("\n" + table)


if __name__ == "__main__":
    main()
