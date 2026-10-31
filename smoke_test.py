"""Pre-deployment verification.

Run:  python smoke_test.py

Checks the things that silently break between a working laptop and a working
Streamlit Cloud deployment:
  1. every artifact loads under the installed library versions;
  2. scoring test_data.csv reproduces reports/metrics_test.csv to 4 dp;
  3. app.py contains no training call;
  4. the degenerate uploads a grader might try produce handled errors, not crashes.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.evaluate import METRIC_ORDER, compute_metrics, format_metric  # noqa: E402

failures = []


def check(label, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{(' - ' + detail) if detail and not condition else ''}")
    if not condition:
        failures.append(label)


def main():
    meta = json.loads((ROOT / "model" / "metadata.json").read_text(encoding="utf-8"))
    published = pd.read_csv(ROOT / "reports" / "metrics_test.csv").set_index("ML Model Name")
    test = pd.read_csv(ROOT / "test_data.csv")
    X, y = test[meta["feature_order"]], test[meta["target_column"]]
    classes = meta["classes"]

    print("\n[1] artifacts load and reproduce the published metrics")
    for label, fname in meta["model_files"].items():
        path = ROOT / "model" / fname
        check(f"{label}: artifact present", path.exists())
        if not path.exists():
            continue
        mismatch = False
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            try:
                pipe = joblib.load(path)
            except InconsistentVersionWarning:
                mismatch = True
        if mismatch:
            # Outside the block, so the "error" filter is no longer active and this
            # load warns instead of raising again.
            pipe = joblib.load(path)
        check(f"{label}: no library version mismatch", not mismatch)
        check(f"{label}: exposes predict_proba", hasattr(pipe, "predict_proba"))
        check(f"{label}: artifact under 25 MB", path.stat().st_size < 25 * 1024**2)

        got = compute_metrics(pipe, X, y, classes)
        for m in METRIC_ORDER:
            exp = float(published.loc[label, m])
            check(f"{label}: {m} matches published", abs(got[m] - exp) < 5e-5,
                  f"got {got[m]:.6f}, published {exp:.6f}")

    print("\n[2] the app cannot retrain")
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    check("app.py contains no .fit( call", ".fit(" not in src)
    check("app.py contains no fit_transform", "fit_transform" not in src)
    check("app.py imports nothing from train_models", "train_models" not in src)

    print("\n[3] degenerate inputs are handled, not crashed on")
    pipe = joblib.load(ROOT / "model" / meta["model_files"][meta["winner"]])
    one_class = test[test[meta["target_column"]] == classes[0]]
    res = compute_metrics(pipe, one_class[meta["feature_order"]],
                          one_class[meta["target_column"]], classes)
    check("single-class subset returns AUC=None rather than raising", res["AUC"] is None)
    check("single-class subset still returns the other five metrics",
          all(res[m] is not None for m in METRIC_ORDER if m != "AUC"))

    # Regression: scikit-learn returns NaN (it does NOT raise) when a class named in
    # labels= is absent from y_true, and NaN rendered as the literal string "nan".
    rare = min(classes, key=lambda c: (test[meta["target_column"]] == c).sum())
    subset = test[test[meta["target_column"]] != rare]
    res2 = compute_metrics(pipe, subset[meta["feature_order"]],
                           subset[meta["target_column"]], classes)
    check(f"subset missing '{rare}' yields AUC=None, not NaN",
          res2["AUC"] is None, f"got {res2['AUC']!r}")
    check("that AUC renders as 'n/a' rather than 'nan'",
          format_metric(res2["AUC"]) == "n/a", f"got {format_metric(res2['AUC'])!r}")
    check("the other five metrics survive a missing class",
          all(res2[m] is not None for m in METRIC_ORDER if m != "AUC"))

    # Regression: NaN arriving via DataFrame construction must also render as n/a.
    check("format_metric renders NaN as 'n/a'", format_metric(float("nan")) == "n/a")
    check("format_metric renders None as 'n/a'", format_metric(None) == "n/a")

    # Regression: unparseable uploads must be caught before any schema check.
    app_src = (ROOT / "app.py").read_text(encoding="utf-8")
    check("app.py handles EmptyDataError on upload", "EmptyDataError" in app_src)
    check("app.py handles a non-UTF-8 upload", "UnicodeDecodeError" in app_src)
    check("app.py retries a version-mismatched load outside catch_warnings",
          "return joblib.load(MODEL_DIR / filename), note" in app_src)

    print("\n[4] repo completeness")
    for rel in ["app.py", "requirements.txt", "README.md", "test_data.csv",
                "train_models.py", "model/metadata.json", "model/preprocessor.joblib"]:
        check(f"{rel} present", (ROOT / rel).exists())

    print(f"\n{'ALL CHECKS PASSED' if not failures else str(len(failures)) + ' CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
