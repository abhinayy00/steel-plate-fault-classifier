"""Single source of truth for the six evaluation metrics.

Imported by BOTH train_models.py and app.py so the numbers published in the README
can never drift from the numbers the Streamlit app shows for the same CSV.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Metric display order, used for the README table and the app tiles alike.
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def compute_metrics(pipeline, X, y_true, class_order):
    """Return the six assignment metrics for one fitted pipeline.

    class_order pins the label axis for every metric, so a test CSV that happens to
    be missing a rare class still lines up with the trained model's classes_ rather
    than silently shifting the probability columns.

    AUC is the fragile one: it needs a full probability matrix whose rows sum to 1,
    and it raises outright when the supplied subset contains fewer than two classes.
    That case returns None so the caller can render "n/a" instead of a misleading 0.0.
    """
    class_order = list(class_order)
    y_pred = pipeline.predict(X)
    multiclass = len(class_order) > 2

    # Threshold metrics come from .predict(); AUC alone comes from .predict_proba().
    # Platt-calibrated SVC can disagree between the two on a handful of rows, so the
    # split is deliberate rather than incidental.
    avg = "macro" if multiclass else "binary"
    kw = {"zero_division": 0}
    if multiclass:
        kw.update(average="macro", labels=class_order)
    else:
        kw.update(average="binary", pos_label=class_order[-1])

    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": _safe_auc(pipeline, X, y_true, class_order, multiclass),
        "Precision": float(precision_score(y_true, y_pred, **kw)),
        "Recall": float(recall_score(y_true, y_pred, **kw)),
        "F1": float(f1_score(y_true, y_pred, **kw)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }

    # Weighted variants are kept as secondary evidence for the observations section:
    # at 12:1 imbalance the macro/weighted gap is exactly what exposes a model that
    # has quietly abandoned the rare classes.
    if multiclass:
        out["F1_weighted"] = float(
            f1_score(y_true, y_pred, average="weighted", labels=class_order, zero_division=0)
        )
    return out


def _safe_auc(pipeline, X, y_true, class_order, multiclass):
    """One-vs-rest macro AUC, or None when the data cannot support it."""
    if not hasattr(pipeline, "predict_proba"):
        return None
    present = set(np.unique(np.asarray(y_true)))
    if len(present) < 2:
        return None
    try:
        proba = pipeline.predict_proba(X)
        if multiclass:
            # labels= pins the column order of `proba` to class_order. Without it,
            # a subset missing a class silently misaligns columns and the score is
            # wrong rather than absent.
            return float(
                roc_auc_score(
                    y_true, proba, multi_class="ovr", average="macro", labels=class_order
                )
            )
        pos_idx = list(pipeline.classes_).index(class_order[-1])
        return float(roc_auc_score(y_true, proba[:, pos_idx]))
    except ValueError:
        # Raised when a class in class_order is absent from y_true.
        return None


def confusion_frame(pipeline, X, y_true, class_order):
    """Confusion matrix as a labelled DataFrame (no matplotlib dependency)."""
    import pandas as pd

    cm = confusion_matrix(y_true, pipeline.predict(X), labels=list(class_order))
    return pd.DataFrame(
        cm,
        index=[f"true:{c}" for c in class_order],
        columns=[f"pred:{c}" for c in class_order],
    )


def format_metric(value):
    """Render a metric for display, distinguishing 'not computable' from zero."""
    return "n/a" if value is None else f"{value:.4f}"


def markdown_table(df, floatfmt="{:.4f}"):
    """Render a DataFrame as a markdown table without pulling in `tabulate`.

    pandas' own .to_markdown() requires the optional `tabulate` package, which would
    become another pinned dependency that has to install cleanly on Streamlit Cloud.
    Not worth it for a fixed-width table of five rows.
    """
    def cell(v):
        if v is None or (isinstance(v, float) and v != v):
            return "n/a"
        return floatfmt.format(v) if isinstance(v, float) else str(v)

    cols = list(df.columns)
    rows = [[cell(v) for v in rec] for rec in df.itertuples(index=False)]
    widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
              for i, c in enumerate(cols)]
    head = "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols)) + " |"
    rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = ["| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(cols))) + " |"
            for r in rows]
    return "\n".join([head, rule, *body])
