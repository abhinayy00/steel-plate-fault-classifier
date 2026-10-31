"""Steel Plate Fault Classifier - evaluation front-end.

This app EVALUATES pre-trained models; it never fits anything. Each artifact in
model/ is a complete scikit-learn Pipeline (imputer -> scaler -> estimator), so the
only estimator calls made here are .predict() and .predict_proba(). Preprocessing
is baked into the persisted object, which makes an accidental fit-on-upload
structurally impossible rather than merely discouraged.

Everything dataset-specific is read from model/metadata.json: there are no
hardcoded column names, class labels or model keys below.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.evaluate import METRIC_ORDER, compute_metrics, confusion_frame, format_metric

MODEL_DIR = ROOT / "model"
BUNDLED_TEST = ROOT / "test_data.csv"

st.set_page_config(page_title="Steel Plate Fault Classifier", page_icon="🔩", layout="wide")


# --------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner=False)
def load_metadata():
    return json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_model(filename):
    """Load one pipeline.

    A scikit-learn version mismatch between the machine that wrote the pickle and
    the machine reading it only raises a UserWarning by default: the model loads and
    may then score incorrectly in silence. Escalating it to an error turns an
    invisible wrong-answer failure into a visible one.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        try:
            return joblib.load(MODEL_DIR / filename), None
        except InconsistentVersionWarning as exc:
            model = joblib.load(MODEL_DIR / filename)
            return model, str(exc)


@st.cache_data(show_spinner=False)
def read_csv_bytes(raw: bytes):
    import io

    return pd.read_csv(io.BytesIO(raw))


def check_schema(df, meta):
    """Collect EVERY schema problem at once.

    Reporting one error per rerun turns a single bad upload into five round trips.
    """
    problems = []
    target = meta["target_column"]
    expected = meta["feature_order"]

    if df.empty:
        problems.append("The file parsed to zero rows.")
        return problems

    if target not in df.columns:
        problems.append(
            f"Missing the target column `{target}`. This app evaluates predictions "
            "against true labels, so the uploaded test file must carry them."
        )

    missing = [c for c in expected if c not in df.columns]
    if missing:
        problems.append(f"Missing {len(missing)} feature column(s): {', '.join(missing[:8])}"
                        + (" ..." if len(missing) > 8 else ""))

    extra = [c for c in df.columns if c not in expected and c != target]
    if extra:
        problems.append(f"{len(extra)} unexpected column(s) will be ignored: "
                        f"{', '.join(extra[:8])}" + (" ..." if len(extra) > 8 else ""))

    present = [c for c in expected if c in df.columns]
    non_numeric = [c for c in present if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        problems.append(f"Non-numeric values in: {', '.join(non_numeric[:8])}")

    if target in df.columns:
        unknown = sorted(set(df[target].dropna().unique()) - set(meta["classes"]))
        if unknown:
            problems.append(f"Unknown class label(s) in `{target}`: {unknown[:6]}")

    return problems


# --------------------------------------------------------------------------- sidebar
meta = load_metadata()
CLASSES = meta["classes"]
MODEL_MAP = meta["model_files"]

st.sidebar.title("Controls")
st.sidebar.caption(
    "Upload a **test** CSV containing the 27 predictors *and* the true `Fault` label. "
    "The models were trained on a disjoint 70% split and are only scored here."
)

upload = st.sidebar.file_uploader("Test data (CSV)", type=["csv"])
use_bundled = st.sidebar.toggle(
    "Use the repository's test_data.csv", value=upload is None,
    help="The 583-row hold-out split produced by src/make_split.py.",
)

chosen = st.sidebar.selectbox(
    "Classifier", list(MODEL_MAP.keys()),
    index=list(MODEL_MAP.keys()).index(meta.get("winner", list(MODEL_MAP)[0])),
)

with st.sidebar.expander("Provenance"):
    st.write(f"**Trained with** scikit-learn {meta['sklearn_version']}, Python {meta['python_version']}")
    st.write(f"**Running** scikit-learn {sklearn.__version__}, Python {sys.version.split()[0]}")
    if meta["sklearn_version"] != sklearn.__version__:
        st.warning("scikit-learn version differs from training; metrics may shift.")
    else:
        st.success("Library versions match training.")
    st.write(f"Seed `{meta['random_state']}` · test size `{meta['test_size']}` · KNN k=`{meta['knn_k']}`")

# --------------------------------------------------------------------------- data in
st.title("🔩 Steel Plate Fault Classification")
st.caption(
    f"Five classifiers over {len(meta['feature_order'])} numeric predictors and "
    f"{len(CLASSES)} mutually-exclusive fault classes · UCI Steel Plates Faults (id 198)"
)

if upload is not None:
    df = read_csv_bytes(upload.getvalue())
    source = f"uploaded file `{upload.name}`"
elif use_bundled and BUNDLED_TEST.exists():
    df = read_csv_bytes(BUNDLED_TEST.read_bytes())
    source = "bundled `test_data.csv`"
else:
    st.info("Upload a test CSV from the sidebar, or switch on the bundled test set.")
    st.stop()

problems = check_schema(df, meta)
blocking = [p for p in problems if not p.startswith(("0 unexpected", "1 unexpected"))
            and "will be ignored" not in p]
for p in problems:
    (st.warning if "will be ignored" in p else st.error)(p)
if blocking:
    st.caption("Expected header:")
    st.code(",".join(meta["feature_order"] + [meta["target_column"]]), language="text")
    st.stop()

X = df[meta["feature_order"]]
y = df[meta["target_column"]]
st.success(f"Scoring **{len(df):,} rows** from {source} · {y.nunique()} of {len(CLASSES)} classes present")

# --------------------------------------------------------------------------- scoring
tab_single, tab_all, tab_data = st.tabs(
    ["Selected model", "Compare all five", "Data preview"]
)

with tab_single:
    model, version_note = load_model(MODEL_MAP[chosen])
    if version_note:
        st.warning(f"Pickle/library version mismatch: {version_note}")

    try:
        scores = compute_metrics(model, X, y, CLASSES)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Scoring failed: {exc}")
        with st.expander("Traceback"):
            st.exception(exc)
        st.stop()

    st.subheader(chosen)
    # Two rows of three rather than one row of six: at six columns Streamlit
    # truncates the value to "0.7..." on anything narrower than a wide desktop.
    for row_start in (0, 3):
        for col, name in zip(st.columns(3), METRIC_ORDER[row_start:row_start + 3]):
            col.metric(name, format_metric(scores[name]))
    if scores["AUC"] is None:
        st.caption(
            "AUC is unavailable: one-vs-rest AUC needs at least two classes present "
            "in the uploaded labels."
        )

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**Confusion matrix** — rows are true, columns predicted")
        cm = confusion_frame(model, X, y, CLASSES)
        st.dataframe(cm.style.background_gradient(cmap="Blues", axis=None),
                     use_container_width=True)
    with right:
        st.markdown("**Classification report**")
        rep = classification_report(
            y, model.predict(X), labels=CLASSES, output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(rep).T.round(4), use_container_width=True)

    abandoned = [c for c in CLASSES if cm.loc[f"true:{c}"].sum() and not cm[f"pred:{c}"].sum()]
    if abandoned:
        st.info(
            f"**{chosen} never predicts {', '.join(abandoned)}.** Their precision and "
            "recall are reported as 0, which is a genuine result rather than a bug — "
            "the macro averages are pulled down accordingly."
        )

with tab_all:
    st.subheader("All five classifiers on this data")
    rows = []
    for label, fname in MODEL_MAP.items():
        mdl, _ = load_model(fname)
        s = compute_metrics(mdl, X, y, CLASSES)
        rows.append({"ML Model Name": label, **{m: s[m] for m in METRIC_ORDER}})
    comp = pd.DataFrame(rows)

    best = comp.sort_values(["F1", "MCC", "AUC"], ascending=False).iloc[0]["ML Model Name"]
    st.dataframe(
        comp.style.format({m: lambda v: format_metric(v) for m in METRIC_ORDER})
            .highlight_max(subset=METRIC_ORDER, color="#d6f5d6"),
        use_container_width=True, hide_index=True,
    )
    st.success(
        f"**Winner on this data: {best}** — ranked by macro-F1, tie-broken on MCC "
        f"then one-vs-rest macro AUC. The rule was fixed before the numbers were seen."
    )
    chart = comp.set_index("ML Model Name")[["Accuracy", "F1", "MCC"]]
    st.bar_chart(chart, height=320)
    st.caption(
        "Accuracy sits well above MCC for every model: with a 12.2:1 class imbalance, "
        "accuracy is flattered by the majority `Other_Faults` class while MCC accounts "
        "for all four cells of every class's confusion matrix."
    )

with tab_data:
    st.subheader("Uploaded data")
    st.dataframe(df.head(50), use_container_width=True, hide_index=True)
    st.markdown("**Class distribution**")
    st.bar_chart(y.value_counts().reindex(CLASSES).fillna(0), height=260)
