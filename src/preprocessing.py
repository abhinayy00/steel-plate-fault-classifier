"""The shared feature transform.

One preprocessor serves all five classifiers so the comparison is strictly
like-for-like. Scaling is mandatory for KNN, SVM and Logistic Regression (they are
distance- or penalty-based), and provably harmless for the other two: a decision
tree is invariant to any monotone rescaling of a feature, and GaussianNB estimates
a separate mean and variance per feature, so a linear rescale cancels out.

Giving the tree and NB an unscaled branch would have meant each model seeing a
different matrix, which is a worse experiment for the sake of nothing.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessor(feature_names):
    """ColumnTransformer over the 27 numeric predictors.

    The imputer is defensive rather than necessary: the training data has no missing
    cells, but an uploaded test CSV may contain blanks, and a NaN reaching an
    estimator raises rather than degrading gracefully.
    """
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("numeric", numeric, list(feature_names))],
        remainder="drop",
        verbose_feature_names_out=False,
    )
