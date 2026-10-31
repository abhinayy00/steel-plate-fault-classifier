"""Fetch the Steel Plates Faults data, collapse the target, and write the splits.

The UCI file ships the label as SEVEN mutually-exclusive indicator columns. Leaving
even one of them inside X makes every classifier trivially perfect, so the collapse
and the leakage assertion below are the load-bearing part of this script.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UCI_ID = 198

# The seven indicator columns that together encode the label.
FAULT_COLUMNS = [
    "Pastry",
    "Z_Scratch",
    "K_Scratch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
]
TARGET = "Fault"

# Derived from the BITS ID (2026AJ05465) rather than the conventional 42, so the
# split - and therefore every number in the report - is specific to this submission.
SEED = 5465

# 70/30 rather than the usual 80/20: it puts roughly 16 Dirtiness and 22 Stains rows
# in the test set instead of ~11 and ~15, which makes the macro-averaged metrics
# meaningfully less noisy on the two rarest classes.
TEST_SIZE = 0.30


def load_raw():
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=UCI_ID)
    X = ds.data.features.copy()
    targets = ds.data.targets.copy()
    return X, targets


def collapse_target(targets):
    missing = set(FAULT_COLUMNS) - set(targets.columns)
    if missing:
        raise ValueError(f"expected fault columns absent from UCI payload: {sorted(missing)}")
    row_sums = targets[FAULT_COLUMNS].sum(axis=1)
    if not (row_sums == 1).all():
        raise ValueError(
            "fault indicator columns are not mutually exclusive; idxmax collapse would "
            f"be invalid (row sums range {row_sums.min()}..{row_sums.max()})"
        )
    return targets[FAULT_COLUMNS].idxmax(axis=1).rename(TARGET)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    X, targets = load_raw()
    y = collapse_target(targets)

    # The single assertion that protects the whole project from a worthless result.
    leaked = set(FAULT_COLUMNS) & set(X.columns)
    if leaked:
        raise AssertionError(f"target leakage: fault columns present in X: {sorted(leaked)}")

    # Prove the assignment's hard constraints from the data rather than asserting them
    # in prose.
    n_rows, n_features = X.shape
    assert n_features >= 12, f"need >=12 predictors, found {n_features}"
    assert n_rows >= 500, f"need >=500 instances, found {n_rows}"

    raw_path = DATA / "steel_plates_faults_raw.csv"
    pd.concat([X, targets[FAULT_COLUMNS]], axis=1).to_csv(raw_path, index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # Every class must survive into the test split, otherwise multi-class AUC cannot
    # be computed on the file the grader uploads.
    assert set(y_test.unique()) == set(y.unique()), "a class vanished from the test split"
    assert set(X_train.index).isdisjoint(X_test.index), "train/test index overlap"

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_df.to_csv(DATA / "train_data.csv", index=False)
    # test_data.csv lives at the repo root because the assignment names it there.
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    meta = {
        "uci_id": UCI_ID,
        "source_url": "https://archive.ics.uci.edu/dataset/198/steel+plates+faults",
        "doi": "10.24432/C5J88N",
        "n_instances": int(n_rows),
        "n_predictors": int(n_features),
        "feature_order": list(X.columns),
        "target_column": TARGET,
        "classes": sorted(y.unique()),
        "class_counts": {k: int(v) for k, v in y.value_counts().items()},
        "random_state": SEED,
        "test_size": TEST_SIZE,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }
    (DATA / "split_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"instances      : {n_rows}")
    print(f"predictors     : {n_features}  (>=12 required)")
    print(f"classes        : {len(meta['classes'])} -> {meta['classes']}")
    print(f"train / test   : {len(train_df)} / {len(test_df)}  (stratified, seed={SEED})")
    print("\ntest-split class counts:")
    for k, v in y_test.value_counts().items():
        print(f"   {k:<14} {v:4d}")
    print(f"\nwrote {DATA/'train_data.csv'}")
    print(f"wrote {ROOT/'test_data.csv'}")


if __name__ == "__main__":
    main()
