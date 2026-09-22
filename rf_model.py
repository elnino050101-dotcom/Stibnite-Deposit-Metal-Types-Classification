#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random forest classifier for stibnite-based metal type classification.

Article: Discrimination of deposit and metal types based on stibnite trace
    elements: A machine learning method
Journal: Natural Resources Research
Authors: Yuan-Xin Wu (1, 2), Xiao-Wen Huang (1, 3), Jia-Hao Zheng (2),
    Yu-Miao Meng (1)

Affiliations:
    1. State Key Laboratory for Critical Mineral Research and Exploration,
       Institute of Geochemistry, Chinese Academy of Sciences,
       Guiyang 550081, China.
    2. Zijin School of Geology and Mining, Fuzhou University,
       Fuzhou 350108, China.
    3. Critical Mineral Artificial Intelligence Research Center,
       Institute of Geochemistry, Chinese Academy of Sciences,
       Guiyang 550081, China.

Corresponding author: Xiao-Wen Huang
E-mail: huangxiaowen@vip.gyig.ac.cn
Original script version: 2.0 (16 September 2026)

Commands:
    lodo     Evaluate the fixed model by leave-one-deposit-out validation.
    fit      Fit the model to the study's reference dataset.
    predict  Apply a fitted model to new spot analyses, grouped by unit.

Input concentrations must be in ppm; logarithmic transformation is internal.
Reference datasets are required for lodo and fit and are supplied separately.
Model settings and reference labels are fixed; no parameter search is run.
Use --help or README.md for input requirements, commands, and output details.
README.pdf contains the manuscript title page and author information.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)

RF_PARAMS = {
    "n_estimators": 300,
    "criterion": "gini",
    "max_depth": 4,
    "min_samples_split": 2,
    "min_samples_leaf": 8,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 1.0,
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "bootstrap": True,
    "oob_score": False,
    "n_jobs": 1,
    "verbose": 0,
    "warm_start": False,
    "class_weight": None,
    "ccp_alpha": 0.0,
    "max_samples": None,
    "monotonic_cst": None,
}

FEATURES = ["Cu", "Zn", "As", "Se", "Ag", "Sn", "Pb"]

CLASSES = ["Sb", "Au", "Au-Sb"]

TRAIN_LABELS = {
    "Chashan": "Sb",
    "Daocaowan": "Sb",
    "Frassine": "Sb",
    "Gerakario": "Au",
    "Gutaishan": "Au-Sb",
    "Jianzhupo": "Sb",
    "La Campigliola": "Sb",
    "Le Cetine di Cotorniano": "Sb",
    "Manaoke": "Au",
    "Maxiong": "Sb",
    "Micciano": "Sb",
    "Pezinok-Kolársky vrch": "Au-Sb",
    "Pezinok-Trojárová": "Au-Sb",
    "Poggio Peloso-Pietratonda": "Sb",
    "Woxi": "Au-Sb",
    "Xiaodong": "Sb",
    "Xikuangshan": "Sb",
    "Yangshan": "Au",
    "Zhaxikang": "Sb",
    "Zhazixi": "Sb",
}

FORMAL_RANDOM_STATE = 42

TAIL_THRESHOLD = 0.70


def transform_train_test(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log-transform cleaned concentrations and impute from training medians.

    The same feature medians in log10 space are applied to training and test data.
    """
    Xtr = np.log10(train_df[FEATURES].to_numpy(dtype=float))
    Xte = np.log10(test_df[FEATURES].to_numpy(dtype=float))

    medians = np.nanmedian(Xtr, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)

    Xtr = np.where(np.isfinite(Xtr), Xtr, medians)
    Xte = np.where(np.isfinite(Xte), Xte, medians)
    return Xtr, Xte, medians


def deposit_equal_weights(train_df: pd.DataFrame) -> np.ndarray:
    """Balance class, deposit, and spot contributions in the training set.

    Classes receive equal total weight, deposits within a class share that weight
    equally, and spots within each deposit receive equal fractions.
    """
    dep_class = {d: TRAIN_LABELS[d] for d in train_df["Deposit"].unique()}
    class_deposits = defaultdict(list)
    for dep, cls in dep_class.items():
        class_deposits[cls].append(dep)

    dep_counts = train_df["Deposit"].value_counts().to_dict()

    weights = []
    for dep in train_df["Deposit"]:
        cls = dep_class[dep]
        w = 1.0 / len(class_deposits[cls]) / dep_counts[dep]
        weights.append(w)
    return np.asarray(weights, dtype=float)


def fit_rf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int = FORMAL_RANDOM_STATE,
) -> tuple[RandomForestClassifier, np.ndarray, np.ndarray]:
    """Fit the fixed RF and return test probabilities and imputation medians."""
    Xtr, Xte, medians = transform_train_test(train_df, test_df)
    ytr = train_df["Deposit"].map(TRAIN_LABELS).to_numpy()
    weights = deposit_equal_weights(train_df)

    clf = RandomForestClassifier(random_state=random_state, **RF_PARAMS)
    clf.fit(Xtr, ytr, sample_weight=weights)
    spot_prob = clf.predict_proba(Xte)
    return clf, spot_prob, medians


def metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Calculate deposit-level classification metrics in the fixed class order."""
    recalls = recall_score(
        y_true, y_pred, labels=CLASSES, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
        ),
        "recall_Sb": float(recalls[0]),
        "recall_Au": float(recalls[1]),
        "recall_Au-Sb": float(recalls[2]),
        "confusion_matrix_labels": CLASSES,
        "confusion_matrix": cm.tolist(),
    }


def aggregate_with_tail(clf, spot_prob):
    """Average spot probabilities and apply the fixed Sb-to-Au-Sb correction."""
    mean_prob = spot_prob.mean(axis=0)
    pmap = {c: float(p) for c, p in zip(clf.classes_, mean_prob)}
    means = {c: pmap.get(c, 0.0) for c in CLASSES}
    au_sb_idx = list(clf.classes_).index("Au-Sb")
    q90_au_sb = float(np.quantile(spot_prob[:, au_sb_idx], 0.90, method="linear"))
    direct = max(means, key=means.get)
    final = "Au-Sb" if (direct == "Sb" and q90_au_sb >= TAIL_THRESHOLD) else direct
    correction = "Sb->Au-Sb" if final != direct else "None"
    return means, direct, q90_au_sb, correction, final


def final_lodo(train_df):
    """Evaluate the fixed RF by leaving out each complete reference deposit."""
    rows = []
    for test_dep in sorted(TRAIN_LABELS):
        fold_train = train_df.loc[train_df["Deposit"].ne(test_dep)].copy()
        fold_test = train_df.loc[train_df["Deposit"].eq(test_dep)].copy()
        clf, spot_prob, _ = fit_rf(
            fold_train, fold_test, random_state=FORMAL_RANDOM_STATE
        )
        p, direct, q90, corr, final = aggregate_with_tail(clf, spot_prob)
        rows.append(
            {
                "Deposit": test_dep,
                "TrueClass": TRAIN_LABELS[test_dep],
                "n_spots": len(fold_test),
                "P_Sb_mean": p["Sb"],
                "P_Au_mean": p["Au"],
                "P_Au-Sb_mean": p["Au-Sb"],
                "Direct_RF": direct,
                "Q90_P_Au-Sb": q90,
                "Correction": corr,
                "Final_Model_A": final,
                "Correct": final == TRAIN_LABELS[test_dep],
            }
        )
    df = pd.DataFrame(rows)
    return df, metrics(df["TrueClass"].tolist(), df["Final_Model_A"].tolist())


def canonical_name(value):
    """Trim whitespace and normalize en and em dashes to hyphens."""
    return str(value).strip().replace("–", "-").replace("—", "-")


def read_points(path, features, unit_col="Deposit"):
    """Read a spot CSV and preserve source tokens in a long-format audit.

    Only finite, strictly positive concentrations enter the numeric model table.
    Zeros and other ineligible values become NaN while retaining distinct audit
    statuses. Blank cells remain unspecified; their analytical origin is not inferred.
    """
    raw = pd.read_csv(
        Path(path), dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    missing = [c for c in [unit_col] + features if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if raw.empty or raw[unit_col].str.strip().eq("").any():
        raise ValueError("Input must contain rows with nonempty unit names.")
    raw[unit_col] = raw[unit_col].map(canonical_name)
    if "Point_ID" in raw and raw["Point_ID"].duplicated().any():
        raise ValueError("Point_ID must be unique.")
    result = raw.copy()
    audit = []
    for element in features:
        values = []
        for row_number, token in enumerate(raw[element], start=1):
            text = token.strip()
            upper = text.upper()
            value = np.nan
            if not text:
                status = "blank/unspecified"
            elif text.startswith(("<", "≤")) or upper in {
                "BDL",
                "DL",
                "ND",
                "N.D.",
                "B.D.L.",
            }:
                status = "below_detection_or_nondetect"
            elif upper in {"NOT ANALYZED", "NOT ANALYSED", "NOT MEASURED"}:
                status = "not_analyzed"
            elif upper in {"NA", "N/A", "NAN", "NULL", "NONE", "-", "--"}:
                status = "missing_token"
            else:
                try:
                    number = float(text)
                    if not np.isfinite(number):
                        status = "nonfinite"
                    elif number > 0:
                        status, value = "positive", number
                    elif number == 0:
                        status = "zero"
                    else:
                        status = "negative"
                except ValueError:
                    status = "unrecognized_text"
            values.append(value)
            audit.append(
                {
                    "Source_Row": row_number,
                    "Unit": raw.iloc[row_number - 1][unit_col],
                    "Element": element,
                    "Source_Token": token,
                    "Analytical_Status": status,
                }
            )
        result[element] = np.asarray(values, dtype=float)
    return result, pd.DataFrame(audit)


def require_training_coverage(table, features):
    """Reject predictors without finite training observations before fitting.

    Inputs are cleaned before this check, so finite values represent eligible
    positive concentrations or their deposit-level log10 summaries.
    """
    missing = [f for f in features if not np.isfinite(table[f].to_numpy(float)).any()]
    if missing:
        raise ValueError(f"No training observations for {missing}.")


def require_unit_coverage(table, features, unit_col="Unit"):
    """Require at least one eligible predictor value per prediction unit."""
    for unit, sub in table.groupby(unit_col, sort=False):
        if not np.isfinite(sub[features].to_numpy(float)).any():
            raise ValueError(f"Unit {unit!r} has no eligible positive predictor value.")


def environment():
    """Return Python and dependency versions for reproducibility records."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def write_json(path, value):
    """Write an indented UTF-8 JSON file, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_run_info(directory, input_path, audit):
    """Save the input hash, software versions, and analytical-status audit."""
    write_json(
        directory / "run_info.json",
        {
            "input_file": str(input_path),
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "environment": environment(),
            "evaluation": "Fixed configuration after development; selection is not nested.",
        },
    )
    audit.to_csv(directory / "input_status_audit.csv", index=False)


def load_fitted_model(path, task):
    """Load a trusted local model and check its task and saved configuration.

    Use models produced by this script's fit command. The scikit-learn version
    must match the fitting environment; joblib files must come from a trusted source.
    """
    bundle = joblib.load(path)
    if bundle.get("task") != task or bundle.get("features") != FEATURES:
        raise ValueError(
            "Saved model task or feature order does not match this script."
        )
    if bundle.get("environment", {}).get("scikit-learn") != sklearn.__version__:
        raise ValueError("Use the fitting environment, or recreate the model with fit.")
    if set(bundle["model"].classes_) != set(CLASSES):
        raise ValueError("Saved model classes do not match this script.")
    medians = np.asarray(bundle["training_medians"], dtype=float)
    if medians.shape != (len(FEATURES),) or not np.isfinite(medians).all():
        raise ValueError("Invalid saved imputation parameters.")
    return bundle


def make_parser(description, default_prefix):
    """Build the lodo, fit, and predict command-line interfaces."""
    parser = argparse.ArgumentParser(description=description)
    commands = parser.add_subparsers(dest="command", required=True)
    lodo = commands.add_parser(
        "lodo", help="Evaluate the fixed reference configuration."
    )
    lodo.add_argument(
        "--input", type=Path, required=True, help="Reference spot CSV, in ppm."
    )
    lodo.add_argument("--outdir", type=Path, default=Path(default_prefix + "_lodo"))
    fit = commands.add_parser(
        "fit", help="Fit the fixed model to its full reference set."
    )
    fit.add_argument(
        "--input", type=Path, required=True, help="Reference spot CSV, in ppm."
    )
    fit.add_argument(
        "--output", type=Path, default=Path(default_prefix + "_fitted.joblib")
    )
    predict = commands.add_parser(
        "predict", help="Apply saved preprocessing and classifier."
    )
    predict.add_argument(
        "--input", type=Path, required=True, help="New spot CSV, in ppm."
    )
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument(
        "--output", type=Path, default=Path(default_prefix + "_predictions.csv")
    )
    predict.add_argument(
        "--unit-col", default="Unit", help="Grouping column; default: Unit."
    )
    return parser


def load_reference(path):
    """Validate reference rows and select the fixed RF training dataset."""
    points, audit = read_points(path, FEATURES)
    required = ["Frozen_Row", "Keep_after_score<=4"]
    if any(c not in points for c in required):
        raise ValueError(
            "RF reference input requires Frozen_Row and Keep_after_score<=4."
        )
    order = pd.to_numeric(points["Frozen_Row"], errors="coerce")
    if (
        order.isna().any()
        or not np.isfinite(order).all()
        or (order <= 0).any()
        or (order % 1 != 0).any()
        or order.duplicated().any()
        or not order.is_monotonic_increasing
    ):
        raise ValueError("Preserve the original increasing, unique Frozen_Row order.")
    flag = points["Keep_after_score<=4"]
    if not flag.isin(["Yes", "No"]).all():
        raise ValueError("The frozen QC flag must be Yes or No; it is not recomputed.")
    train = points.loc[flag.eq("Yes") & points.Deposit.isin(TRAIN_LABELS)].copy()
    if len(train) != 768 or set(train.Deposit) != set(TRAIN_LABELS):
        raise ValueError(
            "RF requires the frozen TRAIN20 membership: 20 deposits, 768 spots."
        )
    require_training_coverage(train, FEATURES)
    return train, audit


def evaluate(path, directory):
    """Write LODO predictions, performance metrics, and input-status records."""
    train, audit = load_reference(path)
    for held in TRAIN_LABELS:
        require_training_coverage(train.loc[train.Deposit.ne(held)], FEATURES)
    predictions, final_metrics = final_lodo(train)
    direct_metrics = metrics(
        predictions.TrueClass.tolist(), predictions.Direct_RF.tolist()
    )
    final_metrics["correct"] = int(predictions.Correct.sum())
    direct_metrics["correct"] = int(
        predictions.TrueClass.eq(predictions.Direct_RF).sum()
    )
    summary = {
        "n_deposits": len(predictions),
        "n_spots": len(train),
        "direct_LODO": direct_metrics,
        "final_LODO": final_metrics,
    }
    directory.mkdir(parents=True, exist_ok=True)
    predictions.rename(columns={"Final_Model_A": "Final_Metal_Type"}).to_csv(
        directory / "rf_lodo.csv", index=False
    )
    write_json(directory / "rf_metrics.json", summary)
    write_run_info(directory, path, audit)
    print(json.dumps(summary, indent=2))
    return predictions, summary


def fit_reference(path, output):
    """Fit the full RF reference set and save the model with preprocessing."""
    train, _ = load_reference(path)
    # No performance claim is made from this reference-data fit.
    classifier, _, medians = fit_rf(train, train)
    bundle = {
        "model": classifier,
        "training_medians": medians,
        "features": FEATURES,
        "task": "metal_type",
        "tail_threshold": TAIL_THRESHOLD,
        "training_deposits": TRAIN_LABELS,
        "environment": environment(),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    print(
        f"Saved RF fitted on {len(TRAIN_LABELS)} deposits / {len(train)} spots: {output}"
    )


def predict(path, model_path, output, unit_col="Unit"):
    """Write one RF metal-type prediction per input unit."""
    points, _ = read_points(path, FEATURES, unit_col=unit_col)
    require_unit_coverage(points, FEATURES, unit_col=unit_col)
    bundle = load_fitted_model(model_path, "metal_type")
    if bundle.get("tail_threshold") != TAIL_THRESHOLD:
        raise ValueError("Saved tail threshold differs from the fixed manuscript rule.")
    classifier, medians = bundle["model"], bundle["training_medians"]
    rows = []
    for unit, sub in points.groupby(unit_col, sort=False):
        x = np.log10(sub[FEATURES].to_numpy(float))
        x = np.where(np.isfinite(x), x, medians)
        probabilities = classifier.predict_proba(x)
        means, direct, q90, correction, final = aggregate_with_tail(
            classifier, probabilities
        )
        rows.append(
            {
                "Unit": unit,
                "N_spots": len(sub),
                **{f + "_N_positive": int(sub[f].notna().sum()) for f in FEATURES},
                **{"P_" + c + "_mean": means[c] for c in CLASSES},
                "Direct_RF": direct,
                "Q90_P_Au-Sb": q90,
                "Correction": correction,
                "Final_Metal_Type": final,
            }
        )
    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Predicted {len(result)} units: {output}")
    return result


if __name__ == "__main__":
    parser = make_parser("Fixed RF metal-type classifier; no tuning.", "rf")
    args = parser.parse_args()
    try:
        if args.command == "lodo":
            evaluate(args.input, args.outdir)
        elif args.command == "fit":
            fit_reference(args.input, args.output)
        else:
            predict(args.input, args.model, args.output, args.unit_col)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")
