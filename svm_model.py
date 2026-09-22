#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Linear SVM classifier for stibnite-based deposit type classification.

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
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC

MODEL_ID = "Genetic4_Stable6Q75_LinearSVM_Core17_v1.0"

FEATURES = ["Te", "Cu", "Pb", "Zn", "As", "Ge"]

CLASSES = ["G1", "G2", "G3", "G4"]

CORE17 = {
    "Maxiong": "G1",
    "Woxi": "G1",
    "Yangshan": "G1",
    "Frassine": "G2",
    "La Campigliola": "G2",
    "Le Cetine di Cotorniano": "G2",
    "Micciano": "G2",
    "Poggio Peloso-Pietratonda": "G2",
    "Zhaxikang": "G2",
    "Gutaishan": "G3",
    "Jianzhupo": "G3",
    "Lianzhushan": "G3",
    "Xiaodong": "G3",
    "Yangla": "G3",
    "Banxi": "G4",
    "Xikuangshan": "G4",
    "Daocaowan": "G4",
}


def build_deposit_q75(df: pd.DataFrame) -> pd.DataFrame:
    """Compute deposit P75 values of log10-transformed positive concentrations."""
    rows = []
    for dep, label in CORE17.items():
        sub = df[df["Deposit"] == dep]
        if sub.empty:
            raise ValueError(f"Core-17 deposit missing from input: {dep}")
        row = {"Deposit": dep, "Class": label, "N_spots": int(len(sub))}
        for element in FEATURES:
            x = pd.to_numeric(sub[element], errors="coerce").to_numpy(dtype=float)
            x = x[np.isfinite(x) & (x > 0)]
            row[element] = (
                float(np.quantile(np.log10(x), 0.75, method="linear"))
                if len(x)
                else np.nan
            )
            row[element + "_N_positive"] = int(len(x))
        rows.append(row)
    return pd.DataFrame(rows)


def fit_fold(train_df: pd.DataFrame):
    """Fit training-set imputation, robust scaling, and the fixed linear SVM."""
    X = train_df[FEATURES].to_numpy(dtype=float)
    y = train_df["Class"].to_numpy()

    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isnan(X), med, X)

    scaler = RobustScaler().fit(X)
    Xs = scaler.transform(X)

    clf = LinearSVC(
        penalty="l2",
        loss="squared_hinge",
        dual="auto",
        tol=1e-4,
        C=1.0,
        multi_class="ovr",
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        verbose=0,
        random_state=20260814,
        max_iter=1000,
    )
    clf.fit(Xs, y)
    return med, scaler, clf


def predict_one(test_df: pd.DataFrame, med, scaler, clf):
    """Return the first deposit prediction, decision scores, and top-two margin."""
    X = test_df[FEATURES].to_numpy(dtype=float)
    X = np.where(np.isnan(X), med, X)
    Xs = scaler.transform(X)
    pred = str(clf.predict(Xs)[0])
    scores = clf.decision_function(Xs)[0]
    score_map = {str(c): float(scores[i]) for i, c in enumerate(clf.classes_)}
    ordered = sorted(score_map.items(), key=lambda z: z[1], reverse=True)
    return pred, score_map, float(ordered[0][1] - ordered[1][1])


def run_lodo(deposit_df: pd.DataFrame):
    """Evaluate the fixed SVM by leaving out each complete reference deposit."""
    out = []
    for held in deposit_df["Deposit"].tolist():
        train = deposit_df[deposit_df["Deposit"] != held].copy()
        test = deposit_df[deposit_df["Deposit"] == held].copy()
        med, scaler, clf = fit_fold(train)
        pred, scores, top_margin = predict_one(test, med, scaler, clf)

        true = str(test["Class"].iloc[0])
        true_score = scores[true]
        best_other = max(v for c, v in scores.items() if c != true)

        out.append(
            {
                "Deposit": held,
                "True": true,
                "Predicted": pred,
                "Correct": int(pred == true),
                "Top1_margin": top_margin,
                "True_minus_best_other": float(true_score - best_other),
                **{f"Score_{c}": scores[c] for c in CLASSES},
            }
        )
    return pd.DataFrame(out)


def summarize(pred_df: pd.DataFrame):
    """Calculate deposit-level metrics and list misclassified deposits."""
    yt = pred_df["True"].to_numpy()
    yp = pred_df["Predicted"].to_numpy()

    acc = float(accuracy_score(yt, yp))
    ba = float(balanced_accuracy_score(yt, yp))
    mf1 = float(f1_score(yt, yp, labels=CLASSES, average="macro", zero_division=0))
    rec = recall_score(yt, yp, labels=CLASSES, average=None, zero_division=0)
    prec = precision_score(yt, yp, labels=CLASSES, average=None, zero_division=0)
    f1 = f1_score(yt, yp, labels=CLASSES, average=None, zero_division=0)
    cm = confusion_matrix(yt, yp, labels=CLASSES)

    return {
        "model_id": MODEL_ID,
        "n_deposits": int(len(pred_df)),
        "correct": int(pred_df["Correct"].sum()),
        "accuracy": acc,
        "balanced_accuracy": ba,
        "macro_f1": mf1,
        "per_class_precision": {c: float(v) for c, v in zip(CLASSES, prec)},
        "per_class_recall": {c: float(v) for c, v in zip(CLASSES, rec)},
        "per_class_f1": {c: float(v) for c, v in zip(CLASSES, f1)},
        "confusion_matrix_order": CLASSES,
        "confusion_matrix": cm.astype(int).tolist(),
        "errors": pred_df.loc[
            pred_df["Correct"] == 0, ["Deposit", "True", "Predicted"]
        ].to_dict("records"),
    }


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
    """Validate SVM reference membership and count, then build P75 features."""
    points, audit = read_points(path, FEATURES)
    if len(points) != 941 or set(points.Deposit) != set(CORE17):
        raise ValueError(
            "SVM requires the complete Core17 snapshot: 17 deposits, 941 spots."
        )
    q75 = build_deposit_q75(points)
    require_training_coverage(q75, FEATURES)
    return q75, audit


def evaluate(path, directory):
    """Write LODO predictions, performance metrics, and input-status records."""
    q75, audit = load_reference(path)
    for held in CORE17:
        require_training_coverage(q75.loc[q75.Deposit.ne(held)], FEATURES)
    predictions = run_lodo(q75)
    summary = summarize(predictions)
    directory.mkdir(parents=True, exist_ok=True)
    q75.to_csv(directory / "svm_p75.csv", index=False)
    predictions.to_csv(directory / "svm_lodo.csv", index=False)
    write_json(directory / "svm_metrics.json", summary)
    write_run_info(directory, path, audit)
    print(json.dumps(summary, indent=2))
    return predictions, summary


def fit_reference(path, output):
    """Fit the full SVM reference set and save the model with preprocessing."""
    q75, _ = load_reference(path)
    medians, scaler, classifier = fit_fold(q75)
    bundle = {
        "model": classifier,
        "training_medians": medians,
        "scaler": scaler,
        "features": FEATURES,
        "task": "deposit_type",
        "training_deposits": CORE17,
        "environment": environment(),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    print(f"Saved SVM fitted on {len(CORE17)} deposit vectors: {output}")


def predict(path, model_path, output, unit_col="Unit"):
    """Write one SVM deposit-type prediction per input unit."""
    points, _ = read_points(path, FEATURES, unit_col=unit_col)
    require_unit_coverage(points, FEATURES, unit_col=unit_col)
    bundle = load_fitted_model(model_path, "deposit_type")
    medians, scaler, classifier = (
        bundle["training_medians"],
        bundle["scaler"],
        bundle["model"],
    )
    rows = []
    for unit, sub in points.groupby(unit_col, sort=False):
        row = {"Unit": unit, "N_spots": len(sub)}
        q75 = []
        for feature in FEATURES:
            values = sub[feature].to_numpy(float)
            values = values[np.isfinite(values) & (values > 0)]
            row[feature + "_N_positive"] = len(values)
            value = (
                float(np.quantile(np.log10(values), 0.75, method="linear"))
                if len(values)
                else np.nan
            )
            row[feature + "_P75_log10"] = value
            q75.append(value)
        x = np.asarray([q75])
        x = np.where(np.isnan(x), medians, x)
        transformed = scaler.transform(x)
        scores = classifier.decision_function(transformed)[0]
        row["Predicted_Deposit_Type"] = str(classifier.predict(transformed)[0])
        row.update(
            {
                "Score_" + str(c): float(scores[i])
                for i, c in enumerate(classifier.classes_)
            }
        )
        ordered = np.sort(scores)
        row["Top1_Margin"] = float(ordered[-1] - ordered[-2])
        rows.append(row)
    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Predicted {len(result)} units: {output}")
    return result


if __name__ == "__main__":
    parser = make_parser("Fixed SVM deposit-type classifier; no tuning.", "svm")
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
