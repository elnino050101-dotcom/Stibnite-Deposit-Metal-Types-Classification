# Stibnite-Based Deposit and Metal Type Classification

Python code accompanying **Discrimination of deposit and metal types based on stibnite trace elements: A machine learning method**.

- **`svm_model.py`**: linear support vector machine (SVM) for deposit type classification.
- **`rf_model.py`**: random forest (RF) for metal type classification (`Sb`, `Au`, `Au-Sb`).

Both scripts provide evaluation, fitting, and prediction using the study's fixed model settings. See [README.pdf](README.pdf) for the user guide and author information.

## Installation

Download and extract the repository using **Code > Download ZIP**, then open a terminal in the extracted folder.

The environment reported in the user guide uses Python **3.12.14** and the following dependencies:

```bash
python -m pip install numpy==2.3.5 pandas==2.2.3 scipy==1.17.0 scikit-learn==1.8.0 joblib==1.5.3
```

Use the same environment for fitting and prediction.

## Input data

**Reference CSV files and fitted models are not included in this repository.** Obtain the study's reference data before running `lodo` or `fit`. For data-access questions, contact the corresponding author listed in [README.pdf](README.pdf).

Inputs must be UTF-8 CSV files with one spot analysis per row and original concentrations in **ppm**. Use the exact column names below. The scripts perform log10 transformation internally. Non-positive values, blanks, and below-detection entries are treated as missing; preserve their reported status rather than replacing them with zeros. Optional `Point_ID` values must be unique.

| Input | Required columns |
| --- | --- |
| RF reference: `rf_points.csv` | `Deposit`, `Cu`, `Zn`, `As`, `Se`, `Ag`, `Sn`, `Pb`, `Frozen_Row`, `Keep_after_score<=4` |
| SVM reference: `svm_points.csv` | `Deposit`, `Te`, `Cu`, `Pb`, `Zn`, `As`, `Ge` |
| RF prediction: `new_rf_points.csv` | `Unit`, `Cu`, `Zn`, `As`, `Se`, `Ag`, `Sn`, `Pb` |
| SVM prediction: `new_svm_points.csv` | `Unit`, `Te`, `Cu`, `Pb`, `Zn`, `As`, `Ge` |

For RF reference data, preserve the original unique, increasing `Frozen_Row` identifiers. The `Keep_after_score<=4` flag must be `Yes` or `No`; retained rows must contain **768 analyses from 20 reference deposits**. SVM requires **941 analyses from its 17 reference deposits**. Deposit names must match the corresponding reference mappings in the scripts. These fitting commands require the study's reference datasets.

For prediction, `Unit` groups spots from each deposit or defined mineral population. Each unit receives one prediction and must contain at least one positive predictor value. If the grouping column is named `Deposit`, append `--unit-col Deposit` to the prediction command.

## Usage

Place the CSV files in the script folder or replace the example filenames with their actual paths. `lodo` runs leave-one-deposit-out evaluation; `fit` creates a model that is then used by `predict`.

### RF: metal type

```bash
python rf_model.py lodo --input rf_points.csv
python rf_model.py fit --input rf_points.csv
python rf_model.py predict --input new_rf_points.csv --model rf_fitted.joblib
```

### SVM: deposit type

```bash
python svm_model.py lodo --input svm_points.csv
python svm_model.py fit --input svm_points.csv
python svm_model.py predict --input new_svm_points.csv --model svm_fitted.joblib
```

Use `--outdir` with `lodo`, or `--output` with `fit` and `predict`, to change output locations. View all options with, for example, `python rf_model.py predict --help`.

## Outputs

| Command | Default outputs |
| --- | --- |
| `lodo` | `rf_lodo/` or `svm_lodo/`: deposit predictions, metrics, and input-status records |
| `fit` | `rf_fitted.joblib` or `svm_fitted.joblib`: fitted model and preprocessing parameters |
| `predict` | `rf_predictions.csv` or `svm_predictions.csv`: one result per unit |

**RF:** `Final_Metal_Type` is the final assignment. An initial `Sb` prediction changes to `Au-Sb` when the 90th percentile of spot-level Au-Sb probabilities is at least **0.70**. Both direct and corrected assignments are reported.

**SVM:** `Predicted_Deposit_Type` is `G1` (orogenic hydrothermal), `G2` (epithermal hydrothermal), `G3` (intrusion-related hydrothermal), or `G4` (basinal/meteoric-fluid-dominated hydrothermal). Decision scores and `Top1_Margin` are **not probabilities**.

## Interpretation

LODO excludes all spots from the held-out deposit during fitting. Model and feature selection are not repeated within each fold; the RF correction is a post-development rule. Interpret predictions alongside geological evidence and the reported positive-observation counts.
