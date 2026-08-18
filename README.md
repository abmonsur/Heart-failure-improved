# Heart Failure Prediction

Machine learning project for predicting heart failure mortality from clinical records. The main target variable is `DEATH_EVENT`, and the project compares multiple supervised classification models using cross-validation, hyperparameter tuning, probability-based evaluation, calibration analysis, and model interpretability outputs.

This repository is intended for educational and research work. It is not a clinical decision support tool.

## Project Overview

The project evaluates common binary classification models for heart failure outcome prediction:

- Logistic Regression
- Decision Tree
- Random Forest
- Gaussian Naive Bayes
- K-Nearest Neighbors
- Support Vector Classifier

The repository includes original notebook experiments, hyperparameter tuning notebooks, and an improved Python analysis script that provides a cleaner, leakage-safe evaluation workflow.

## Repository Structure

```text
.
├── improved_heart_failure_analysis.py   # Main improved analysis pipeline
├── requirements-improved.txt            # Python dependencies for the improved workflow
├── posttuning.ipynb                     # Baseline heart failure notebook
├── hccnormal.ipynb                      # Related HCC survival notebook
├── PROJECT_EXPLANATION.md               # Detailed explanation of the original notebooks
├── PROJECT_IMPROVEMENTS.md              # Recommendations and implemented improvements
└── tuning/                              # Grid search and randomized search experiments
```

The `tuning/` directory contains experiments across:

- `GridSearchCV` and `RandomizedSearchCV`
- Standard scaling and min-max scaling
- 5-fold and 10-fold cross-validation
- Runs with and without `SMOTEENN`

## Dataset

The heart failure dataset is not included in this repository. The code expects a CSV file with a `DEATH_EVENT` target column.

The original notebooks use this Kaggle-style path:

```text
../input/heart-failure/heart_failure.csv
```

When running locally, place the dataset anywhere you prefer and pass the path with `--data`:

```bash
python improved_heart_failure_analysis.py --data path/to/heart_failure.csv
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Heart-Failure-Prediction.git
cd Heart-Failure-Prediction
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements-improved.txt
```

If you want to run the notebooks, also install Jupyter:

```bash
pip install notebook
```

## Running the Improved Analysis

Run the main workflow:

```bash
python improved_heart_failure_analysis.py --data path/to/heart_failure.csv
```

Run the same workflow with `SMOTEENN` class balancing inside the cross-validation pipeline:

```bash
python improved_heart_failure_analysis.py --data path/to/heart_failure.csv --use-smoteenn
```

Skip SHAP plots if you want a faster run or do not have SHAP configured:

```bash
python improved_heart_failure_analysis.py --data path/to/heart_failure.csv --skip-shap
```

By default, results are written to:

```text
outputs/improved_analysis/
```

You can choose another output directory:

```bash
python improved_heart_failure_analysis.py --data path/to/heart_failure.csv --output outputs/run_01
```

## Generated Outputs

The improved script generates model comparison tables and plots, including:

- `model_performance_summary.csv`
- `calibration_comparison.csv`
- `permutation_importance_by_fold.csv`
- `permutation_importance_summary.csv`
- `run_config.json`
- `roc_curves.png`
- `precision_recall_curves.png`
- `calibration_curves.png`
- `probability_histograms.png`
- `permutation_importance.png`
- `partial_dependence_top_features.png`
- `shap_summary.png`, if SHAP succeeds
- `shap_bar.png`, if SHAP succeeds

## Evaluation Approach

The improved workflow uses:

- Leakage-safe `sklearn` and `imblearn` pipelines
- Repeated stratified cross-validation
- Bootstrap confidence intervals
- Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, and expected calibration error
- Probability calibration comparison
- Cross-validated permutation importance
- Partial dependence plots for important features
- Optional SHAP-based interpretability plots

This provides a more reliable evaluation than a single train-test split, especially because heart failure datasets are often small and class-imbalanced.

## Notebooks

`posttuning.ipynb` contains the original heart failure modeling workflow. It trains and evaluates the same family of classifiers using cross-validation, confusion matrices, ROC curves, and AUC scores.

`hccnormal.ipynb` follows a similar pattern for a hepatocellular carcinoma survival dataset. It is related modeling work but uses a different dataset and target column.

The notebooks in `tuning/` are historical experiment files for comparing scalers, search strategies, cross-validation folds, and class-balancing settings. If you run them locally, update their dataset paths as needed.

## Important Notes

- This project is for learning, experimentation, and research discussion only.
- Do not use the model outputs for medical diagnosis or treatment decisions.
- The dataset is small, so reported metrics should be interpreted with uncertainty.
- Add a license file before publishing if you want others to reuse the code under clear terms.

