"""Improved heart failure prediction analysis.

This script implements the main recommendations from PROJECT_IMPROVEMENTS.md:

- leakage-safe sklearn/imblearn pipelines
- repeated stratified cross-validation
- bootstrap confidence intervals
- probability-based ROC-AUC and PR-AUC
- calibration curves, Brier score, and expected calibration error
- optional probability recalibration
- cross-validated permutation importance and feature stability
- optional SHAP plots when shap is installed

Example:
    python improved_heart_failure_analysis.py --data ../input/heart-failure/heart_failure.csv
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontconfig-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from imblearn.combine import SMOTEENN
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:  # pragma: no cover - handled at runtime
    SMOTEENN = None
    ImbPipeline = None


warnings.filterwarnings("ignore", category=UserWarning)


TARGET = "DEATH_EVENT"
DEFAULT_DATA = Path("../input/heart-failure/heart_failure.csv")
DEFAULT_OUTPUT = Path("outputs/improved_analysis")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: BaseEstimator
    needs_scaling: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run improved heart failure model evaluation."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to heart_failure.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for generated CSV, JSON, and plot files.",
    )
    parser.add_argument(
        "--target",
        default=TARGET,
        help="Name of the binary target column.",
    )
    parser.add_argument(
        "--use-smoteenn",
        action="store_true",
        help="Use SMOTEENN inside the cross-validation pipeline.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of stratified folds.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Number of repeats for repeated stratified cross-validation.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Number of bootstrap samples for confidence intervals.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used throughout the analysis.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["sigmoid", "isotonic"],
        default="sigmoid",
        help="Calibration method. Sigmoid is safer for small datasets.",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip optional SHAP analysis even if shap is installed.",
    )
    return parser.parse_args()


def load_dataset(path: Path, target: str) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Pass --data with the path to heart_failure.csv."
        )

    data = pd.read_csv(path)
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in {path}.")

    x = data.drop(columns=[target])
    y = data[target].astype(int)
    return x, y


def build_preprocessor(feature_names: list[str], needs_scaling: bool) -> BaseEstimator:
    if not needs_scaling:
        return "passthrough"

    return ColumnTransformer(
        transformers=[("num", StandardScaler(), feature_names)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def model_specs(random_state: int) -> list[ModelSpec]:
    return [
        ModelSpec(
            "logistic_regression",
            LogisticRegression(max_iter=2000, random_state=random_state),
            True,
        ),
        ModelSpec(
            "decision_tree",
            DecisionTreeClassifier(random_state=random_state),
            False,
        ),
        ModelSpec(
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",
            ),
            False,
        ),
        ModelSpec("gaussian_nb", GaussianNB(), True),
        ModelSpec("knn", KNeighborsClassifier(), True),
        ModelSpec(
            "svc",
            SVC(
                kernel="rbf",
                probability=True,
                random_state=random_state,
                class_weight="balanced",
            ),
            True,
        ),
    ]


def build_pipeline(
    spec: ModelSpec,
    feature_names: list[str],
    use_smoteenn: bool,
    random_state: int,
) -> BaseEstimator:
    needs_scaling = spec.needs_scaling or use_smoteenn
    steps = [("preprocess", build_preprocessor(feature_names, needs_scaling))]

    if use_smoteenn:
        if SMOTEENN is None or ImbPipeline is None:
            raise ImportError(
                "imbalanced-learn is required for --use-smoteenn. "
                "Install it with: pip install imbalanced-learn"
            )
        steps.append(("smoteenn", SMOTEENN(random_state=random_state)))
        steps.append(("model", spec.estimator))
        return ImbPipeline(steps)

    steps.append(("model", spec.estimator))
    return Pipeline(steps)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)
    ece = 0.0

    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        observed = y_true[mask].mean()
        predicted = y_prob[mask].mean()
        ece += (mask.mean()) * abs(observed - predicted)

    return float(ece)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred_or_prob: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int,
    random_state: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    values = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sampled_true = y_true[idx]
        sampled_pred = y_pred_or_prob[idx]
        if len(np.unique(sampled_true)) < 2:
            continue
        try:
            values.append(metric(sampled_true, sampled_pred))
        except ValueError:
            continue

    if not values:
        return float("nan"), float("nan")

    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def point_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
    }


def metric_ci_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int,
    random_state: int,
) -> dict[str, tuple[float, float]]:
    y_pred = (y_prob >= 0.5).astype(int)
    metrics: dict[str, tuple[np.ndarray, Callable[[np.ndarray, np.ndarray], float]]] = {
        "accuracy": (y_pred, accuracy_score),
        "balanced_accuracy": (y_pred, balanced_accuracy_score),
        "precision": (y_pred, lambda yt, yp: precision_score(yt, yp, zero_division=0)),
        "recall": (y_pred, lambda yt, yp: recall_score(yt, yp, zero_division=0)),
        "f1": (y_pred, lambda yt, yp: f1_score(yt, yp, zero_division=0)),
        "roc_auc": (y_prob, roc_auc_score),
        "pr_auc": (y_prob, average_precision_score),
        "brier_score": (y_prob, brier_score_loss),
        "ece": (y_prob, expected_calibration_error),
    }

    return {
        name: bootstrap_ci(y_true, values, fn, n_bootstrap, random_state)
        for name, (values, fn) in metrics.items()
    }


def evaluate_models(
    x: pd.DataFrame,
    y: pd.Series,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, BaseEstimator], dict[str, np.ndarray]]:
    feature_names = list(x.columns)
    repeated_cv = RepeatedStratifiedKFold(
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )
    oof_cv = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.random_state,
    )
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "brier_score": "neg_brier_score",
    }

    rows = []
    pipelines = {}
    oof_probabilities = {}

    for spec in model_specs(args.random_state):
        print(f"Evaluating {spec.name}...")
        pipeline = build_pipeline(spec, feature_names, args.use_smoteenn, args.random_state)
        pipelines[spec.name] = pipeline

        cv_scores = cross_validate(
            pipeline,
            x,
            y,
            cv=repeated_cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        y_prob = cross_val_predict(
            pipeline,
            x,
            y,
            cv=oof_cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        oof_probabilities[spec.name] = y_prob

        point = point_metrics(y.to_numpy(), y_prob)
        ci = metric_ci_table(
            y.to_numpy(),
            y_prob,
            args.bootstrap_samples,
            args.random_state,
        )

        row = {
            "model": spec.name,
            "use_smoteenn": args.use_smoteenn,
            "cv_splits": args.n_splits,
            "cv_repeats": args.n_repeats,
        }

        for metric_name in [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
        ]:
            values = cv_scores[f"test_{metric_name}"]
            row[f"{metric_name}_cv_mean"] = values.mean()
            row[f"{metric_name}_cv_std"] = values.std()
            row[f"{metric_name}_oof"] = point[metric_name]
            row[f"{metric_name}_ci_low"] = ci[metric_name][0]
            row[f"{metric_name}_ci_high"] = ci[metric_name][1]

        brier_values = -cv_scores["test_brier_score"]
        row["brier_score_cv_mean"] = brier_values.mean()
        row["brier_score_cv_std"] = brier_values.std()
        row["brier_score_oof"] = point["brier_score"]
        row["brier_score_ci_low"] = ci["brier_score"][0]
        row["brier_score_ci_high"] = ci["brier_score"][1]
        row["ece_oof"] = point["ece"]
        row["ece_ci_low"] = ci["ece"][0]
        row["ece_ci_high"] = ci["ece"][1]
        rows.append(row)

    return pd.DataFrame(rows), pipelines, oof_probabilities


def save_roc_pr_plots(
    y: pd.Series,
    probabilities: dict[str, np.ndarray],
    output: Path,
) -> None:
    y_true = y.to_numpy()

    plt.figure(figsize=(8, 6))
    for name, y_prob in probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_value = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{name} AUC={auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Probability-based ROC curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output / "roc_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    for name, y_prob in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap_value = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, label=f"{name} AP={ap_value:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output / "precision_recall_curves.png", dpi=200)
    plt.close()


def save_calibration_plots(
    y: pd.Series,
    probabilities: dict[str, np.ndarray],
    output: Path,
) -> None:
    y_true = y.to_numpy()

    plt.figure(figsize=(8, 6))
    for name, y_prob in probabilities.items():
        observed, predicted = calibration_curve(
            y_true,
            y_prob,
            n_bins=10,
            strategy="quantile",
        )
        plt.plot(predicted, observed, marker="o", label=name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("Calibration curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output / "calibration_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    for name, y_prob in probabilities.items():
        plt.hist(y_prob, bins=10, alpha=0.35, label=name)
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Predicted probability distributions")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output / "probability_histograms.png", dpi=200)
    plt.close()


def make_calibrated_classifier(
    estimator: BaseEstimator,
    method: str,
    cv: int,
) -> CalibratedClassifierCV:
    try:
        return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method=method, cv=cv)


def evaluate_calibrated_best_model(
    best_model_name: str,
    pipeline: BaseEstimator,
    x: pd.DataFrame,
    y: pd.Series,
    args: argparse.Namespace,
    output: Path,
) -> pd.DataFrame:
    cv = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.random_state,
    )
    calibrated = make_calibrated_classifier(
        clone(pipeline),
        method=args.calibration_method,
        cv=args.n_splits,
    )

    base_prob = cross_val_predict(
        pipeline,
        x,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    calibrated_prob = cross_val_predict(
        calibrated,
        x,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    rows = []
    for label, prob in [
        (f"{best_model_name}_uncalibrated", base_prob),
        (f"{best_model_name}_{args.calibration_method}_calibrated", calibrated_prob),
    ]:
        metrics = point_metrics(y.to_numpy(), prob)
        rows.append({"model": label, **metrics})

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "calibration_comparison.csv", index=False)

    save_calibration_plots(
        y,
        {
            "uncalibrated": base_prob,
            f"{args.calibration_method}_calibrated": calibrated_prob,
        },
        output,
    )
    return comparison


def cross_validated_permutation_importance(
    pipeline: BaseEstimator,
    x: pd.DataFrame,
    y: pd.Series,
    args: argparse.Namespace,
) -> pd.DataFrame:
    cv = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.random_state,
    )
    rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        estimator = clone(pipeline)
        estimator.fit(x.iloc[train_idx], y.iloc[train_idx])
        result = permutation_importance(
            estimator,
            x.iloc[test_idx],
            y.iloc[test_idx],
            scoring="roc_auc",
            n_repeats=30,
            random_state=args.random_state + fold,
            n_jobs=-1,
        )

        ranks = pd.Series(
            result.importances_mean,
            index=x.columns,
        ).rank(ascending=False, method="min")

        for feature, mean, std, rank in zip(
            x.columns,
            result.importances_mean,
            result.importances_std,
            ranks,
        ):
            rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "importance_mean": mean,
                    "importance_std": std,
                    "rank": int(rank),
                    "top_5": rank <= 5,
                }
            )

    return pd.DataFrame(rows)


def save_importance_outputs(
    importance: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    summary = (
        importance.groupby("feature", as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_std=("importance_mean", "std"),
            top_5_folds=("top_5", "sum"),
            mean_rank=("rank", "mean"),
        )
        .sort_values(["importance_mean", "top_5_folds"], ascending=False)
    )

    importance.to_csv(output / "permutation_importance_by_fold.csv", index=False)
    summary.to_csv(output / "permutation_importance_summary.csv", index=False)

    top = summary.head(15).sort_values("importance_mean")
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    plt.xlabel("Mean ROC-AUC decrease after permutation")
    plt.title("Cross-validated permutation importance")
    plt.tight_layout()
    plt.savefig(output / "permutation_importance.png", dpi=200)
    plt.close()
    return summary


def save_partial_dependence_plots(
    pipeline: BaseEstimator,
    x: pd.DataFrame,
    y: pd.Series,
    importance_summary: pd.DataFrame,
    output: Path,
) -> None:
    top_features = importance_summary.head(4)["feature"].tolist()
    if not top_features:
        return

    estimator = clone(pipeline).fit(x, y)
    fig, ax = plt.subplots(figsize=(10, 8))
    PartialDependenceDisplay.from_estimator(
        estimator,
        x,
        features=top_features,
        kind="average",
        ax=ax,
    )
    fig.suptitle("Partial dependence for top features")
    fig.tight_layout()
    fig.savefig(output / "partial_dependence_top_features.png", dpi=200)
    plt.close(fig)


def save_optional_shap_outputs(
    pipeline: BaseEstimator,
    x: pd.DataFrame,
    y: pd.Series,
    output: Path,
    skip_shap: bool,
) -> None:
    if skip_shap:
        return

    try:
        import shap
    except ImportError:
        print("SHAP is not installed; skipping SHAP plots.")
        return

    estimator = clone(pipeline).fit(x, y)
    sample = x.sample(min(len(x), 100), random_state=42)

    try:
        explainer = shap.Explainer(estimator.predict_proba, sample)
        shap_values = explainer(sample)
        shap_class_1 = shap_values[:, :, 1]

        plt.figure()
        shap.plots.beeswarm(shap_class_1, show=False)
        plt.tight_layout()
        plt.savefig(output / "shap_summary.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.plots.bar(shap_class_1, show=False)
        plt.tight_layout()
        plt.savefig(output / "shap_bar.png", dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as exc:  # pragma: no cover - depends on optional shap behavior
        print(f"SHAP analysis failed and was skipped: {exc}")


def save_run_config(args: argparse.Namespace, output: Path) -> None:
    config = {
        "data": str(args.data),
        "target": args.target,
        "use_smoteenn": args.use_smoteenn,
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "bootstrap_samples": args.bootstrap_samples,
        "random_state": args.random_state,
        "calibration_method": args.calibration_method,
    }
    with (output / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    save_run_config(args, args.output)

    x, y = load_dataset(args.data, args.target)
    summary, pipelines, probabilities = evaluate_models(x, y, args)
    summary = summary.sort_values("roc_auc_cv_mean", ascending=False)
    summary.to_csv(args.output / "model_performance_summary.csv", index=False)

    save_roc_pr_plots(y, probabilities, args.output)
    save_calibration_plots(y, probabilities, args.output)

    best_model_name = summary.iloc[0]["model"]
    best_pipeline = pipelines[best_model_name]
    evaluate_calibrated_best_model(best_model_name, best_pipeline, x, y, args, args.output)

    importance = cross_validated_permutation_importance(best_pipeline, x, y, args)
    importance_summary = save_importance_outputs(importance, args.output)
    save_partial_dependence_plots(best_pipeline, x, y, importance_summary, args.output)
    save_optional_shap_outputs(best_pipeline, x, y, args.output, args.skip_shap)

    print("\nAnalysis complete.")
    print(f"Best model by repeated CV ROC-AUC: {best_model_name}")
    print(f"Outputs saved to: {args.output}")


if __name__ == "__main__":
    main()
