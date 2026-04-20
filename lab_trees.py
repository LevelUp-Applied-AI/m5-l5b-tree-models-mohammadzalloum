import os

# Use a non-interactive matplotlib backend so plots save cleanly in CI
# and on headless environments.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
    "senior_citizen",
    "has_partner",
    "has_dependents",
    "contract_months",
]


def load_and_split(filepath="data/telecom_churn.csv", random_state=42):
    """Load the Petra Telecom dataset and split 80/20 with stratification.

    Args:
        filepath: Path to telecom_churn.csv.
        random_state: Random seed for reproducible split.

    Returns:
        Tuple (X_train, X_test, y_train, y_test) where X contains only
        NUMERIC_FEATURES and y is the `churned` column.
    """
    df = pd.read_csv(filepath)

    X = df[NUMERIC_FEATURES]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def build_decision_tree(X_train, y_train, max_depth=5, random_state=42):
    """Train a DecisionTreeClassifier.

    Args:
        max_depth: Maximum tree depth (None means unconstrained).
        random_state: Random seed.

    Returns:
        Fitted DecisionTreeClassifier.
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error using equal-count (quantile) binning.

    Sort samples by predicted probability, split into `n_bins` equal-size
    chunks, and sum the bin-weighted absolute difference between each bin's
    mean predicted probability and its fraction of true positives.

    A perfectly calibrated model has ECE = 0. Higher ECE means predicted
    probabilities don't correspond to empirical rates.

    Args:
        y_true: 1D array-like of true binary labels (0 or 1).
        y_prob: 1D array-like of predicted probabilities for class 1.
        n_bins: Number of equal-count bins.

    Returns:
        ECE as a float in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    order = np.argsort(y_prob)
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]

    n = len(y_true_sorted)
    bins = np.array_split(np.arange(n), n_bins)

    ece = 0.0

    for bin_idx in bins:
        if len(bin_idx) == 0:
            continue

        bin_true = y_true_sorted[bin_idx]
        bin_prob = y_prob_sorted[bin_idx]

        mean_prob = np.mean(bin_prob)
        frac_positive = np.mean(bin_true)

        ece += (len(bin_idx) / n) * abs(mean_prob - frac_positive)

    return float(ece)


def compare_dt_calibration(X_train, X_test, y_train, y_test):
    """Compare calibration of an unbounded DT vs a depth-5 DT.

    Teaches that pure-leaf trees (unbounded depth) produce extreme
    probabilities → poor calibration; depth-constrained trees smooth
    probabilities → better calibration.

    Returns:
        Dict with keys 'ece_unbounded' and 'ece_depth_5' (floats in [0, 1]).
    """
    dt_unbounded = build_decision_tree(
        X_train, y_train, max_depth=None, random_state=42
    )
    prob_unbounded = dt_unbounded.predict_proba(X_test)[:, 1]
    ece_unbounded = compute_ece(y_test, prob_unbounded)

    dt_depth_5 = build_decision_tree(
        X_train, y_train, max_depth=5, random_state=42
    )
    prob_depth_5 = dt_depth_5.predict_proba(X_test)[:, 1]
    ece_depth_5 = compute_ece(y_test, prob_depth_5)

    return {
        "ece_unbounded": float(ece_unbounded),
        "ece_depth_5": float(ece_depth_5),
    }


def build_random_forest(
    X_train,
    y_train,
    n_estimators=100,
    max_depth=10,
    class_weight=None,
    random_state=42,
):
    """Train a RandomForestClassifier.

    Args:
        class_weight: None for default, 'balanced' to reweight the loss
            so minority-class samples count more during training.
        random_state: Random seed.

    Returns:
        Fitted RandomForestClassifier.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def get_feature_importances(model, feature_names):
    """Return a dict of feature_name -> importance, sorted descending."""
    importances = zip(feature_names, model.feature_importances_)
    sorted_importances = sorted(importances, key=lambda x: x[1], reverse=True)
    return dict(sorted_importances)


def evaluate_recall_at_threshold(model, X_test, y_test, threshold=0.5):
    """Recall for class 1 at a specified decision threshold.

    Standard .predict() uses threshold 0.5. Passing a different threshold
    lets you observe how recall responds to operating-point choice — which
    is what `class_weight='balanced'` effectively shifts.

    Returns:
        Recall as a float in [0, 1].
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return recall_score(y_test, y_pred, zero_division=0)


def compute_pr_auc(model, X_test, y_test):
    """PR-AUC (average precision) for the positive class.

    Threshold-independent: measures the model's ability to rank positives
    above negatives across all thresholds. Unlike recall at a specific
    threshold, PR-AUC does not change when you apply class_weight='balanced'
    in a way that merely shifts predicted probabilities uniformly — the
    ranking is what matters.

    Returns:
        Float in [0, 1].
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    return average_precision_score(y_test, y_prob)


def plot_pr_curves(rf_default, rf_balanced, X_test, y_test, output_path):
    """Plot PR curves for both RF models on the same axes and save as PNG.

    Args:
        output_path: Destination path (e.g., 'results/pr_curves.png').
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    PrecisionRecallDisplay.from_estimator(
        rf_default,
        X_test,
        y_test,
        ax=ax,
        name="RF default",
    )

    PrecisionRecallDisplay.from_estimator(
        rf_balanced,
        X_test,
        y_test,
        ax=ax,
        name="RF balanced",
    )

    ax.set_title("Precision-Recall Curves: Random Forest Models")
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_curves(rf_default, rf_balanced, X_test, y_test, output_path):
    """Plot calibration curves for both RF models and save as PNG."""
    fig, ax = plt.subplots(figsize=(8, 6))

    CalibrationDisplay.from_estimator(
        rf_default,
        X_test,
        y_test,
        n_bins=10,
        ax=ax,
        name="RF default",
    )

    CalibrationDisplay.from_estimator(
        rf_balanced,
        X_test,
        y_test,
        n_bins=10,
        ax=ax,
        name="RF balanced",
    )

    ax.set_title("Calibration Curves: Random Forest Models")
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def build_logistic_regression(X_train_scaled, y_train, random_state=42):
    """Train a LogisticRegression baseline on scaled features.

    Linear models need their inputs on a common scale, otherwise features
    with larger numeric ranges (total_charges ~ 0-9000) swamp features with
    smaller ranges (binary indicators at 0/1). Apply StandardScaler to the
    training features BEFORE calling this function.

    Returns:
        Fitted LogisticRegression(max_iter=1000).
    """
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    return model


def find_tree_vs_linear_disagreement(
    rf_model,
    lr_model,
    X_test_raw,
    X_test_scaled,
    y_test,
    feature_names,
    min_diff=0.15,
):
    """Find ONE test sample where RF and LR predicted probabilities differ most.

    The tree-vs-linear capability demonstration. The random forest can
    capture feature interactions, non-monotonic relationships, and threshold
    effects that a linear model cannot express with per-feature coefficients.
    Finding a sample where the two models disagree — and explaining WHY in
    structural terms — is the lab's evidence that trees have capabilities
    linear models don't, regardless of aggregate PR-AUC.
    """
    rf_proba = rf_model.predict_proba(X_test_raw)[:, 1]
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

    prob_diff = np.abs(rf_proba - lr_proba)
    max_pos = int(np.argmax(prob_diff))
    max_diff = float(prob_diff[max_pos])

    if max_diff < min_diff:
        return None

    if hasattr(X_test_raw, "iloc"):
        row = X_test_raw.iloc[max_pos]
        feature_values = {name: row[name] for name in feature_names}
    else:
        row = X_test_raw[max_pos]
        feature_values = {name: row[i] for i, name in enumerate(feature_names)}

    if hasattr(y_test, "iloc"):
        true_label = int(y_test.iloc[max_pos])
    else:
        true_label = int(y_test[max_pos])

    return {
        "sample_idx": max_pos,
        "feature_values": feature_values,
        "rf_proba": float(rf_proba[max_pos]),
        "lr_proba": float(lr_proba[max_pos]),
        "prob_diff": max_diff,
        "true_label": true_label,
    }


def main():
    """Orchestrate all 7 lab tasks. Run with: python lab_trees.py"""
    os.makedirs("results", exist_ok=True)

    # Task 1: Load + split
    result = load_and_split()
    if not result:
        print("load_and_split not implemented. Exiting.")
        return

    X_train, X_test, y_train, y_test = result
    print(f"Train: {len(X_train)}  Test: {len(X_test)}  Churn rate: {y_train.mean():.2%}")

    # Task 2: Decision tree + calibration comparison
    dt = build_decision_tree(X_train, y_train)
    if dt is not None:
        print("\n--- Decision Tree (max_depth=5) ---")
        print(classification_report(y_test, dt.predict(X_test), zero_division=0))

        dt_pr_auc = compute_pr_auc(dt, X_test, y_test)
        print(f"DT PR-AUC: {dt_pr_auc:.3f}")

        plt.figure(figsize=(14, 8))
        plot_tree(
            dt,
            feature_names=NUMERIC_FEATURES,
            max_depth=3,
            filled=True,
            fontsize=8,
        )
        plt.savefig("results/decision_tree.png", dpi=100, bbox_inches="tight")
        plt.close()

    cal = compare_dt_calibration(X_train, X_test, y_train, y_test)
    if cal:
        print(f"DT ECE (max_depth=None): {cal['ece_unbounded']:.3f}")
        print(f"DT ECE (max_depth=5):    {cal['ece_depth_5']:.3f}")

    # Task 3: Random forest + feature importances
    rf = build_random_forest(X_train, y_train)
    if rf is not None:
        print("\n--- Random Forest (max_depth=10) ---")
        imp = get_feature_importances(rf, NUMERIC_FEATURES)
        if imp:
            print("Feature importances:")
            for name, value in imp.items():
                print(f"  {name:<22s} {value:.3f}")

    # Task 4: Balanced RF + recall@0.5 comparison + PR-AUC
    rf_bal = build_random_forest(X_train, y_train, class_weight="balanced")
    if rf is not None and rf_bal is not None:
        print("\n--- RF default classification report ---")
        print(classification_report(y_test, rf.predict(X_test), zero_division=0))

        print("\n--- RF balanced classification report ---")
        print(classification_report(y_test, rf_bal.predict(X_test), zero_division=0))

        r_def = evaluate_recall_at_threshold(rf, X_test, y_test, threshold=0.5)
        r_bal = evaluate_recall_at_threshold(rf_bal, X_test, y_test, threshold=0.5)

        print("\n--- class_weight effect at default 0.5 threshold ---")
        print(f"  RF default recall@0.5:  {r_def:.3f}")
        print(f"  RF balanced recall@0.5: {r_bal:.3f}  (ratio: {r_bal / max(r_def, 1e-9):.2f}x)")

        auc_def = compute_pr_auc(rf, X_test, y_test)
        auc_bal = compute_pr_auc(rf_bal, X_test, y_test)

        print("\n--- PR-AUC (threshold-independent ranking quality) ---")
        print(f"  RF default:  {auc_def:.3f}")
        print(f"  RF balanced: {auc_bal:.3f}")
        print(
            "Note: class_weight='balanced' shifts the operating point at a fixed "
            "threshold; it does not improve the underlying ranking (PR-AUC)."
        )

        # Task 5: PR curves + calibration curves
        plot_pr_curves(rf, rf_bal, X_test, y_test, "results/pr_curves.png")
        plot_calibration_curves(
            rf, rf_bal, X_test, y_test, "results/calibration_curves.png"
        )

    # Task 6: Tree-vs-linear disagreement
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = build_logistic_regression(X_train_scaled, y_train)
    if rf is not None and lr is not None:
        d = find_tree_vs_linear_disagreement(
            rf, lr, X_test, X_test_scaled, y_test, NUMERIC_FEATURES
        )
        if d:
            print(f"\n--- Tree-vs-linear disagreement (sample idx={d['sample_idx']}) ---")
            print(f"  RF P(churn=1)={d['rf_proba']:.3f}  LR P(churn=1)={d['lr_proba']:.3f}")
            print(f"  |diff| = {d['prob_diff']:.3f}   true label = {d['true_label']}")
            print(f"  Feature values: {d['feature_values']}")

    print("\nSaved files:")
    print("  - results/decision_tree.png")
    print("  - results/pr_curves.png")
    print("  - results/calibration_curves.png")


if __name__ == "__main__":
    main()
