import sys
import os
import pandas as pd
import pickle
from tqdm import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(__file__))
from data_preprocessing import load_data, preprocess, get_Xy


# ── GPU config ────────────────────────────────────────────────────────────────
XGB_GPU = dict(device="cuda", tree_method="hist", verbosity=0, eval_metric="mlogloss")


# ── LEWrapper ─────────────────────────────────────────────────────────────────
class _LEWrapper(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper that encodes string labels before passing to XGBoost."""

    def __init__(self, estimator, le):
        self.estimator = estimator
        self.le = le

    def fit(self, X, y):
        self.le = LabelEncoder()
        self.le.fit(y)
        self.estimator.fit(X, self.le.transform(y))
        self.classes_ = self.le.classes_
        return self

    def predict(self, X):
        return self.le.inverse_transform(self.estimator.predict(X))

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def get_params(self, deep=True):
        return {"estimator": self.estimator, "le": self.le}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ── Base learners ─────────────────────────────────────────────────────────────
# SVM removed — 0.63 accuracy and extremely slow, not worth the cost in stacking
def build_base_learners():
    """Returns list of (name, estimator) tuples for the stacking base layer."""
    return [
        (
            "random_forest",
            RandomForestClassifier(max_depth=12, n_estimators=169, random_state=42, n_jobs=-1),
        ),
        (
            "xgboost",
            _LEWrapper(
                XGBClassifier(
                    max_depth=6, n_estimators=12, learning_rate=0.2,
                    random_state=42, **XGB_GPU
                ),
                LabelEncoder(),
            ),
        ),
    ]


# ── Meta-learner options ──────────────────────────────────────────────────────
META_LEARNERS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest":       RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42, n_jobs=-1),
}


def _cross_val_score_with_progress(stack, X_train, y_train, skf, meta_name):
    """
    Manual CV loop so we can show a fold-level progress bar.
    Replaces cross_val_score for visibility.
    """
    X_arr = X_train.values if hasattr(X_train, "values") else X_train
    y_arr = y_train.values if hasattr(y_train, "values") else y_train

    scores = []
    folds = list(skf.split(X_arr, y_arr))

    with tqdm(folds, desc=f"  CV folds [{meta_name}]", unit="fold", leave=False) as pbar:
        for train_idx, val_idx in pbar:
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            stack.fit(X_tr, y_tr)
            preds = stack.predict(X_val)
            fold_acc = accuracy_score(y_val, preds)
            scores.append(fold_acc)
            pbar.set_postfix({"fold_acc": f"{fold_acc:.4f}"})

    return scores


def find_best_meta_learner(X_train, y_train, cv=3):
    """
    Cross-validate each meta-learner with a fold-level progress bar.
    cv=3 keeps runtime manageable (stacking CV is expensive).
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    meta_results = []

    print(f"  Running {cv}-fold CV for each meta-learner...\n")

    for name, meta in tqdm(META_LEARNERS.items(), desc="meta-learner search", unit="meta"):
        stack = StackingClassifier(
            estimators=build_base_learners(),
            final_estimator=meta,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            passthrough=False,
            n_jobs=1,
        )
        scores = _cross_val_score_with_progress(stack, X_train, y_train, skf, name)
        mean_acc = sum(scores) / len(scores)
        std_acc  = pd.Series(scores).std()
        meta_results.append({
            "meta_learner":     name,
            "mean_cv_accuracy": mean_acc,
            "std_cv_accuracy":  std_acc,
        })
        print(f"  {name:25s}  CV={mean_acc:.4f} ± {std_acc:.4f}  (folds: {[round(s,4) for s in scores]})")

    return pd.DataFrame(meta_results).sort_values("mean_cv_accuracy", ascending=False)


def get_base_layer_accuracies(X_train, X_test, y_train, y_test):
    """Fit each base learner independently and record its standalone test accuracy."""
    base_learners = build_base_learners()
    rows = []
    for name, est in tqdm(base_learners, desc="base learner eval", unit="model"):
        print(f"  Training {name}...")
        est.fit(X_train, y_train)
        acc = accuracy_score(y_test, est.predict(X_test))
        rows.append({"model": name, "accuracy": acc})
        print(f"  {name:20s}  Accuracy={acc:.4f}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    data, clinic = load_data()
    df = preprocess(data, clinic)
    X, y = get_Xy(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ── Step 1: individual base learner accuracy ──────────────────────────────
    print("\n[1/3] Evaluating base learners individually...")
    base_acc_df = get_base_layer_accuracies(X_train, X_test, y_train, y_test)
    print("\nBase learner results:")
    print(base_acc_df.to_string(index=False))

    # ── Step 2: find best meta-learner via CV ─────────────────────────────────
    print("\n[2/3] Searching best meta-learner (3-fold CV)...")
    meta_results_df = find_best_meta_learner(X_train, y_train, cv=3)
    best_meta_name = meta_results_df.iloc[0]["meta_learner"]
    print(f"\n  → Best meta-learner: {best_meta_name}")

    # ── Step 3: train final stacked model ─────────────────────────────────────
    print(f"\n[3/3] Training final stacking model (meta={best_meta_name})...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    stack_model = StackingClassifier(
        estimators=build_base_learners(),
        final_estimator=META_LEARNERS[best_meta_name],
        cv=skf,
        passthrough=False,
        n_jobs=1,
    )

    # progress bar for final fit — StackingClassifier runs folds internally
    print(f"  Fitting final model over 5 folds (this may take a few minutes)...")
    with tqdm(total=1, desc="  Final model", unit="fit") as pbar:
        stack_model.fit(X_train, y_train)
        pbar.update(1)

    y_pred   = stack_model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred, labels=stack_model.classes_)

    # ── Accuracy comparison table ─────────────────────────────────────────────
    stacked_row = pd.DataFrame([{
        "model":    f"stacking ({best_meta_name})",
        "accuracy": accuracy_score(y_test, y_pred),
    }])
    comparison_df = (
        pd.concat([base_acc_df, stacked_row], ignore_index=True)
        .sort_values("accuracy", ascending=False)
        .reset_index(drop=True)
    )

    results = {
        "model":           stack_model,
        "X_test":          X_test,
        "y_test":          y_test,
        "y_pred":          y_pred,
        "classes":         stack_model.classes_,
        "accuracy":        accuracy_score(y_test, y_pred),
        "conf_mat":        conf_mat,
        "meta_results_df": meta_results_df,
        "base_acc_df":     base_acc_df,
        "comparison_df":   comparison_df,
        "best_meta_name":  best_meta_name,
    }

    os.makedirs("models", exist_ok=True)
    with open("models/stacking_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"\nStacking Accuracy : {results['accuracy']:.4f}")
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    print("\nSaved → models/stacking_results.pkl")