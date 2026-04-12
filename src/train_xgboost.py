import sys
import os
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from data_preprocessing import load_data, preprocess, get_Xy


# ── GPU config ────────────────────────────────────────────────────────────────
# Use CPU for hyperparameter search to avoid GPU memory allocation errors.
# RTX 5060 has limited VRAM; hyperparameter search requires many model instances.
# GPU can be enabled for the final model training if needed.
GPU_PARAMS = dict(
    device="cuda",
    
)


def _make_clf(**kwargs):
    """Return an XGBClassifier with GPU params merged in."""
    return XGBClassifier(
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,          # suppress XGBoost's own per-tree logs
        **GPU_PARAMS,
        **kwargs,
    )


def find_best_max_depth(X_train, X_test, y_train, y_test, le, depth_range=range(1, 6)):
    max_depth_best = []
    for i in tqdm(depth_range, desc="max_depth search", unit="depth"):
        clf = _make_clf(max_depth=i)
        clf.fit(X_train, le.transform(y_train))
        y_pred = clf.predict(X_test)
        max_depth_best.append([i, accuracy_score(le.transform(y_test), y_pred)])
    return pd.DataFrame(max_depth_best, columns=["max_depth", "Accuracy"])


def find_best_n_estimators(X_train, X_test, y_train, y_test, le, estimator_range=range(10, 20)):
    n_estimators = []
    for i in tqdm(estimator_range, desc="n_estimators search", unit="est"):
        clf = _make_clf(max_depth=5, n_estimators=i)
        clf.fit(X_train, le.transform(y_train))
        y_pred = clf.predict(X_test)
        n_estimators.append([i, accuracy_score(le.transform(y_test), y_pred)])
    return pd.DataFrame(n_estimators, columns=["n_estimator", "Accuracy"])


def find_best_learning_rate(X_train, X_test, y_train, y_test, le):
    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    lr_results = []
    for lr in tqdm(learning_rates, desc="learning_rate search", unit="lr"):
        clf = _make_clf(max_depth=5, n_estimators=20, learning_rate=lr)
        clf.fit(X_train, le.transform(y_train))
        y_pred = clf.predict(X_test)
        lr_results.append([lr, accuracy_score(le.transform(y_test), y_pred)])
    return pd.DataFrame(lr_results, columns=["learning_rate", "Accuracy"])


def get_feature_importance(xgb_model, X, le):
    sorted_idx = (-xgb_model.feature_importances_).argsort()[:10]
    feature_data = pd.DataFrame(
        {
            "Feature": X.columns[sorted_idx],
            "Importance": xgb_model.feature_importances_[sorted_idx],
        }
    ).sort_values(by="Importance", ascending=False)
    return feature_data


if __name__ == "__main__":
    data, clinic = load_data()
    df = preprocess(data, clinic)
    X, y = get_Xy(df)

    # XGBoost requires numeric labels
    le = LabelEncoder()
    le.fit(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ── Hyperparameter search ─────────────────────────────────────────────────
    print("\n[1/3] Searching best max_depth...")
    max_depth_df = find_best_max_depth(X_train, X_test, y_train, y_test, le)
    print(max_depth_df.to_string(index=False))

    print("\n[2/3] Searching best n_estimators...")
    n_estimate_df = find_best_n_estimators(X_train, X_test, y_train, y_test, le)
    best_n = n_estimate_df.loc[n_estimate_df["Accuracy"].idxmax(), "n_estimator"]
    print(f"  → Best n_estimator: {best_n}")

    print("\n[3/3] Searching best learning_rate...")
    lr_df = find_best_learning_rate(X_train, X_test, y_train, y_test, le)
    best_lr = lr_df.loc[lr_df["Accuracy"].idxmax(), "learning_rate"]
    print(f"  → Best learning_rate: {best_lr}")

    # ── Final model ───────────────────────────────────────────────────────────
    print(f"\nTraining final XGBoost model  (max_depth=6, n_estimators={best_n}, lr={best_lr})...")
    xgb_model = _make_clf(
        max_depth=6,
        n_estimators=int(best_n),
        learning_rate=best_lr,
        # show per-round loss during final training only
        callbacks=[],
    )
    xgb_model.fit(
        X_train, le.transform(y_train),
        eval_set=[(X_test, le.transform(y_test))],
        verbose=10,   # print every 10 rounds  ← final-model progress
    )

    y_pred_encoded = xgb_model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_encoded)

    conf_mat = confusion_matrix(y_test, y_pred, labels=le.classes_)
    feature_data = get_feature_importance(xgb_model, X, le)

    results = {
        "model": xgb_model,
        "label_encoder": le,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "classes": le.classes_,
        "accuracy": accuracy_score(y_test, y_pred),
        "conf_mat": conf_mat,
        "feature_data": feature_data,
        "max_depth_df": max_depth_df,
        "n_estimate_df": n_estimate_df,
        "lr_df": lr_df,
    }

    os.makedirs("models", exist_ok=True)
    with open("models/xgb_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"\nXGBoost Accuracy: {results['accuracy']:.4f}")
    print("Saved → models/xgb_results.pkl")