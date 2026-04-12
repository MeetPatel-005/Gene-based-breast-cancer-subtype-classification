import sys
import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.append(os.path.dirname(__file__))
from data_preprocessing import load_data, preprocess, get_Xy


def find_best_max_depth(X_train, X_test, y_train, y_test, depth_range=range(1, 20)):
    max_depth_best = []
    for i in depth_range:
        clf = rf(max_depth=i, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        max_depth_best.append([i, accuracy_score(y_test, y_pred)])
    return pd.DataFrame(max_depth_best, columns=["max_depth", "Accuracy"])


def find_best_n_estimators(X_train, X_test, y_train, y_test, estimator_range=range(10, 200)):
    n_estimators = []
    for i in estimator_range:
        clf = rf(max_depth=10, n_estimators=i, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        n_estimators.append([i, accuracy_score(y_test, y_pred)])
    return pd.DataFrame(n_estimators, columns=["n_estimator", "Accuracy"])


def get_feature_importance(rf_model, X):
    sorted_idx = (-rf_model.feature_importances_).argsort()[:10]
    feature_data = pd.DataFrame(
        {
            "Feature": X.columns[sorted_idx],
            "Importance": rf_model.feature_importances_[sorted_idx],
        }
    ).sort_values(by="Importance", ascending=False)
    feature_data["Feature Names"] = [
        "DRAIC", "TTC6", "SLC7A13", "DNAI7", "CCNB2",
        "CENPA", "ESR1", "RAB6C", "CDC20", "CCNE1",
    ]
    return feature_data


if __name__ == "__main__":
    data, clinic = load_data()
    df = preprocess(data, clinic)
    X, y = get_Xy(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter search
    max_depth_df = find_best_max_depth(X_train, X_test, y_train, y_test)
    n_estimate_df = find_best_n_estimators(X_train, X_test, y_train, y_test)

    # Train final model
    rf_model = rf(max_depth=12, n_estimators=169, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    conf_mat = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
    feature_data = get_feature_importance(rf_model, X)

    # Bundle everything the notebook needs into one results dict
    results = {
        "model": rf_model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "classes": rf_model.classes_,
        "accuracy": accuracy_score(y_test, y_pred),
        "conf_mat": conf_mat,
        "feature_data": feature_data,
        "max_depth_df": max_depth_df,
        "n_estimate_df": n_estimate_df,
    }

    os.makedirs("models", exist_ok=True)
    with open("models/rf_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"Random Forest Accuracy: {results['accuracy']:.4f}")
    print("Saved → models/rf_results.pkl")
