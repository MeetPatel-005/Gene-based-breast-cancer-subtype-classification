import sys
import os
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.append(os.path.dirname(__file__))
from data_preprocessing import load_data, preprocess, get_Xy


def find_best_C(X_train, X_test, y_train, y_test, C_range=range(1, 10)):
    svm_class = []
    for i in C_range:
        clf = make_pipeline(StandardScaler(), SVC(C=i))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        svm_class.append([i, accuracy_score(y_test, y_pred)])
    return pd.DataFrame(svm_class, columns=["C parameter", "Accuracy"])


if __name__ == "__main__":
    data, clinic = load_data()
    df = preprocess(data, clinic)
    X, y = get_Xy(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter search
    svm_class = find_best_C(X_train, X_test, y_train, y_test)

    # Train final model
    svm_model = make_pipeline(StandardScaler(), SVC(C=2))
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)

    conf_mat = confusion_matrix(y_test, y_pred_svm, labels=svm_model.classes_)

    # Bundle everything the notebook needs into one results dict
    results = {
        "model": svm_model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred_svm,
        "classes": svm_model.classes_,
        "accuracy": accuracy_score(y_test, y_pred_svm),
        "conf_mat": conf_mat,
        "svm_class_df": svm_class,
    }

    os.makedirs("models", exist_ok=True)
    with open("models/svm_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"SVM Accuracy: {results['accuracy']:.4f}")
    print("Saved → models/svm_results.pkl")
