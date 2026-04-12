# BRCA Subtype Classification

Classify breast cancer subtypes from TCGA gene expression (TPM) data.

---

## File Structure

```
brca_project/
│
├── datasets/                        ← place your raw .tsv files here
│   ├── TCGA_BRCA_tpm.tsv
│   └── brca_tcga_pan_can_atlas_2018_clinical_data_filtered.tsv
│
├── models/                          ← auto-created; stores results from each training script
│   ├── rf_results.pkl               ← saved by train_random_forest.py
│   ├── svm_results.pkl              ← saved by train_svm.py
│   └── <model_name>_results.pkl     ← saved by any new train_*.py you add
│
├── src/                             ← training scripts, run from terminal
│   ├── data_preprocessing.py        ← shared: load_data(), preprocess(), get_Xy()
│   ├── train_random_forest.py
│   ├── train_svm.py
│   └── train_<new_model>.py         ← add new models here
│
├── main.ipynb                       ← EDA + loads & displays all model results
└── README.md
```

---

## Workflow

### Step 1 — Train models (terminal)
```bash
cd src
python train_random_forest.py
python train_svm.py
# python train_<new_model>.py
```
Each script saves a results dict (model, predictions, confusion matrix, accuracy, 
hyperparameter search data) to `models/<name>_results.pkl`.

### Step 2 — View everything (notebook)
```bash
# from brca_project/
jupyter notebook main.ipynb
```
The notebook covers:
- **Section 1** — Full EDA (data preview, subtype counts, scatter plots)
- **Section 2** — One subsection per model, loaded from its `.pkl` file

---

## Adding a New Model

1. Create `src/train_<model_name>.py`
2. At the end, save a results dict:
```python
results = {
    "model":    trained_model,
    "X_test":   X_test,
    "y_test":   y_test,
    "y_pred":   y_pred,
    "classes":  trained_model.classes_,
    "accuracy": accuracy_score(y_test, y_pred),
    "conf_mat": confusion_matrix(y_test, y_pred, labels=trained_model.classes_),
    # add any extra keys you want to chart in the notebook
}
with open("../models/<model_name>_results.pkl", "wb") as f:
    pickle.dump(results, f)
```
3. Add a new section in `main.ipynb` under **Section 2** to load and display it.
