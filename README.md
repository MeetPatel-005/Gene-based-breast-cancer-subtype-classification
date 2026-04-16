# 🧬 Gene-Based Breast Cancer Subtype Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green)

A comprehensive machine learning pipeline and interactive web application for the accurate classification of breast cancer intrinsic subtypes (Luminal A, Luminal B, HER2-enriched, Basal-like, and Normal-like) using TCGA gene expression (TPM) data.

This project focuses on leveraging high-dimensional transcriptomic datasets to establish robust and clinical-grade predictive models. By transitioning from raw TCGA `.tsv` clinical expression datasets to a finalized Stacked Ensemble method, this suite offers both an in-depth analytical review and a seamless real-time visualization interface.

---

## 🌟 Project Overview

- **End-to-End ML Pipeline**: Automated preprocessing of clinical and transcriptomic data, implementing techniques such as robust feature scaling and SMOTE to balance class representation.
- **Ensemble & Deep Models**: Integrates diverse and high-performing algorithms including Random Forest, Support Vector Machines, XGBoost, LightGBM, custom Neural Networks, and two powerful Stacking Ensembles for maximum classification accuracy.
- **Deep EDA Insights**: An expansive Jupyter notebook `main.ipynb` equipped with deep dimensionality topology visualizations, class-balance reviews, PCA distributions, and extensive metric evaluations across all trained models.
- **Production-Ready Web App**: 
  - A comprehensive Flask ecosystem `brca_webapp` serving as the frontend bridge to our sequential prediction inference.
  - A premium, dark-themed, glassmorphism UI paired with smooth CSS micro-animations.
  - Interactive **Demo Mode** allows instantaneous testing using representative subtype data without the need to prepare patient files manually.

---

## 📁 Repository Structure

```text
brca_project/
│
├── datasets/                                 ← Location for raw .tsv datasets (TCGA profiles & clinical data)
│
├── models/                                   ← Serialized machine learning models and pipelines (.pkl)
│   ├── lgbm_results.pkl
│   ├── nn_results.pkl
│   ├── rf_results.pkl
│   ├── stacking_improved_results.pkl
│   ├── stacking_results.pkl
│   ├── svm_results.pkl
│   └── xgb_results.pkl
│
├── src/                                      ← Core model training and processing logic
│   ├── data_preprocessing.py                 ← Shared utilities: loading data, preprocessing, SMOTE
│   ├── train_lightgbm.py                     
│   ├── train_neural_network.py               
│   ├── train_random_forest.py                
│   ├── train_stacking.py                     
│   ├── train_stacking_improved.py            ← Optimized meta-learner pipeline integration
│   ├── train_svm.py                          
│   └── train_xgboost.py                      
│
├── main.ipynb                                ← Complete Jupyter Notebook for EDA, visualization & results matching
│
├── brca_webapp/                              ← 🌐 The Interactive Flask Application Sub-system
│   ├── app.py                                ← Main Flask server governing model loading & API inference
│   ├── models/                               ← Lightweight symlinks or copies of models for sequential inference
│   ├── data/                                 ← Benchmark standard representation data for UI Demo Mode
│   ├── static/                      
│   │   ├── style.css                         ← Highly customized UI aesthetic (Dark mode, glass effects)
│   │   └── script.js                         ← Asynchronous model invocation handling & frontend flow
│   ├── templates/
│   │   └── index.html                        ← HTML layout
│   └── uploads/                              ← Secure, isolated landing zone for user `.csv` inferences
│
└── README.md                                 
```

---

## 🚀 Getting Started

### 1. Training the Models
Make sure your raw datasets are placed inside the `datasets/` directory. Run the training scripts in `src/` to trigger the preprocessing pipeline and construct your model `.pkl` files.

```bash
cd src
python train_random_forest.py
python train_svm.py
python train_lightgbm.py
python train_neural_network.py
python train_xgboost.py

# Run ensembling modules last
python train_stacking.py
python train_stacking_improved.py
```
*Each script automatically executes cross-validation, predicts on your holdout set, and saves the detailed result dictionaries directly into `/models`.*

### 2. Exploring Data & Outcomes
Check out the centralized `main.ipynb` to absorb high-level dimensionality metrics, PCA distributions, and evaluate precisely rendered confusion matrices corresponding to the models you just trained!

```bash
# Provide this from your root brca_project folder
jupyter notebook main.ipynb
```

### 3. Running the Web Application
Start up the web interface to spin up the sequentially stable interactive endpoint. This handles inference dynamically without overwhelming background resources. 

```bash
cd brca_webapp
python app.py
```
**Access the Web App via:** `http://localhost:5000`

---

## 💻 Web App Functionality

Integrating heavy Machine Learning algorithms with an active web-server can lead to significant resource bottlenecking. The `brca_webapp` implements stability measures explicitly designed for processing high-dimensional TPM models on the fly:

- **Interactive Demo Validation**: Pre-configured diagnostic endpoints allow users to rapidly select 1 of 5 exact tumor examples (Basal, HER2, LumA, LumB, Normal) and visualize the model formulating predictions across each algorithm smoothly in real-time.
- **Sequential Execution Pipeline**: Instead of executing multithreaded inferences that instantly exhaust standard web memory, the script strategically queues LightGBM, RF, SVM, XGBoost, Network logic, and the Stacked Ensemble sequentially, guaranteeing 100% stability without dropping concurrent API sessions.

---

## 🛠 Model Expansion & Results

To benchmark new classifier topologies:
1. Generate an encapsulated `src/train_<algorithm>.py` that utilizes the established `data_preprocessing.py`.
2. Evaluate metrics on `X_test` / `y_test`.
3. Store the output dictionary (containing the `trained_model`, `confusion_matrix`, and `accuracy`) using pickle in `models/<algorithm>_results.pkl`.
4. Run `main.ipynb` to digest the newly saved object, which will auto-render precision/recall charts parallel to the other implemented algorithms.

---

## 🤝 Contributions
Pull requests, issues, and exploratory forks are highly welcomed. Let's conquer breast cancer informatics together!
