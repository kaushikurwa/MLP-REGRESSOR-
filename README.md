# Medical Insurance Charges Prediction (MLP Regressor)

This project uses a **Multi-Layer Perceptron (MLP) Regressor** which is a type of neural network to predict **medical insurance costs** based on factors like age, BMI, smoking status, and region. It shows how to preprocess data, build a regression model, and evaluate it using real metrics. Built as part of a **Problem-Based Learning** module at NMAM Institute of Technology.

---

## 🚀 Overview

* **Tech Stack:**

  * `Python` — Core language for ML
  * `pandas` — Data handling
  * `scikit-learn` — MLP Regressor, preprocessing, splitting
  * `NumPy` — Numerical operations
  * `matplotlib` — Optional for EDA and plots

---

## 🧩 Project Structure & Flow

### 1️⃣ Data Preprocessing

* **Features:**

  * `age` — Age of the individual
  * `sex` — Gender (male/female, label encoded)
  * `bmi` — Body Mass Index
  * `children` — Number of dependents
  * `smoker` — Smoking status (yes/no, label encoded)
  * `region` — Location region (label encoded)
  * `charges` — Insurance cost (target)

* **Steps:**

  * Label Encode categorical columns (`sex`, `smoker`, `region`)
  * Normalize numerical columns using Min-Max Scaler
  * Split data: 80% train, 20% test

### 2️⃣ MLP Regressor

* 3 hidden layers: 200 → 100 → 50 neurons
* Activation: ReLU
* Optimizer: Backpropagation with Gradient Descent
* Max Iterations: 2000

### 3️⃣ Evaluation

* **Metrics:**

  * Mean Squared Error (MSE)
  * Total Squared Error (TSE)
  * R² Score

---

## 📊 Results

| Metric   | Training | Testing |
| -------- | -------- | ------- |
| MSE      | 0.00488  | 0.00622 |
| TSE      | 5.22     | 1.66    |
| R² Score | 86.72%   | 84.26%  |

---


## 🎯 Purpose

This project is **educational only**. It shows how to combine basic preprocessing with a neural network regressor for a real-world prediction task in healthcare analytics.

---

## 📂 References

* Kaggle Medical Insurance Dataset
* Related literature: SGTM, Ridge Regression, Gradient Boosting for healthcare cost prediction.

---

## 📌 To Do / Possible Extensions

* Add hyperparameter tuning for better accuracy.
* Compare MLP with other regressors (XGBoost, SVR, etc.).
* Build an interactive web app for user input.
* Deploy with Flask or Streamlit.

---

## 🏫 Institution

NMAM Institute of Technology, Nitte (Deemed to be University)

---

## 📜 License

This project is for **educational use only** and not for production or commercial deployment.

---

**Stay curious and may your MLPs converge faster than your coffee cools!**
