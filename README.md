# 🩺 Breast Cancer Detection using Machine Learning & Flask

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask%203.0-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%201.2-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An end-to-end Machine Learning web application designed to predict whether a breast tumor is **Malignant** or **Benign** based on cell nucleus characteristics extracted from digitized fine needle aspirate (FNA) images.

---

## 📌 Project Overview

Early detection of breast cancer plays a vital role in improving survival rates. This repository implements data exploration, feature analysis, model selection, and deployment of a trained Machine Learning model via a Flask web application.

- **Dataset**: Kaggle / UCI Breast Cancer Wisconsin (Diagnostic) Dataset.
- **Model**: Trained Classifier using Scikit-Learn (predicting Malignant vs. Benign).
- **Web App**: Interactive web user interface built with **Flask**, allowing users to input nuclear features and receive immediate predictions.

---

## 📂 Repository Structure

```text
├── Kaggle_Breast_Cancer_dataset_.ipynb  # Jupyter Notebook for EDA, data cleaning & model training
├── app.py                               # Flask application script & prediction API
├── templates/
│   └── home.html                        # Frontend HTML form template for feature input
├── model.pkl                            # Serialized machine learning model
├── scaling_new.pkl                      # Feature scaling pipeline (if applicable)
├── data.csv                             # Breast Cancer Wisconsin dataset
├── requirements.txt                     # Project dependencies
├── run_app.bat                          # One-click Windows batch launcher
├── LICENSE                              # Apache-2.0 License
└── README.md                            # Documentation
```

---

## 📊 Dataset & Feature Inputs

The application utilizes **10 mean features** computed from digitized images of cell nuclei:

| Feature | Description |
| :--- | :--- |
| **Radius Mean** | Mean distance from center to points on the perimeter |
| **Texture Mean** | Standard deviation of gray-scale values |
| **Perimeter Mean** | Mean size of the core tumor perimeter |
| **Area Mean** | Mean area of the tumor nucleus |
| **Smoothness Mean** | Mean of local variation in radius lengths |
| **Compactness Mean** | $\text{perimeter}^2 / \text{area} - 1.0$ |
| **Concavity Mean** | Severity of concave portions of the contour |
| **Concave Points Mean** | Number of concave portions of the contour |
| **Symmetry Mean** | Symmetry score of the cell nucleus |
| **Fractal Dimension Mean** | "Coastline approximation" $- 1$ |

**Target Output:**
- `1` $\rightarrow$ **Malignant (Cancer Detected)**
- `0` $\rightarrow$ **Benign (No Cancer Detected)**

---

## 🚀 Getting Started

### Prerequisites

Ensure you have **Python 3.8+** installed on your system.

### 1. Clone the Repository

```bash
git clone https://github.com/pankajatd/Kaggle_Breast_Cancer_Data.git
cd Kaggle_Breast_Cancer_Data
```

### 2. Set Up Virtual Environment (Optional but Recommended)

- **Windows (Command Prompt / PowerShell):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 💻 Running the Web Application

### Option 1: Direct Python Execution
Run the following command to start the Flask server:

```bash
python app.py
```

The application automatically opens your default browser at `http://127.0.0.1:5000`.

### Option 2: Windows Batch Launcher
If you are on Windows, simply double-click `run_app.bat` or run:

```cmd
run_app.bat
```

---

## ⚙️ Model Training Workflow

The project lifecycle documented in `Kaggle_Breast_Cancer_dataset_.ipynb` includes:
1. **Data Preprocessing & Cleaning**: Dropping irrelevant columns (`id`, `Unnamed: 32`) and encoding diagnosis labels (`M: 1, B: 0`).
2. **Exploratory Data Analysis (EDA)**: Visualizing distributions and feature correlations with Matplotlib and Seaborn.
3. **Feature Selection**: Identifying top discriminative features (such as `concave points_mean`, `area_mean`, `radius_mean`).
4. **Model Training & Evaluation**: Evaluating Logistic Regression, Decision Trees, and Random Forest algorithms using Cross-Validation.
5. **Model Serialization**: Saving the optimized model to `model.pkl` for production deployment.

---

## 📜 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for full details.
