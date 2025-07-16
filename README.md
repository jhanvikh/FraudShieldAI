# FraudShield AI – Real-Time Transaction Risk Classification

This project demonstrates an end-to-end machine learning pipeline to detect fraudulent credit card transactions using an imbalanced classification dataset.

## 🔍 Project Overview

The goal is to build a classifier that accurately flags fraudulent transactions in real-time using statistical analysis and ML models. The dataset used contains anonymized transaction data with a heavy class imbalance (fraud cases are rare).

## 📌 Key Features

- **EDA & Visualization**: Analyzed class distribution, correlation matrix, and transaction behavior.
- **Data Preprocessing**: Scaled numerical features, removed outliers, and handled imbalance using SMOTE.
- **Modeling**: Trained and compared multiple classifiers (XGBoost, Random Forest) with emphasis on recall and ROC-AUC.
- **Evaluation**: Used metrics such as F1-score, Confusion Matrix, and ROC curve to evaluate models.
- **Simulated Inference**: Classified new unseen transaction rows to simulate real-time fraud detection.

## 🧠 Algorithms Used

- XGBoost Classifier  
- Random Forest Classifier  
- SMOTE (Synthetic Minority Over-sampling Technique)

## 📊 Libraries & Tools

- Python, Pandas, NumPy  
- Scikit-learn, XGBoost, Imbalanced-learn  
- Matplotlib, Seaborn

## 📂 Dataset

- [Kaggle: Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## 🚀 How to Run

```bash
pip install -r requirements.txt
jupyter notebook fraud_detection_model.ipynb
