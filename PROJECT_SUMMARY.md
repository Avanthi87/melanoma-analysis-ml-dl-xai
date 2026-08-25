# Project Summary

This MSc Data Science project combines clinical-data modelling and dermoscopic image classification for melanoma analysis.

## Core components

- Logistic Regression baseline on clinical variables
- Feed-forward ANN for clinical prediction
- SHAP explainability to identify influential features
- SHAP-guided risk-score feature engineering
- CNN classification of HAM10000 images
- Class-weighted CNN experiment
- Balanced-dataset CNN experiment to improve minority-class melanoma detection

## Main findings

- Logistic Regression achieved approximately **78% accuracy** on the clinical task.
- The CNN trained on 2,010 matched images achieved approximately **91.29% accuracy**.
- High overall CNN accuracy masked severe class imbalance: the class-weighted model still failed to detect melanoma cases in the reported test split.
- Balancing the dataset reduced overall accuracy to about **76%** but improved melanoma recall to about **17%**, demonstrating why recall matters in medical classification.
- SHAP highlighted tumor thickness, age, and ulceration as important clinical features and motivated an interpretable risk-score experiment.

## Portfolio takeaway

The project demonstrates end-to-end data preparation, classical machine learning, neural networks, convolutional neural networks, explainable AI, class-imbalance analysis, and model evaluation beyond accuracy.
