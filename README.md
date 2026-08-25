# Melanoma Analysis: Machine Learning, Deep Learning & Explainable AI

An end-to-end MSc Data Science project exploring melanoma prediction using **clinical data** and **dermoscopic images**. The project combines classical machine learning, neural networks, CNNs, SHAP explainability, feature engineering, and class-imbalance analysis.

![Project workflow](assets/workflow.png)

## Project objectives

- Analyse clinical melanoma data and build a Logistic Regression baseline.
- Compare a feed-forward neural network with the classical ML model.
- Use **SHAP** to explain influential clinical features.
- Create a SHAP-guided risk score using tumor thickness, age, and ulceration.
- Build a **CNN** for binary melanoma classification using HAM10000 images.
- Investigate class imbalance using class weights and a balanced training subset.
- Evaluate models with accuracy, confusion matrices, precision, and recall.

## Key results

| Model | Approx. result | Key observation |
|---|---:|---|
| Logistic Regression | 78.05% accuracy | Strong interpretable baseline |
| ANN | 73–78% accuracy | Non-linear clinical model |
| SHAP-guided risk-score model | 75.6% accuracy | Explainable feature-engineering experiment |
| CNN (2,010 images) | 91.29% accuracy | Highest overall accuracy |
| Class-weighted CNN | ~91% accuracy, 0 melanoma recall | Accuracy was misleading under severe imbalance |
| Balanced CNN | ~76% accuracy, 17% melanoma recall | Detected melanoma cases missed by the high-accuracy models |

## Why the imbalance result matters

The original matched image subset contained **1,835 non-melanoma** and **175 melanoma** images. A model could therefore achieve high accuracy while missing the minority disease class. The balanced experiment showed the trade-off clearly: overall accuracy fell, but melanoma detection improved.

![CNN confusion matrix](assets/cnn_confusion_matrix.png)

## Explainable AI

SHAP was used to identify influential clinical predictors. Tumor thickness, age, and ulceration were among the most important features and were used in a risk-score experiment.

![Risk score distribution](assets/risk_score_distribution.png)

## Repository structure

```text
.
├── README.md
├── PROJECT_SUMMARY.md
├── requirements.txt
├── notebooks/
│   ├── melanoma_analysis.ipynb
│   └── melanoma_analysis_outputs.html
├── assets/
│   ├── workflow.png
│   ├── cnn_confusion_matrix.png
│   └── risk_score_distribution.png
└── data/
    └── README.md
```

## Tools used

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, TensorFlow/Keras, SHAP, and OpenCV.

## Running the notebook

1. Create a Python environment and install the packages in `requirements.txt`.
2. Obtain the clinical melanoma dataset and HAM10000 metadata/images.
3. Place the files in your working environment.
4. Update the image-folder paths if your directory structure differs from the original Colab setup.
5. Run `notebooks/melanoma_analysis.ipynb`.

## Limitations

- Only a subset of HAM10000 images was used because of computational constraints.
- The image task is binary rather than multi-class.
- The CNN architecture is intentionally simple rather than a state-of-the-art transfer-learning model.
- The work is an academic analysis and is **not a clinical diagnostic system**.

## Future improvements

Transfer learning (for example ResNet/EfficientNet), stronger data augmentation, focal loss, larger image resolution, full-dataset training, Grad-CAM, and a multimodal model combining clinical and image features.
