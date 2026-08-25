# # Melanoma Analysis using Clinical Data and Image-Based Deep Learning with Explainable AI
#
# This notebook reconstructs the final project workflow from the submitted Jupyter/Colab export. It includes clinical-data modelling, SHAP explainability, a SHAP-guided risk score, CNN image classification, and class-imbalance experiments.
#
# **Data note:** large datasets are not stored in this repository. See `data/README.md` for the expected files and folder layout.

# Part 1: Clinical Data Analysis

import pandas as pd

#Load Data
df = pd.read_csv("melanoma.csv")

# Retrieving First Few rows

df.head()

df.info()

df.columns

# Data Cleaning(binary Classification, 1:death, 0:alive)
df['status'] = df['status'].replace({1: 1, 2: 0, 3: 0})

df.isnull().sum()

df = df.dropna()

df['sex'].unique()

df.info()

df.describe()

# Understanding Data(EDA)
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='status', data=df)
plt.title("Survival Distribution")
plt.show()

sns.boxplot(x='status', y='age', data=df)
plt.title("Age vs Survival")
plt.show()

sns.boxplot(x='status', y='thickness', data=df)
plt.title("Tumor Thickness vs Survival")
plt.show()

sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()

# Feature Selection
## Define features and target
X = df[['age', 'sex', 'thickness', 'ulcer']]
y = df['status']

# Train - Test Split
#import library
from sklearn.model_selection import train_test_split

#split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#check shapes
print(X_train.shape)
print(X_test.shape)

# Part 2: Logistic Regression

#import model
from sklearn.linear_model import LogisticRegression

#create model
model = LogisticRegression()

#train model
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Model
## Accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

## Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

## ROC Curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label="AUC = " + str(roc_auc))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Part 3 : Deep learning Analysis

# Import Deep Learning Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build model
model_dl = Sequential()

model_dl.add(Dense(8, activation='relu', input_dim=4))
model_dl.add(Dense(1, activation='sigmoid'))

# Compile Model
model_dl.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train Model
model_dl.fit(X_train, y_train, epochs=50, batch_size=10)

# Evaluate Model
loss, accuracy = model_dl.evaluate(X_test, y_test)
print("Accuracy:", accuracy)

# Part 4: Explainable AI (SHAP)

# Install SHAP
# Colab shell command omitted: pip install shap

import shap

# create explainer
explainer = shap.Explainer(model_dl, X_train)

# Generate Explanations
shap_values = explainer(X_test)

# plot feature importance
shap.summary_plot(shap_values, X_test)

# PART 5: SHAP-GUIDED RISK SCORE FEATURE EXPERIMENT

import pandas as pd

df = pd.read_csv("melanoma.csv")

# Convert status to binary
df['status'] = df['status'].replace({1: 1, 2: 0, 3: 0})

# Create SHAP-guided risk score
df['risk_score'] = (
    df['thickness'] * 0.5 +
    df['age'] * 0.02 +
    df['ulcer'] * 2
)

df[['age', 'thickness', 'ulcer', 'risk_score', 'status']].head()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df[['age_n','thickness_n']] = scaler.fit_transform(
    df[['age','thickness']]
)

df['risk_score'] = (
    0.6 * df['thickness_n']
    + 0.3 * df['age_n']
    + 0.1 * df['ulcer']
)

# Train Model with risk score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

X = df[['age', 'sex', 'thickness', 'ulcer', 'risk_score']]
y = df['status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model_risk = LogisticRegression()
model_risk.fit(X_train, y_train)

y_pred = model_risk.predict(X_test)
y_prob = model_risk.predict_proba(X_test)[:, 1]

print("Accuracy with Risk Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label="AUC = " + str(round(roc_auc, 3)))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve with SHAP-Guided Risk Score")
plt.legend()
plt.show()

# Part 6: Image Classification
# : CNN

import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

import os
print(os.listdir())

# In Colab, ensure the HAM10000 zip/image folders are available in the working directory.

# UnZip Files

# Extract the HAM10000 image archives before running the image-classification section.

import os

print(os.listdir())

folder1 = "/content/HAM10000_images_sub_part 1"
folder2 = "/content/HAM10000_images_sub_part 2"

# Check Number of Images
import os

folder1 = "/content/HAM10000_images_sub_part 1"
folder2 = "/content/HAM10000_images_sub_part 2"

files1 = [f for f in os.listdir(folder1) if f.endswith(".jpg")]
files2 = [f for f in os.listdir(folder2) if f.endswith(".jpg")]

print("Part 1 images:", len(files1))
print("Part 2 images:", len(files2))
print("Total images:", len(files1) + len(files2))

# Load Metadata and create labels
import pandas as pd

df_img = pd.read_csv("HAM10000_metadata.csv")

df_img["label"] = df_img["dx"].apply(lambda x: 1 if x == "mel" else 0)

print(df_img["dx"].value_counts())
print(df_img["label"].value_counts())

# Match only your available 2012 images
all_files = files1 + files2
available_ids = [f.replace(".jpg", "") for f in all_files]

df_filtered = df_img[df_img["image_id"].isin(available_ids)].copy()

print("Matched metadata rows:", len(df_filtered))
print(df_filtered["label"].value_counts())

print("Matched metadata rows:", len(df_filtered))
print(df_filtered["label"].value_counts())

# Load and Preprocess the data
import cv2
import numpy as np

images = []
labels = []

for _, row in df_filtered.iterrows():
    image_name = row["image_id"] + ".jpg"

    path1 = os.path.join(folder1, image_name)
    path2 = os.path.join(folder2, image_name)

    if os.path.exists(path1):
        img_path = path1
    elif os.path.exists(path2):
        img_path = path2
    else:
        continue

    img = cv2.imread(img_path)

    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))

        images.append(img)
        labels.append(row["label"])

# Convert to arrays
X_img = np.array(images)
y_img = np.array(labels)

print(X_img.shape)
print(y_img.shape)
print("Melanoma:", sum(y_img))
print("Non-melanoma:", len(y_img) - sum(y_img))

# Normalize
X_img = X_img / 255.0

X_img.shape

# Train-test split
from sklearn.model_selection import train_test_split

X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
    X_img,
    y_img,
    test_size=0.2,
    random_state=42,
    stratify=y_img
)

print(X_train_img.shape)
print(X_test_img.shape)
print(y_train_img.shape)
print(y_test_img.shape)

# Build CNN Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

cnn_model = Sequential()

cnn_model.add(Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)))
cnn_model.add(MaxPooling2D(2,2))

cnn_model.add(Conv2D(64, (3,3), activation='relu'))
cnn_model.add(MaxPooling2D(2,2))

cnn_model.add(Flatten())
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dropout(0.3))
cnn_model.add(Dense(1, activation='sigmoid'))

# Compile
cnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
history = cnn_model.fit(
    X_train_img,
    y_train_img,
    epochs=10,
    batch_size=32,
    validation_data=(X_test_img, y_test_img)
)

# Evaluation
loss, accuracy = cnn_model.evaluate(X_test_img, y_test_img)
print("CNN Accuracy with 2010 Images:", accuracy)

# PART 7 : CLASS-WEIGHTED CNN

# Check Class Balance
import numpy as np

unique, counts = np.unique(y_train_img, return_counts=True)
print(dict(zip(unique, counts)))

# Compute class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_img),
    y=y_train_img
)

class_weights_dict = dict(enumerate(class_weights))
print(class_weights_dict)

# Build the same CNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

cnn_weighted = Sequential()

cnn_weighted.add(Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)))
cnn_weighted.add(MaxPooling2D(2,2))

cnn_weighted.add(Conv2D(64, (3,3), activation='relu'))
cnn_weighted.add(MaxPooling2D(2,2))

cnn_weighted.add(Flatten())
cnn_weighted.add(Dense(64, activation='relu'))
cnn_weighted.add(Dropout(0.3))
cnn_weighted.add(Dense(1, activation='sigmoid'))

# Compile
cnn_weighted.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train with Class weights
history_weighted = cnn_weighted.fit(
    X_train_img,
    y_train_img,
    epochs=20,
    batch_size=32,
    validation_data=(X_test_img, y_test_img),
    class_weight=class_weights_dict
)

# Evaluate
loss_w, accuracy_w = cnn_weighted.evaluate(X_test_img, y_test_img)
print("Class-Weighted CNN Accuracy:", accuracy_w)

# Add Confusion Matrix and Classification report
from sklearn.metrics import confusion_matrix, classification_report

y_prob_weighted = cnn_weighted.predict(X_test_img)
y_pred_weighted = (y_prob_weighted > 0.5).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_test_img, y_pred_weighted))

print("Classification Report:")
print(classification_report(y_test_img, y_pred_weighted))

y_prob_weighted = cnn_weighted.predict(X_test_img)

print(y_prob_weighted[:20])
print("Minimum:", y_prob_weighted.min())
print("Maximum:", y_prob_weighted.max())
print("Mean:", y_prob_weighted.mean())

mel_df = df_filtered[df_filtered['label'] == 1]
nonmel_df = df_filtered[df_filtered['label'] == 0]

print("Melanoma:", len(mel_df))
print("Non-Melanoma:", len(nonmel_df))

nonmel_sample = nonmel_df.sample(
    n=500,
    random_state=42
)

df_balanced = pd.concat([
    mel_df,
    nonmel_sample
])

print(df_balanced['label'].value_counts())

mel_df = df_filtered[df_filtered['label'] == 1]
nonmel_df = df_filtered[df_filtered['label'] == 0]

print("Melanoma:", len(mel_df))
print("Non-Melanoma:", len(nonmel_df))

# Part 8: Balanced Dataset CNN Experiment

# Create balanced Dataset
mel_df = df_filtered[df_filtered['label'] == 1]
nonmel_df = df_filtered[df_filtered['label'] == 0]

nonmel_sample = nonmel_df.sample(
    n=500,
    random_state=42
)

df_balanced = pd.concat([
    mel_df,
    nonmel_sample
])

print(df_balanced['label'].value_counts())

# Load balanced images
import cv2
import numpy as np
import os

images = []
labels = []

for _, row in df_balanced.iterrows():

    image_name = row["image_id"] + ".jpg"

    path1 = os.path.join(folder1, image_name)
    path2 = os.path.join(folder2, image_name)

    if os.path.exists(path1):
        img_path = path1
    elif os.path.exists(path2):
        img_path = path2
    else:
        continue

    img = cv2.imread(img_path)

    if img is not None:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64,64))

        images.append(img)
        labels.append(row["label"])

X_img_bal = np.array(images)
y_img_bal = np.array(labels)

print(X_img_bal.shape)
print(y_img_bal.shape)

print("Melanoma:", sum(y_img_bal))
print("Non-Melanoma:", len(y_img_bal)-sum(y_img_bal))

# Normalize
X_img_bal = X_img_bal / 255.0

# Train - Test split
from sklearn.model_selection import train_test_split

X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(
    X_img_bal,
    y_img_bal,
    test_size=0.2,
    random_state=42,
    stratify=y_img_bal
)

print(X_train_bal.shape)
print(X_test_bal.shape)

# Build CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

cnn_balanced = Sequential()

cnn_balanced.add(
    Conv2D(32,(3,3),
           activation='relu',
           input_shape=(64,64,3))
)

cnn_balanced.add(MaxPooling2D(2,2))

cnn_balanced.add(
    Conv2D(64,(3,3),
           activation='relu')
)

cnn_balanced.add(MaxPooling2D(2,2))

cnn_balanced.add(Flatten())

cnn_balanced.add(Dense(64, activation='relu'))

cnn_balanced.add(Dropout(0.3))

cnn_balanced.add(Dense(1, activation='sigmoid'))

# Compile
cnn_balanced.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
history_balanced = cnn_balanced.fit(
    X_train_bal,
    y_train_bal,
    epochs=10,
    batch_size=32,
    validation_data=(X_test_bal,y_test_bal)
)

# Evaluate
loss_bal, accuracy_bal = cnn_balanced.evaluate(
    X_test_bal,
    y_test_bal
)

print("Balanced CNN Accuracy:", accuracy_bal)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

y_prob_bal = cnn_balanced.predict(X_test_bal)

y_pred_bal = (
    y_prob_bal > 0.5
).astype(int)

print(confusion_matrix(
    y_test_bal,
    y_pred_bal
))

print(classification_report(
    y_test_bal,
    y_pred_bal
))
