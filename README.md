# 🤟 Sign Language Recognition Using ANN

## 📌 Project Overview

This project is a **Sign Language Recognition system** that uses hand landmark features and an **Artificial Neural Network (ANN)** to recognize American Sign Language (ASL) alphabet signs.

The system takes hand landmark coordinates as input, processes them using the same preprocessing pipeline used during model training, and predicts the corresponding sign.

The project recognizes **28 classes**:

* A–Z
* `del`
* `space`

---

## 🎯 Objective

The main objective of this project is to develop a machine learning-based system that can recognize hand signs from hand landmark data and provide the predicted sign along with its confidence score.

---

## 📊 Dataset

The project uses `asl_landmarks_final.csv`.

The dataset contains:

* **2,203 samples**
* **63 input features**
* **1 target variable (`label`)**
* **28 classes**

The 63 features represent **21 hand landmarks**, with each landmark containing:

* X coordinate
* Y coordinate
* Z coordinate

Therefore:

```text
21 landmarks × 3 coordinates = 63 features
```

The target variable represents the corresponding sign:

```text
A, B, C, ... Z, del, space
```

---

## 🔄 Project Workflow

```text
Hand Image
    ↓
MediaPipe Hand Landmark Extraction
    ↓
21 Hand Landmarks
    ↓
63 X, Y, Z Features
    ↓
EDA
    ↓
X / y Separation
    ↓
Train-Test Split
    ↓
Feature Scaling
    ↓
ANN Model
    ↓
Optuna Hyperparameter Tuning
    ↓
Model Evaluation
    ↓
Saved ANN Model
    ↓
Streamlit Application
    ↓
Sign Prediction
```

---

## 🔍 Exploratory Data Analysis

The following EDA techniques were performed:

### Univariate Analysis

* Class distribution
* Feature distributions
* Histograms
* Box plots
* Statistical summary

### Bivariate Analysis

* Landmark feature vs sign class
* Feature-to-feature relationships
* Scatter plots

### Multivariate Analysis

* Correlation heatmap
* Pairplot
* PCA visualization

The dataset was also checked for:

* Missing values
* Duplicate records
* Class imbalance
* Feature correlations

---

## 🧹 Preprocessing

The dataset was divided into:

```text
X → 63 landmark features
y → Sign label
```

The target labels were encoded using `LabelEncoder`.

The input features were scaled using `StandardScaler`.

The scaler was fitted only on the training data and then used to transform the test data.

---

## 🧠 Artificial Neural Network

The project uses a **feed-forward Artificial Neural Network** for multiclass classification.

The baseline architecture consists of:

```text
63 Input Features
        ↓
Dense Layer
        ↓
ReLU Activation
        ↓
Dropout
        ↓
Dense Layer
        ↓
ReLU Activation
        ↓
Dropout
        ↓
28 Output Classes
        ↓
Softmax
```

The output layer contains **28 neurons**, corresponding to the 28 sign classes.

---

## ⚙️ Hyperparameter Optimization

**Optuna** was used for hyperparameter tuning.

The following parameters can be optimized:

* Number of hidden layers
* Number of neurons
* Learning rate
* Dropout rate
* Batch size

The optimized model is compared with the baseline ANN.

---

## 📈 Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report
* Confusion Matrix
* ROC-AUC

### Baseline Model Result

The baseline ANN achieved:

```text
Training Accuracy   : 95.95%
Validation Accuracy : 94.90%
Test Accuracy       : 96.83%
```

These results indicate that the baseline model performs well on unseen test data.

---

## 🖐️ MediaPipe Integration

MediaPipe is used for **hand landmark extraction**.

It detects the hand and identifies **21 landmarks**.

Each landmark provides X, Y and Z coordinates:

```text
21 landmarks
      ×
3 coordinates
      =
63 features
```

These 63 features are passed through the saved scaler and then to the trained ANN.

```text
Webcam / Image
      ↓
MediaPipe
      ↓
21 Landmarks
      ↓
63 Features
      ↓
Scaler
      ↓
ANN
      ↓
Label Encoder
      ↓
Predicted Sign
```

---

## 🖥️ Streamlit Application

A Streamlit application is used as the user interface.

The application supports:

* Image upload
* Camera input
* Hand landmark extraction
* Sign prediction
* Prediction confidence

Example output:

```text
Predicted Sign: A

Confidence: 97.42%
```

---

## 📁 Project Structure

```text
sign-language-recognition-ann/
│
├── app.py
├── sign_language_ann.keras
├── scaler.pkl
├── label_encoder.pkl
├── requirements.txt
├── README.md
│
└── models/
    └── hand_landmarker.task
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow / Keras
* Optuna
* MediaPipe
* OpenCV
* Streamlit
* GitHub
* Render

---

## 🚀 Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/Sagar18122002/sign-language-recognition-ann.git
```

### 2. Navigate to the project

```bash
cd sign-language-recognition-ann
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Saved Model Files

The project uses three important saved objects:

### `sign_language_ann.keras`

The trained ANN model used for sign classification.

### `scaler.pkl`

The `StandardScaler` used during training.

### `label_encoder.pkl`

The encoder used to convert numerical predictions back into the original sign labels.

All three are required for consistent prediction.

---

## ☁️ Deployment

The Streamlit application can be deployed using **Render**.

Deployment workflow:

```text
GitHub Repository
       ↓
Render Web Service
       ↓
Install requirements
       ↓
Run Streamlit
       ↓
Live Web Application
```

### Render Start Command

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

---

## 🔮 Future Scope

Possible future improvements include:

* Real-time continuous sign recognition
* Recognition of complete words and sentences
* Text-to-speech conversion
* Support for more sign-language gestures
* Improved handling of multiple hands
* Mobile application integration
* Improved model performance using larger datasets

---

## 👨‍💻 Author

**D. Sagar**

**Project:** Sign Language Recognition Using ANN

**Technologies:** Python | MediaPipe | ANN | TensorFlow | Streamlit
