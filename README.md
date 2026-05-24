# knn-digits-classifier
KNN Classifier on the Digits dataset — applies StandardScaler for feature normalization, trains with k=5 Euclidean distance, and evaluates on 1,797 handwritten digit samples (0–9) using accuracy score and confusion matrix.
# 🔍 K-Nearest Neighbors Classifier on Digits Dataset

A clean implementation of the **K-Nearest Neighbors (KNN) Classifier** using Python and Scikit-learn. This project applies KNN with feature scaling to classify handwritten digits (0–9), evaluating performance through accuracy score and a confusion matrix.

---

## 🔍 Overview

This notebook walks through a complete KNN classification workflow:

1. **Load** the Digits dataset from Scikit-learn (1,797 handwritten digit samples)
2. **Split** data into training and test sets (75/25, stratified)
3. **Scale** features using StandardScaler to normalize the 64-pixel input values
4. **Train** a KNN classifier with k=5 and Euclidean distance metric
5. **Evaluate** performance with accuracy score and confusion matrix

---

## 📁 Project Structure

```
knn-digits-classifier/
│
├── K-NN_implenentation.ipynb   # Main Jupyter Notebook
└── README.md                   # Project documentation
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| Scikit-learn | Dataset, model, scaling & evaluation |
| NumPy | Array operations |
| Jupyter Notebook | Interactive development environment |

---

## 📊 Dataset — Digits

The built-in **Digits dataset** from Scikit-learn:

| Property | Value |
|---|---|
| Total Samples | 1,797 |
| Classes | 10 (digits 0–9) |
| Features | 64 (8×8 pixel image flattened) |
| Training Samples | 1,347 |
| Test Samples | 450 |

Each sample is an 8×8 grayscale image of a handwritten digit, flattened into 64 pixel-intensity features.

---

## ⚙️ Model Configuration

```python
# Feature Scaling (critical for KNN — distance-based algorithm)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

# KNN Model
KNeighborsClassifier(
    n_neighbors=5,
    metric="euclidean"
)
```

- **`n_neighbors=5`** — classifies based on the 5 closest training samples
- **`metric="euclidean"`** — measures straight-line distance between feature vectors
- **`StandardScaler`** — normalizes all 64 features to zero mean and unit variance; essential for KNN since it is distance-based
- **`stratify=y`** — preserves digit class balance in the train/test split

---

## 📈 Evaluation Metrics

- **Accuracy Score** — overall percentage of correct digit predictions
- **Confusion Matrix** — 10×10 grid showing true vs. predicted labels for each digit class (0–9)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/knn-digits-classifier.git
cd knn-digits-classifier
```

### 2. Install dependencies

```bash
pip install scikit-learn numpy notebook
```

### 3. Launch the notebook

```bash
jupyter notebook K-NN_implenentation.ipynb
```

---

## 💡 Key Concepts Covered

- How KNN works — lazy learning, no explicit training phase
- Why feature scaling is **mandatory** for distance-based algorithms like KNN
- The role of `k` (n_neighbors) in controlling the bias-variance tradeoff
- Euclidean distance as a similarity measure in high-dimensional space
- Evaluating a 10-class classifier with a confusion matrix

---

## 🔗 Related Projects

| Project | Algorithm | Dataset |
|---|---|---|
| [Linear Regression](https://github.com/Mubinbilal/linear-regression-basic) | Regression | Custom |
| [Decision Tree](https://github.com/Mubinbilal/decision-tree-iris) | Tree-based | Iris |
| [Random Forest](https://github.com/Mubinbilal/random-forest-iris) | Ensemble | Iris |
| [SVM](https://github.com/Mubinbilal/svm-breast-cancer) | Kernel-based | Breast Cancer |

---

## 🙌 Acknowledgements

Built as part of a hands-on Machine Learning series to explore and compare core supervised learning algorithms using the Scikit-learn library.
