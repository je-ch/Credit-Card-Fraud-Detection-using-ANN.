# Credit Card Fraud Detection using ANN

A deep learning model to detect fraudulent credit card transactions using a Single Feed-Forward Multi-Layer Perceptron (MLP) ANN, trained on a highly imbalanced real-world dataset of 284,807 transactions.

---

## Problem Statement

Credit card fraud causes billions in losses annually. With only 0.17% of transactions being fraudulent, traditional models tend to ignore the minority class entirely - prioritizing overall accuracy over actual fraud detection. This project builds an ANN that reliably detects fraud under extreme class imbalance, focusing on high recall and balanced precision to minimize false negatives while avoiding excessive false alarms.

---

## Dataset

- **Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Features:** 30 - Time, V1–V28 (PCA-transformed), Amount
- **Class distribution:** 284,315 legitimate (99.83%) vs 492 fraudulent (0.17%)

---

## Approach

1. **Exploratory Data Analysis (EDA)**
   - Visualized class imbalance, transaction amount distributions by class, feature correlations, and KDE plots for PCA components (V1–V4)

2. **Preprocessing**
   - Standardized `Amount` using StandardScaler; dropped raw `Amount` and `Time`
   - Stratified 80/20 train-test split to preserve fraud ratio in both sets

3. **Handling Class Imbalance**
   - Applied `compute_class_weight('balanced')` from Scikit-learn to assign higher importance to rare fraud cases during training - passed directly to the model via `class_weight` parameter

4. **Model Architecture**

   | Layer | Details |
   |-------|---------|
   | Input + Dense | 32 neurons, ReLU activation |
   | Dropout | 30% |
   | Dense | 16 neurons, ReLU activation |
   | Dropout | 30% |
   | Output | 1 neuron, Sigmoid activation |

   - Optimizer: Adam (lr=0.001)
   - Loss: Binary Crossentropy
   - Epochs: 20 | Batch Size: 2048

5. **Evaluation**
   - Classification Report (Precision, Recall, F1-score)
   - ROC-AUC Score
   - Precision-Recall Curve
   - Confusion Matrix
   - Training/Validation Loss & Accuracy curves

---

## Results

| Metric | Score |
|--------|-------|
| Overall Accuracy | 98% |
| ROC-AUC Score | 0.977 |
| Fraud Class Recall | 91% |
| Dataset Size | 284,807 transactions |

> High recall on the fraud class (91%) was prioritized - missed fraud carries a far higher cost than a false alarm.

---

## Tools & Libraries

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

---

## How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/je-ch/Credit-Card-Fraud-Detection-using-ANN.git
   cd credit-card-fraud-detection
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Add the dataset
   - Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Place it in the root directory

4. Run the notebook
   ```bash
   jupyter notebook Credit_Card_Fraud_Detection_using_ANN.ipynb
   ```

---

## Project Structure

```
credit-card-fraud-detection/
│
├── Credit_Card_Fraud_Detection_using_ANN.ipynb
├── requirements.txt
└── README.md
```

---

## Key Takeaway

Class weighting proved more effective than SMOTE for this architecture - directly penalizing misclassification of fraud during training rather than synthetically oversampling. The model achieves strong fraud detection (91% recall) while maintaining high overall accuracy, demonstrating a practical approach to high-stakes imbalanced classification.
