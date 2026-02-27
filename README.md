# 💳 Fraud Detection Project
### Financial Transaction Fraud Detection using Machine Learning

---

## 📌 Project Overview

This project focuses on detecting fraudulent financial transactions using a simulated dataset of **6,362,620 transactions** over a **30-day period**. The dataset contains various transaction types including CASH-IN, CASH-OUT, DEBIT, PAYMENT, and TRANSFER. The goal is to build a machine learning model that accurately identifies fraudulent transactions and provides actionable business insights to prevent financial fraud.

---

## 📂 Dataset Description

| Column | Description |
|--------|-------------|
| `step` | Unit of time (1 step = 1 hour). Total 744 steps (30 days) |
| `type` | Transaction type: CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER |
| `amount` | Transaction amount in local currency |
| `nameOrig` | Customer who initiated the transaction |
| `oldbalanceOrg` | Sender's balance before the transaction |
| `newbalanceOrig` | Sender's balance after the transaction |
| `nameDest` | Recipient of the transaction |
| `oldbalanceDest` | Recipient's balance before the transaction |
| `newbalanceDest` | Recipient's balance after the transaction |
| `isFraud` | Target variable — 1 if fraudulent, 0 if legitimate |
| `isFlaggedFraud` | Business rule flag — flags transfers above 200,000 |

- **Total Rows:** 6,362,620
- **Total Columns:** 11
- **Fraud Cases:** ~8,213 (0.13% of total transactions)

---

## 🔍 Key Findings from EDA

- Fraud **only occurs in CASH_OUT and TRANSFER** transaction types
- Fraudsters aim to **completely drain accounts** — making zero balance after transaction a strong fraud signal
- **16 probe transactions** were found with zero amount — all were fraudulent (probe fraud pattern)
- **Merchant accounts** (nameDest starting with 'M') have no balance information — this is expected and valid
- Amount outliers (5.31% of data) show a **fraud rate of 1.14%** — nearly 9x higher than the overall average

---

## 🧹 Data Cleaning

- **No actual missing values** found in the dataset
- **Zero values** were carefully analyzed — legitimate zeros (mathematically justified) were kept, suspicious zeros were imputed using **median by transaction type**
- **287,929 suspicious zeros** in `newbalanceDest` were identified and imputed
- **16 zero-amount transactions** were flagged as probe fraud and kept as a feature
- **Merchant accounts** were excluded from destination balance imputation
- **Outliers** were retained since they represent real high-value transactions and were handled using **log transformation** on the amount column

---

## ⚙️ Feature Engineering

New features created to improve model performance:

| Feature | Description |
|---------|-------------|
| `errorBalanceDest` | Balance mismatch at destination — strong fraud signal |
| `amount_log` | Log-transformed amount to handle skewness |
| `isZeroAmount` | Flag for zero-amount probe transactions |
| `isMerchant` | Flag for merchant destination accounts |
| `bothZeroOrig` | Flag when sender balance is zero before and after transaction |
| `type_*` | One-hot encoded transaction type columns |

**Dropped columns:** `nameOrig`, `nameDest` (ID columns with no predictive value), `isFlaggedFraud` (business rule, not a real predictor), `oldbalanceOrg`, `oldbalanceDest` (highly correlated with new balance columns)

---

## 📊 Multicollinearity Treatment

Correlation matrix revealed:
- `oldbalanceOrg` vs `newbalanceOrig` = **1.00** (perfect correlation)
- `oldbalanceDest` vs `newbalanceDest` = **0.98** (near perfect)

**Solution:** Dropped redundant old balance columns and created error balance features. Final correlation matrix showed all values below 0.50 — multicollinearity resolved.

---

## 🤖 Model

**Algorithm:** Random Forest Classifier

**Why Random Forest?**
- Handles class imbalance well with `class_weight='balanced'`
- Robust on large datasets (6+ million rows)
- Provides feature importance scores directly
- No need for feature scaling
- Handles non-linear relationships

**Configuration:**
```python
RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

**Train/Test Split:** 80% training, 20% testing with stratification to maintain fraud ratio

---

## 📈 Model Performance

### Default Threshold (0.5)
| Metric | Score |
|--------|-------|
| Precision (Fraud) | 0.95 |
| Recall (Fraud) | 0.68 |
| F1-Score (Fraud) | 0.79 |
| AUC-ROC | 0.9347 |

### Optimized Threshold (0.3)
| Metric | Score |
|--------|-------|
| Precision (Fraud) | 0.90 |
| Recall (Fraud) | 0.71 |
| F1-Score (Fraud) | 0.80 |
| AUC-ROC | 0.9347 |

### Confusion Matrix (Threshold 0.3)
|  | Predicted Not Fraud | Predicted Fraud |
|--|---------------------|-----------------|
| **Actual Not Fraud** | 1,270,826 | 55 |
| **Actual Fraud** | 529 | 1,114 |

> **Note:** Recall is prioritized over precision in fraud detection — it is more costly to miss a fraud than to flag a legitimate transaction for review.

---

## 🏆 Key Fraud Predictors (Feature Importance)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | newbalanceDest | 0.1424 |
| 2 | bothZeroOrig | 0.1274 |
| 3 | step | 0.1176 |
| 4 | amount | 0.1163 |
| 5 | newbalanceOrig | 0.1052 |
| 6 | errorBalanceDest | 0.1010 |
| 7 | type_TRANSFER | 0.06 |
| 8 | type_CASH_OUT | 0.05 |

**Why do these make sense?**
- `newbalanceDest` — Destination account balance pattern reveals fraud collection accounts
- `bothZeroOrig` — Fraudsters drain accounts completely to zero
- `step` — Time of transaction matters; fraud concentrates in specific hours
- `amount` — Large transaction amounts are 9x more likely to be fraud
- `errorBalanceDest` — Balance mismatches indicate manipulated transactions

---

## 🛡️ Prevention Recommendations

1. **Two-Step Verification** — Implement real-time notification and confirmation for all CASH_OUT and TRANSFER transactions, similar to Google login alerts
2. **Unusual Behavior Flagging** — Flag accounts that suddenly use transaction types they have never used before
3. **Restrict Full Balance Withdrawal** — Prevent 100% account balance withdrawal in a single transaction or within a short time window
4. **Daily Transaction Limits** — Set dynamic limits based on account history and transaction frequency
5. **Suspicious Destination Account Monitoring** — Flag newly opened accounts with zero balance receiving large amounts from multiple sources
6. **Real-time ML Scoring** — Deploy the Random Forest model to score every CASH_OUT and TRANSFER in real-time before processing

---

## 📏 Measuring Prevention Effectiveness

1. **Fraud Rate Ratio** — Track weekly fraud rate (fraud/total transactions). Dramatic decline confirms prevention is working
2. **Financial Loss Comparison** — Compare monthly monetary losses before and after implementation
3. **Two-Step Denial Rate** — Track how many customers actively deny transactions via notification — high denial rate proves fraud is being caught
4. **Limit Issue Monitoring** — Track customer requests to increase limits, especially merchant accounts
5. **Model Accuracy Monitoring** — Retrain model monthly and maintain AUC above 0.90
6. **False Positive Rate** — Monitor legitimate transactions being incorrectly blocked
7. **Monthly Dashboard** — Track all metrics together and compare against pre-implementation baseline

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Platform:** Google Colab
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn

---

## 📁 Project Structure

```
fraud-detection/
│
├── data/
│   └── transactions.csv          # Raw dataset
│
├── notebooks/
│   └── Fraud_Detection.ipynb     # Main project notebook
│
└── README.md                     # Project documentation
```

---

## 👤 Author

**Anirudh**
Internship Project — Fraud Detection using Machine Learning
