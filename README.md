# Customer Churn Prediction — IBM Telco Dataset

End-to-end churn analysis on the IBM Telco Customer Churn dataset: exploratory analysis to identify churn drivers, followed by classification models to predict churn probability per customer.

## Dataset

The [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — one row per customer, with account, service, and billing attributes plus a `Churn` (Yes/No) label. ~7,000 customers, ~27% churn rate (imbalanced).

## 1. Exploratory Data Analysis (`customer_churn_eda.ipynb`)

- Cleans `TotalCharges` (stored as text with blanks) and drops the non-predictive `customerID`.
- Breaks down churn by contract type, internet service, online security/tech support, paperless billing, and payment method.
- Computes churn-rate tables per category and a correlation heatmap of the numeric features.

**Key findings:**
- **Month-to-month contracts** churn at a much higher rate than one- or two-year contracts.
- Customers **without online security or tech support** churn more.
- **Paperless billing** and electronic-check payment correlate with higher churn.

## 2. Modeling (`churn_modeling.py`)

Trained two classifiers on the one-hot-encoded, 80/20 train-test split (class-weighted to handle the ~27% churn imbalance):

| Model | Accuracy | ROC AUC | F1 (churn class) | Recall (churn class) |
|---|---|---|---|---|
| Random Forest | **74.8%** | **0.839** | 0.62 | 0.78 |
| Logistic Regression | 72.6% | 0.835 | 0.61 | 0.80 |

Class weighting was used deliberately: in churn prediction, missing an at-risk customer (false negative) is usually costlier than a false alarm, so both models are tuned to catch ~78-80% of actual churners (recall) rather than optimizing for raw accuracy alone.

**Top predictive features (Random Forest):** `tenure`, `TotalCharges`, having a **two-year contract**, `MonthlyCharges`, **fiber-optic internet**, and paying by **electronic check** — consistent with the EDA findings above.

## Files

- `customer_churn_eda.ipynb` — data cleaning and exploratory analysis.
- `churn_modeling.py` — model training and evaluation.
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` — source dataset.

## Dependencies

- Python 3, pandas, numpy, scikit-learn, matplotlib, seaborn

## Usage

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook customer_churn_eda.ipynb   # EDA
python churn_modeling.py                    # train & evaluate models
```
