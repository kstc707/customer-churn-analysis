# Customer Churn Analysis — IBM Telco Dataset

Exploratory data analysis identifying the strongest drivers of customer churn in the IBM Telco Customer Churn dataset, as a first phase toward a churn prediction model.

## Dataset

The [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle) — one row per customer, with account, service, and billing attributes plus a `Churn` (Yes/No) label.

## What this notebook covers

1. **Data cleaning** — converts `TotalCharges` from string to numeric, handles resulting missing values, drops the non-predictive `customerID` column.
2. **Exploratory analysis** — churn distribution, and churn broken out by contract type, internet service, online security/tech support, paperless billing, and payment method.
3. **Churn-rate tables** — computes churn % by category to rank which features separate churners from non-churners most strongly.
4. **Correlation analysis** — relationship between tenure, monthly charges, and total charges.
5. **Modeling prep** — binary-encodes the target and one-hot encodes categorical features, staging the dataset for a classification model.

## Key findings

- **Month-to-month contracts** churn at a much higher rate than one- or two-year contracts.
- Customers **without online security or tech support** churn more.
- **Paperless billing** and certain payment methods (electronic check) correlate with higher churn.
- `TotalCharges` and `tenure` are strongly correlated, as expected (charges accumulate over time), while `MonthlyCharges` adds independent signal.

## Status

This is the EDA / churn-driver-identification phase. The dataset is fully cleaned and encoded and ready for the next step: training and evaluating a classification model (e.g. logistic regression, random forest, or a small neural network) to predict churn probability per customer.

## Dependencies

- Python 3, pandas, numpy, matplotlib, seaborn

## Usage

```bash
pip install pandas numpy matplotlib seaborn
jupyter notebook customer_churn_eda.ipynb
```
