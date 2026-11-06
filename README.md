# 📡 Telco Customer Churn Predictor

> **Live App:** [Click](https://telco-churn-predictor-123.streamlit.app/)

A complete end-to-end machine learning project from raw data to a live deployed web app — built to help a telecom company predict which customers are about to leave so they can intervene before it's too late.

---

## 🎯 Business Problem

A telecom company is losing customers every month. This project answers two questions:

1. **Why** do customers churn? (Exploratory Data Analysis)
2. **Who** will churn next? (Machine Learning Prediction)

---

## 📊 Dataset

**IBM Telco Customer Churn** — `telco_churn.csv`
- 7,043 customers · 20 features · target: `Churn` (Yes / No)
- Churn rate: **26.5%**
- Source: IBM Sample Dataset (available on Kaggle)

---

## 🗂️ Project Structure

```
telco-churn/
├── app.py                      # Streamlit web app
├── requirements.txt            # Python dependencies
├── telco_churn.csv             # Dataset
├── Telco_Churn_Analysis.ipynb  # Full analysis notebook
├── model/
│   ├── churn_pipeline.pkl      # Trained sklearn Pipeline (preprocessor + Random Forest)
│   ├── num_cols.pkl            # Numeric column names
│   └── cat_cols.pkl            # Categorical column names
└── README.md
```

---

## 📓 Notebook Walkthrough (`Telco_Churn_Analysis.ipynb`)

| Step | What it covers |
|------|----------------|
| 1 | Load data (`telco_churn.csv`) |
| 2 | First look — dtypes, duplicates, missing values |
| 3 | Data cleaning — fix `TotalCharges`, drop `customerID`, encode target |
| 4 | Exploratory Data Analysis: churn by contract, tenure, charges, payment method |
| 5 | Train/test split (80/20, stratified) |
| 6 | Build sklearn Pipeline: `SimpleImputer → StandardScaler → OneHotEncoder` |
| 7 | Compare three models: Logistic Regression, Decision Tree, Random Forest |
| 8 | Improve Recall with `class_weight='balanced'` on all three models |
| 9 | Tune best model with **GridSearchCV** (5-fold CV, optimised for Recall) |
| 10 | Evaluate: confusion matrix, classification report, feature importance |


---

## 🏆 Model Results

| Model | Accuracy | Recall (Churn) | ROC-AUC |
|-------|----------|----------------|---------|
| Logistic Regression (balanced) | ~0.74 | ~0.76 | ~0.83 |
| Decision Tree (balanced) | ~0.72 | ~0.73 | ~0.75 |
| **Random Forest (balanced + GridSearchCV)** | ~0.78 | **~0.78** | **~0.84** |

> **Why Recall?** Missing a churner costs more than falsely flagging a loyal customer.
> A missed churner = lost revenue. A false positive = a small retention offer.

> **Why `class_weight='balanced'`?** Only 26.5% of customers churn. Without balancing,
> models optimise for the majority class and miss many churners.
> `class_weight='balanced'` automatically penalises missing a churner more heavily.

---

## 🔑 Key Findings (EDA)

- **Month-to-month customers churn at ~42%** vs ~3% for 2-year contracts
- **Short tenure = high churn** — most churners leave in the first 12 months
- **High monthly charges** correlate strongly with churn
- **Electronic check users** churn at ~45% — highest of any payment method
- **No OnlineSecurity + Fiber optic** is a high-risk combination

---

## 💡 Business Recommendations

1. **Incentivise annual contracts** — offer a discount in the first 3 months (biggest lever)
2. **Early intervention** — trigger retention outreach at months 3, 6, and 9
3. **Loyalty pricing reviews** — proactively offer better plans to high-charge customers
4. **Promote auto-pay** — $3–5/month discount to switch from electronic check
5. **Bundle OnlineSecurity/TechSupport** into Fiber plans as default

---

## 🌐 Streamlit App Features

- Enter any customer's account details across demographics, services, and billing
- Get a **High / Medium / Low** churn risk label with probability score
- See personalised retention recommendations based on that customer's risk factors

---

## 🚀 Run Locally

```bash
git clone https://github.com/armankhan8150/telco-churn-predictor.git
cd telco-churn-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## 🛠️ Tech Stack

`Python` · `pandas` · `scikit-learn` · `Streamlit` · `matplotlib` · `seaborn` · `joblib`

---

*Dataset: IBM Telco Customer Churn. Built as a portfolio project for PSEB AI Bootcamp.*
