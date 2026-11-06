"""
Telco Customer Churn Predictor
Streamlit app — predicts whether a customer will churn and returns a High / Medium / Low risk label.
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
)

# ── Load artefacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    base = os.path.dirname(__file__)
    pipeline = joblib.load(os.path.join(base, "model", "churn_pipeline.pkl"))
    num_cols = joblib.load(os.path.join(base, "model", "num_cols.pkl"))
    cat_cols = joblib.load(os.path.join(base, "model", "cat_cols.pkl"))
    return pipeline, num_cols, cat_cols

pipeline, NUM_COLS, CAT_COLS = load_artefacts()

# ── Risk thresholds ───────────────────────────────────────────────────────────
# Based on the trained model's probability distribution:
#   < 0.35  → Low risk
#   0.35–0.60 → Medium risk
#   > 0.60  → High risk
LOW_THRESHOLD  = 0.35
HIGH_THRESHOLD = 0.60

def get_risk_level(prob: float) -> tuple[str, str]:
    """Return (label, colour) based on churn probability."""
    if prob >= HIGH_THRESHOLD:
        return "🔴 HIGH RISK",   "error"
    elif prob >= LOW_THRESHOLD:
        return "🟡 MEDIUM RISK", "warning"
    else:
        return "🟢 LOW RISK",    "success"

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📡 Telco Customer Churn Predictor")
st.markdown(
    "Enter a customer's account details to predict whether they are likely to churn. "
    "The model focuses on **high recall**: tuned to catch as many at-risk customers as possible."
)
st.divider()

# ── Sidebar: about ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About This App")
    st.markdown(
        "**Model:** Random Forest (GridSearchCV tuned)  \n"
        "**Imbalance handling:** `class_weight='balanced'`  \n"
        "**Dataset:** IBM Telco Customer Churn (7,043 customers)  \n"
        "**Primary metric:** Recall (catch churners early)  \n"
    )
    st.divider()
    st.markdown("**Risk Levels**")
    st.error("🔴 HIGH — probability > 60%")
    st.warning("🟡 MEDIUM — probability 35–60%")
    st.success("🟢 LOW — probability < 35%")
    st.divider()
    st.markdown(
        "**Top churn risk factors:**\n"
        "- Month-to-month contract\n"
        "- Short tenure (< 12 months)\n"
        "- High monthly charges\n"
        "- No OnlineSecurity or TechSupport\n"
        "- Electronic check payment"
    )

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    gender         = st.selectbox("Gender",         ["Female", "Male"])
    senior_raw     = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner        = st.selectbox("Partner",        ["Yes", "No"])
    dependents     = st.selectbox("Dependents",     ["No", "Yes"])

with col2:
    st.markdown("**Account & Charges**")
    tenure         = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges= st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total_charges  = st.number_input("Total Charges ($)", 0.0, 9000.0,
                                     float(monthly_charges * max(tenure, 1)), step=1.0)
    contract       = st.selectbox("Contract Type",
                                  ["Month-to-month", "One year", "Two year"])
    paperless      = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment        = st.selectbox("Payment Method",
                                  ["Electronic check", "Mailed check",
                                   "Bank transfer (automatic)",
                                   "Credit card (automatic)"])

with col3:
    st.markdown("**Services**")
    phone_service   = st.selectbox("Phone Service",     ["Yes", "No"])
    multiple_lines  = st.selectbox("Multiple Lines",    ["No", "Yes", "No phone service"])
    internet        = st.selectbox("Internet Service",  ["DSL", "Fiber optic", "No"])
    online_sec      = st.selectbox("Online Security",   ["No", "Yes", "No internet service"])
    online_backup   = st.selectbox("Online Backup",     ["Yes", "No", "No internet service"])
    device_prot     = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support    = st.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
    streaming_tv    = st.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
    streaming_movies= st.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
def build_input_df() -> pd.DataFrame:
    """Build a one-row DataFrame matching the original training columns.
    The pipeline handles all preprocessing (impute → scale → one-hot).
    SeniorCitizen is 0/1 in the original dataset, so we pass integers.
    """
    row = {
        # Numeric
        "tenure"          : tenure,
        "MonthlyCharges"  : monthly_charges,
        "TotalCharges"    : float(monthly_charges * max(tenure, 1))
                           if total_charges == 0 else total_charges,
        # Categorical (passed as raw strings — pipeline encodes them)
        "gender"          : gender,
        "SeniorCitizen"   : 1 if senior_raw == "Yes" else 0,
        "Partner"         : partner,
        "Dependents"      : dependents,
        "PhoneService"    : phone_service,
        "MultipleLines"   : multiple_lines,
        "InternetService" : internet,
        "OnlineSecurity"  : online_sec,
        "OnlineBackup"    : online_backup,
        "DeviceProtection": device_prot,
        "TechSupport"     : tech_support,
        "StreamingTV"     : streaming_tv,
        "StreamingMovies" : streaming_movies,
        "Contract"        : contract,
        "PaperlessBilling": paperless,
        "PaymentMethod"   : payment,
    }
    return pd.DataFrame([row])


if st.button("🔍 Predict Churn Risk", type="primary", use_container_width=True):
    input_df = build_input_df()
    prob = pipeline.predict_proba(input_df)[0][1]
    risk_label, risk_style = get_risk_level(prob)

    st.subheader("Prediction Result")
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        # Risk badge
        if risk_style == "error":
            st.error(f"### {risk_label}\n**Churn Probability: {prob*100:.1f}%**")
        elif risk_style == "warning":
            st.warning(f"### {risk_label}\n**Churn Probability: {prob*100:.1f}%**")
        else:
            st.success(f"### {risk_label}\n**Churn Probability: {prob*100:.1f}%**")

        st.metric(
            "Churn Probability",
            f"{prob*100:.1f}%",
            delta=f"{(prob - 0.265)*100:+.1f}% vs avg (26.5%)",
        )
        st.progress(min(int(prob * 100), 100))

    with res_col2:
        st.markdown("#### Personalised Retention Recommendations")
        tips = []
        if contract == "Month-to-month":
            tips.append("📋 **Offer an annual contract discount**: month-to-month customers churn at 42%.")
        if tenure < 12:
            tips.append("🕐 **Schedule a check-in call**: first 12 months are the highest churn window.")
        if monthly_charges > 70:
            tips.append("💰 **Offer a loyalty pricing review**: high monthly charges are a top churn driver.")
        if online_sec == "No" and internet != "No":
            tips.append("🔒 **Offer Online Security add-on** (first month free).")
        if tech_support == "No" and internet != "No":
            tips.append("🛠️ **Bundle Tech Support**: customers without it churn significantly more.")
        if payment == "Electronic check":
            tips.append("💳 **Incentivise auto-pay**: $3/month discount for bank transfer or credit card.")
        if senior_raw == "Yes":
            tips.append("👴 **Assign a dedicated support contact**: senior customers value personal service.")
        if not tips:
            tips.append("✅ This customer shows low risk. Continue standard engagement.")

        for tip in tips:
            st.markdown(f"- {tip}")

    # Probability breakdown
    st.divider()
    st.markdown("**Probability Breakdown**")
    prob_df = pd.DataFrame({
        "Outcome"    : ["Will NOT Churn", "Will Churn"],
        "Probability": [1 - prob, prob],
    })
    st.bar_chart(prob_df.set_index("Outcome"))

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with Streamlit · Model: Random Forest with class_weight='balanced' + GridSearchCV · "
    "Dataset: IBM Telco Customer Churn · "
    "Focus metric: Recall (catch churners before they leave)"
)
