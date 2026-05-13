import streamlit as st
import pickle
import os
import pandas as pd

# ==========================================================
#  STEP 1 — PAGE CONFIG
# ==========================================================

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("Harshit — Customer Churn Prediction Agent")
st.write("Enter customer transaction details below to predict churn risk.")

# ==========================================================
#  STEP 2 — LOAD MODEL + FEATURES
# ==========================================================

MODEL_PATH   = "C:/Users/harih/OneDrive - University of Hertfordshire/herts study materials/Data Science BootCamp/ai-agent-hackathon-main/models/model.pkl"
FEATURE_PATH = "C:/Users/harih/OneDrive - University of Hertfordshire/herts study materials/Data Science BootCamp/ai-agent-hackathon-main/models/features.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Please train your model first.")
    st.stop()

if not os.path.exists(FEATURE_PATH):
    st.error("features.pkl not found. Please run training notebook.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(FEATURE_PATH, "rb") as f:
    feature_columns = pickle.load(f)

# ==========================================================
#  STEP 3 — INPUT FEATURES
# ==========================================================

st.subheader("Customer Transaction Details")

col1, col2 = st.columns(2)

with col1:
    age              = st.number_input("Age",                      min_value=18,  max_value=80,   value=30)
    num_transactions = st.number_input("Number of Transactions",   min_value=0,   max_value=200,  value=10)
    total_spend      = st.number_input("Total Spend (£)",          min_value=0.0,                 value=500.0)
    monthly_spend    = st.number_input("Monthly Spend (£)",        min_value=0.0,                 value=100.0)

with col2:
    avg_transaction_value = st.number_input("Avg Transaction Value (£)", min_value=0.0,           value=50.0)
    tenure           = st.number_input("Tenure (months)",          min_value=0,   max_value=120,  value=12)
    last_login       = st.number_input("Days Since Last Login",    min_value=0,   max_value=365,  value=5)

# ==========================================================
#  STEP 4 — PREDICT BUTTON
# ==========================================================

if st.button("Predict Churn"):

    try:
        # ==================================================
        #  AUTO FEATURE ALIGNMENT
        # ==================================================

        input_df = pd.DataFrame(columns=feature_columns)
        input_df.loc[0] = 0

        # --- Base features ---
        if "age"                   in input_df.columns: input_df["age"]                   = age
        if "monthly_spend"         in input_df.columns: input_df["monthly_spend"]         = monthly_spend
        if "total_spend"           in input_df.columns: input_df["total_spend"]           = total_spend
        if "num_transactions"      in input_df.columns: input_df["num_transactions"]      = num_transactions
        if "avg_transaction_value" in input_df.columns: input_df["avg_transaction_value"] = avg_transaction_value
        if "tenure_months"         in input_df.columns: input_df["tenure_months"]         = tenure
        if "last_login_days"       in input_df.columns: input_df["last_login_days"]       = last_login

        # --- Engineered features (must match training exactly) ---
        if "activity_score" in input_df.columns:
            input_df["activity_score"] = num_transactions / (tenure + 1)

        if "engagement_ratio" in input_df.columns:
            input_df["engagement_ratio"] = last_login / (tenure + 1)

        if "high_spender_flag" in input_df.columns:
            input_df["high_spender_flag"] = int(total_spend > 500)        # median from training

        if "frequent_user_flag" in input_df.columns:
            input_df["frequent_user_flag"] = int(num_transactions > 10)   # median from training

        if "customer_value_score" in input_df.columns:
            input_df["customer_value_score"] = (total_spend * num_transactions) / (tenure + 1)

        if "spend_per_transaction" in input_df.columns:
            input_df["spend_per_transaction"] = total_spend / (num_transactions + 1)

        if "recency_frequency_ratio" in input_df.columns:                  # ✅ your custom feature
            input_df["recency_frequency_ratio"] = last_login / (num_transactions + 1)

        input_df = input_df.fillna(0)

        # ==================================================
        #  STEP 5 — PREDICTION
        # ==================================================

        prediction = model.predict(input_df)

        # ==================================================
        #  STEP 6 — OUTPUT
        # ==================================================

        st.divider()
        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error("⚠️ High Churn Risk — This customer is likely to churn.")
        else:
            st.success("✅ Low Churn Risk — This customer is likely to stay.")

        # --- Confidence Score ---
        try:
            prob       = model.predict_proba(input_df)[0][1]
            confidence = round(prob * 100, 2)

            st.metric(label="Churn Probability", value=f"{confidence}%")
            st.progress(int(confidence))

            # --- Risk band ---
            if confidence >= 70:
                st.warning("🔴 Risk Band: HIGH — Consider immediate retention action.")
            elif confidence >= 40:
                st.warning("🟡 Risk Band: MEDIUM — Monitor this customer closely.")
            else:
                st.info("🟢 Risk Band: LOW — Customer appears healthy.")

        except:
            st.warning("Confidence score not available for this model.")

        # --- Input summary ---
        with st.expander("View Input Summary"):
            summary = {
                "Age": age, "Total Spend (£)": total_spend,
                "Monthly Spend (£)": monthly_spend, "Num Transactions": num_transactions,
                "Avg Transaction Value (£)": avg_transaction_value,
                "Tenure (months)": tenure, "Days Since Last Login": last_login
            }
            st.table(pd.DataFrame(summary, index=["Value"]).T)

    except Exception as e:
        st.error("Prediction failed. Check feature alignment.")
        st.text(str(e))