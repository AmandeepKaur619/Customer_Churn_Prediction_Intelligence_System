import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title ="Customer Churn Prediction System",
    layout ="wide"
)

st.title("Customer Churn Prediction System")
st.subheader("Customer Churn Prediction and Retention Analytics")

st.write("Predict customer churn, identify high-risk customers, "
    "analyze churn drivers, and prioritize retention actions."
)

model = joblib.load("models/customer_churn_model.pkl")

preprocessor = joblib.load("models/customer_churn_preprocessor.pkl")

customer_data = pd.read_csv("data/customer_churn_risk_analysis.csv")

total_customers = len(customer_data)

high_risk_customers= (customer_data['RiskLevel'] =="High Risk").sum()

priority_1_customers= (customer_data['RetentionPriority'] =="Priority 1").sum()

revenue_at_risk= (customer_data['ExpectedRevenueAtRisk'].sum())

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{total_customers:,}")

col2.metric("High-Risk Customers",f"{high_risk_customers:,}")

col3.metric("Priority 1 Customers", f"{priority_1_customers:,}")

col4.metric("Revenue at Risk", f"{revenue_at_risk:,.0f}")

st.header("Customer Risk Analysis")

st.dataframe(customer_data[
        [
            "customerID",
            "ChurnProbability",
            "RiskLevel",
            "ExpectedRevenueAtRisk",
            "RetentionPriority"
        ]
    ],
    use_container_width=True
)
