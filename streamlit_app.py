import joblib
import streamlit as st
import pandas as pd

# Load trained model
model = joblib.load("churn_dt_model.pkl")

st.title("Customer Churn Prediction App")
st.write("Enter customer details to predict churn risk")
st.info("Use the controls below to simulate different customer profiles.")

# User Inputs 
tenure = st.slider("Tenure (months)", 0, 72, 12)
if tenure == 0:
    st.warning("Tenure is 0 months. This may indicate a new customer.")
monthly_charges = st.slider("Monthly Charges ($)", 0, 200, 70)
total_charges = st.slider("Total Charges ($)", 0, 10000, 1000)
if total_charges == 0 and tenure > 12:
    st.warning(
        "Total charges are 0 despite long tenure. "
        "Please verify if the input is correct."
    )

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

st.caption("All inputs must be reviewed before generating a prediction.")

#  Predict button 
if st.button("Predict Customer Churn"):

    # Create input dataframe
    input_data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method
    }

    df_input = pd.DataFrame([input_data])

    # One-hot encoding
    df_input = pd.get_dummies(df_input)

    # Align with model features
    df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predict
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    if prediction == 1:
        st.error(f"Customer is likely to churn (Probability: {probability:.2f})")
    else:
        st.success(f"Customer is not likely to churn (Probability: {probability:.2f})")
