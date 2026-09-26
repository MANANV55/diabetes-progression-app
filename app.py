import streamlit as st
import joblib
import numpy as np

# Load saved artefacts
reg_model  = joblib.load("diabetes_reg_model.joblib")
clf_model  = joblib.load("diabetes_clf_model.joblib")
scaler     = joblib.load("diabetes_scaler.joblib")
median_val = joblib.load("diabetes_median.joblib")

st.set_page_config(page_title="Diabetes Progression Predictor", page_icon="🩺")
st.title("🩺 Diabetes Progression & Risk Predictor")
st.write(
    "Enter the patient's measurements below. The model will estimate "
    "their diabetes progression score and classify their risk level."
)
st.caption("For educational use only — not a clinical tool.")

# Feature inputs — sliders tuned to realistic ranges
col1, col2 = st.columns(2)

with col1:
    age              = st.slider("Age (scaled)",           -0.10, 0.10, 0.00, 0.001)
    sex              = st.slider("Sex (scaled)",           -0.05, 0.05, 0.00, 0.001)
    bmi              = st.slider("BMI (scaled)",           -0.10, 0.10, 0.00, 0.001)
    blood_pressure   = st.slider("Blood Pressure (scaled)",-0.10, 0.10, 0.00, 0.001)
    total_cholesterol= st.slider("Total Cholesterol",      -0.10, 0.10, 0.00, 0.001)

with col2:
    ldl              = st.slider("LDL (scaled)",           -0.10, 0.10, 0.00, 0.001)
    hdl              = st.slider("HDL (scaled)",           -0.10, 0.10, 0.00, 0.001)
    chol_hdl_ratio   = st.slider("Chol/HDL Ratio",         -0.10, 0.10, 0.00, 0.001)
    triglycerides    = st.slider("Triglycerides (scaled)", -0.10, 0.10, 0.00, 0.001)
    blood_sugar      = st.slider("Blood Sugar (scaled)",   -0.10, 0.10, 0.00, 0.001)

if st.button("Predict", type="primary"):
    input_data = np.array([[
        age, sex, bmi, blood_pressure, total_cholesterol,
        ldl, hdl, chol_hdl_ratio, triglycerides, blood_sugar
    ]])

    # Regression: numeric progression score
    score = reg_model.predict(input_data)[0]

    # Classification: risk label + confidence
    risk = clf_model.predict(input_data)[0]
    proba = clf_model.predict_proba(input_data)[0]

    st.markdown("---")
    st.subheader("Results")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Predicted Progression Score", f"{score:.1f}")
        st.caption(f"(Dataset median: {median_val:.1f})")
    with col_b:
        if risk == 1:
            st.error("Risk Level: **HIGH**")
        else:
            st.success("Risk Level: **LOW**")

    st.write("**Confidence:**")
    st.progress(float(proba[1]),
                text=f"High Risk: {proba[1]*100:.1f}%")
    st.progress(float(proba[0]),
                text=f"Low Risk:  {proba[0]*100:.1f}%")

st.caption("Model: Linear Regression + Random Forest · Dataset: sklearn load_diabetes()")
