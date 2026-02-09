import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student At-Risk Detector", layout="centered")

st.title("🎓 Early Student At-Risk Detection")
st.write("Fill in the student details below. The model will predict if the student is **at risk of failing**.")

# Load saved model + default row
model = joblib.load("artifacts/model.joblib")
default_row = joblib.load("artifacts/default_row.joblib")  # a 1-row DataFrame

# --- Form UI ---
with st.form("student_form"):
    st.subheader("Student Information")

    age = st.number_input("Age", min_value=10, max_value=25, value=17)

    subject = st.selectbox("Subject", ["Math", "Portuguese"])

    failures = st.selectbox("Past failures", [0, 1, 2, 3])
    absences = st.number_input("Absences", min_value=0, max_value=100, value=0)

    studytime = st.selectbox("Weekly study time (1=low, 4=high)", [1, 2, 3, 4])

    schoolsup = st.selectbox("Extra school support?", ["yes", "no"])
    famsup = st.selectbox("Family educational support?", ["yes", "no"])

    internet = st.selectbox("Internet access at home?", ["yes", "no"])
    higher = st.selectbox("Wants higher education?", ["yes", "no"])

    # Use your earlier grades (these are allowed because we dropped G3, not G1/G2)
    g1 = st.number_input("Grade 1 (G1)", min_value=0, max_value=20, value=10)
    g2 = st.number_input("Grade 2 (G2)", min_value=0, max_value=20, value=10)

    submitted = st.form_submit_button("Predict Risk")

# --- Prediction ---
if submitted:
    # Start from default row so all required columns exist
    row = default_row.copy()

    # Update only the fields we collected
    if "age" in row.columns: row.loc[row.index[0], "age"] = age
    if "subject" in row.columns: row.loc[row.index[0], "subject"] = subject
    if "failures" in row.columns: row.loc[row.index[0], "failures"] = failures
    if "absences" in row.columns: row.loc[row.index[0], "absences"] = absences
    if "studytime" in row.columns: row.loc[row.index[0], "studytime"] = studytime

    if "schoolsup" in row.columns: row.loc[row.index[0], "schoolsup"] = schoolsup
    if "famsup" in row.columns: row.loc[row.index[0], "famsup"] = famsup
    if "internet" in row.columns: row.loc[row.index[0], "internet"] = internet
    if "higher" in row.columns: row.loc[row.index[0], "higher"] = higher

    if "G1" in row.columns: row.loc[row.index[0], "G1"] = g1
    if "G2" in row.columns: row.loc[row.index[0], "G2"] = g2

    # Predict
    pred = model.predict(row)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row)[0][1]

    # Display results nicely
    if pred == 1:
        st.error("⚠️ Prediction: AT RISK")
        st.write("Recommendation: Provide early intervention (consultation, extra lessons, counselling).")
    else:
        st.success("✅ Prediction: NOT AT RISK")
        st.write("Recommendation: Continue monitoring and provide normal support.")

    if proba is not None:
        st.write(f"Confidence (probability of at-risk): **{proba:.2f}**")

    # Optional: show the final row used
    with st.expander("See the full input sent to the model"):
        st.dataframe(row)
