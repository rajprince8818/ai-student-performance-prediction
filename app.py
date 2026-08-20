from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/student_performance_model.joblib")

st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 AI-Driven Student Performance Prediction System")
st.write(
    "Enter student learning indicators to estimate the expected final academic score."
)

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python generate_data.py` and `python train_model.py` first.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

with st.sidebar:
    st.header("Student Inputs")

    attendance = st.slider("Attendance (%)", 0, 100, 80)
    study_hours = st.slider("Study hours/week", 0.0, 40.0, 10.0, 0.5)
    previous_score = st.slider("Previous exam score", 0.0, 100.0, 70.0, 1.0)
    assignment_avg = st.slider("Assignment average", 0.0, 100.0, 75.0, 1.0)
    participation = st.slider("Class participation (%)", 0, 100, 70)
    sleep_hours = st.slider("Average sleep hours", 3.0, 12.0, 7.0, 0.5)
    previous_failures = st.number_input(
        "Previous failures", min_value=0, max_value=10, value=0, step=1
    )
    internet_access = st.selectbox("Reliable internet access", ["Yes", "No"])
    extracurricular = st.selectbox("Extracurricular activity", ["Yes", "No"])

input_row = pd.DataFrame([{
    "attendance": attendance,
    "study_hours": study_hours,
    "previous_score": previous_score,
    "assignment_avg": assignment_avg,
    "participation": participation,
    "sleep_hours": sleep_hours,
    "previous_failures": previous_failures,
    "internet_access": 1 if internet_access == "Yes" else 0,
    "extracurricular": 1 if extracurricular == "Yes" else 0,
}])[features]

prediction = float(model.predict(input_row)[0])
prediction = max(0, min(100, prediction))

if prediction >= 85:
    band = "Excellent"
elif prediction >= 70:
    band = "Good"
elif prediction >= 50:
    band = "Average"
else:
    band = "At Risk"

col1, col2 = st.columns(2)
with col1:
    st.metric("Predicted Final Score", f"{prediction:.1f}/100")
with col2:
    st.metric("Performance Band", band)

st.subheader("Prediction Interpretation")

if band == "At Risk":
    st.warning(
        "The model estimates that this student may need additional academic support."
    )
elif band == "Average":
    st.info(
        "The student is in the average range. Consistent study and assignment performance may help."
    )
elif band == "Good":
    st.success(
        "The student is predicted to perform well. Maintaining current habits is recommended."
    )
else:
    st.success(
        "The student is predicted to perform excellently based on the entered indicators."
    )

st.caption(
    "This is an educational prediction tool using synthetic data. "
    "It should not be used as the sole basis for decisions about real students."
)
