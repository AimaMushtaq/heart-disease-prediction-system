"""
Heart Disease Prediction System - Streamlit Web App
----------------------------------------------------
Loads the Support Vector Classifier trained in
Heart_Disease_Prediction.ipynb (svc_trained_model.pkl) and predicts
whether heart disease is present from 4 input attributes:
Sex, Chest Pain type, Exercise induced angina, and Oldpeak.

All 4 inputs are already numeric, so no Label Encoding is used.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# ---------------------------------------------------------------------------
# Load the trained model (same file the notebook produced)
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()

# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------
st.title("❤️ Heart Disease Prediction System")
st.markdown(
    """
    This app uses a trained **Support Vector Classifier (SVC)** to predict whether
    heart disease is present, using only **4 input attributes** from the UCI
    Cleveland Heart Disease dataset.

    *Model accuracy on held-out test data: **84%** (303-patient dataset).*
    """
)

st.divider()

# Option values are the same numbers used in the dataset (no encoding needed);
# format_func only changes what the user sees on screen.
col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox(
        "Sex",
        options=[1, 0],
        format_func=lambda v: {1: "Male", 0: "Female"}[v],
    )
    chest_pain = st.selectbox(
        "Chest Pain Type",
        options=[1, 2, 3, 4],
        index=3,
        format_func=lambda v: {
            1: "1 - Typical angina",
            2: "2 - Atypical angina",
            3: "3 - Non-anginal pain",
            4: "4 - Asymptomatic",
        }[v],
    )
with col2:
    exercise_angina = st.selectbox(
        "Exercise Induced Angina",
        options=[0, 1],
        format_func=lambda v: {0: "No", 1: "Yes"}[v],
    )
    oldpeak = st.number_input(
        "Oldpeak (ST depression)", min_value=0.0, max_value=7.0, value=1.0, step=0.1
    )

if st.button("Predict Heart Disease", type="primary", use_container_width=True):
    features = pd.DataFrame({
        "Sex": [sex],
        "ChestPain": [chest_pain],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
    })

    prediction = model.predict(features)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE LIKELY")
    else:
        st.success("### ✅ NO HEART DISEASE LIKELY")

    with st.expander("See the feature vector sent to the model"):
        st.dataframe(features, hide_index=True)

st.divider()
st.caption(
    "Built for learning purposes using only 4 attributes. This is NOT a medical "
    "tool - do not use it for real diagnosis. Please consult a doctor."
)
