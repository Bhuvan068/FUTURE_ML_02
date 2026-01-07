import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Spotify Churn Dashboard", layout="centered")
st.title("🎧 Spotify Customer Churn Prediction Dashboard")

# Load model and features
try:
    model = joblib.load("spotify_churn_model.pkl")
    feature_names = joblib.load("spotify_feature_names.pkl")
    st.success("Model Loaded Successfully ✅")
except Exception as e:
    st.error(f"Error loading model files: {e}")

# Input form
st.header("🔮 Predict Customer Churn")
with st.form("churn_form"):
    Age = st.selectbox("Age Group", ["20-35", "12-20", "35-60", "Others"])
    Gender = st.selectbox("Gender", ["Male", "Female", "Others"])
    spotify_usage_period = st.selectbox("Usage Period", ["Less than 6 months", "6 months to 1 year", "1 year to 2 years", "More than 2 years"])
    spotify_listening_device = st.selectbox("Listening Device", ["Smartphone", "Computer or laptop", "Smart speakers or voice assistants"])
    spotify_subscription_plan = st.selectbox("Subscription Plan", ["Free (ad-supported)", "Premium (paid subscription)"])
    music_lis_frequency = st.selectbox("Music Listening Frequency", ["Daily", "Several times a week", "Once a week", "Rarely"])
    music_recc_rating = st.slider("Music Recommendation Rating", 1, 5, 3)

    submit = st.form_submit_button("Predict Churn")

if submit:
    # Create DataFrame from input
    input_df = pd.DataFrame([{
        "Age": Age, "Gender": Gender, "spotify_usage_period": spotify_usage_period,
        "spotify_listening_device": spotify_listening_device, "spotify_subscription_plan": spotify_subscription_plan,
        "music_lis_frequency": music_lis_frequency, "music_recc_rating": music_recc_rating
    }])

    # One-hot encode input
    input_encoded = pd.get_dummies(input_df)
    # Ensure same columns as training
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_names]

    # Predict
    churn_prob = model.predict_proba(input_encoded)[0][1]
    if churn_prob < 0.33: risk = "🟢 Low Risk"
    elif churn_prob < 0.66: risk = "🟡 Medium Risk"
    else: risk = "🔴 High Risk"

    st.subheader("📊 Prediction Result")
    st.metric("Churn Probability", f"{churn_prob:.2%}")
    st.write("Risk Level:", risk)
