import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Spotify Churn Prediction Dashboard",
    layout="centered"
)

# Title
st.title("🎧 Spotify Customer Churn Prediction Dashboard")

# =========================
# Load saved models
# =========================
@st.cache_resource
def load_models():
    model = joblib.load("spotify_churn_model.pkl")
    preprocessor = joblib.load("spotify_preprocessor.pkl")
    feature_names = joblib.load("spotify_feature_names.pkl")
    return model, preprocessor, feature_names

model, preprocessor, feature_names = load_models()
st.success("Model & Preprocessor Loaded Successfully ✅")

# =========================
# 1️⃣ CUSTOMER INPUT FORM
# =========================
st.header("🔮 Predict Customer Churn")

with st.form("churn_form"):
    Age = st.selectbox("Age Group", ["20-35", "12-20", "35-60", "Others"])
    Gender = st.selectbox("Gender", ["Male", "Female", "Others"])
    spotify_usage_period = st.selectbox(
        "Usage Period",
        ["Less than 6 months", "6 months to 1 year", "1 year to 2 years", "More than 2 years"]
    )
    spotify_listening_device = st.selectbox(
        "Listening Device",
        ["Smartphone", "Computer or laptop", "Smart speakers or voice assistants"]
    )
    spotify_subscription_plan = st.selectbox(
        "Subscription Plan",
        ["Free (ad-supported)", "Premium (paid subscription)"]
    )
    music_lis_frequency = st.selectbox(
        "Music Listening Frequency",
        ["Daily", "Several times a week", "Once a week", "Rarely"]
    )
    music_recc_rating = st.slider("Music Recommendation Rating", 1, 5, 3)

    submit = st.form_submit_button("Predict Churn")

if submit:
    input_data = pd.DataFrame([{
        "Age": Age,
        "Gender": Gender,
        "spotify_usage_period": spotify_usage_period,
        "spotify_listening_device": spotify_listening_device,
        "spotify_subscription_plan": spotify_subscription_plan,
        "music_lis_frequency": music_lis_frequency,
        "music_recc_rating": music_recc_rating
    }])

    # Fill missing columns
    for col in preprocessor.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = "Unknown"

    X = preprocessor.transform(input_data)
    churn_prob = model.predict_proba(X)[0][1]

    if churn_prob < 0.33:
        risk = "🟢 Low Risk"
    elif churn_prob < 0.66:
        risk = "🟡 Medium Risk"
    else:
        risk = "🔴 High Risk"

    st.subheader("📊 Prediction Result")
    st.metric("Churn Probability", f"{churn_prob:.2%}")
    st.write("Risk Level:", risk)

# =========================
# 2️⃣ FEATURE IMPORTANCE
# =========================
st.header("📌 Top Churn Drivers")

importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:10]

top_features = [feature_names[i] for i in indices]
top_importances = importances[indices]

fig1, ax1 = plt.subplots()
ax1.barh(top_features[::-1], top_importances[::-1])
ax1.set_xlabel("Importance Score")
ax1.set_title("Top 10 Features Influencing Customer Churn")
st.pyplot(fig1)

# =========================
# 3️⃣ MODEL METRICS
# =========================
st.header("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", "0.76")
col2.metric("Precision", "0.86")
col3.metric("Recall", "0.75")
col4.metric("F1-Score", "0.80")

st.subheader("Confusion Matrix")

cm = np.array([[29, 8],
               [17, 50]])

fig2, ax2 = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
ax2.set_title("Confusion Matrix")
st.pyplot(fig2)
