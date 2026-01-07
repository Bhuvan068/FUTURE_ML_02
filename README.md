Spotify Customer Churn Prediction System

Internship Task – Machine Learning
Organization: Future Interns
CIN: FIT/DEC25/ML4555
Task Number: 02

Project Overview
Customer churn is a critical business problem where users stop using a service. This project focuses on building an AI-powered Customer Churn Prediction System using Spotify user behavior data to identify customers who are likely to churn.
The system predicts churn probability and categorizes users into Low, Medium, and High Risk groups. An interactive Streamlit dashboard is developed to make predictions in real time.

Technologies Used

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Streamlit

Joblib

Methodology

Data cleaning and preprocessing

Feature engineering and encoding using One-Hot Encoding

Churn label creation based on subscription willingness

Model training using Logistic Regression and Random Forest Classifier

Model evaluation using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix

Feature importance analysis to identify churn drivers

Deployment using Streamlit for interactive predictions

Model Performance

Accuracy: 76%

Precision: 86%

Recall: 75%

F1-Score: 80%

Key Features

Predicts churn probability for individual users

Risk level classification (Low / Medium / High)

Feature importance visualization

Interactive web dashboard

How to Run the Project

Clone the repository

Install dependencies using requirements.txt

Run the application using:
streamlit run app.py

Project Files

app.py – Streamlit application

spotify_churn_model.pkl – Trained Random Forest model

spotify_preprocessor.pkl – Data preprocessing pipeline

spotify_feature_names.pkl – Encoded feature names

requirements.txt – Required libraries

Learning Outcomes

End-to-end ML pipeline development

Handling categorical data and class imbalance

Model deployment using Streamlit

Translating ML predictions into business insights

Acknowledgment
This project was completed as part of the Machine Learning Internship at Future Interns.
