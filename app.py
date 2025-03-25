import streamlit as st
import joblib

vectorizer = joblib.load("tfidf_vectorizer.pkl")
nb_model = joblib.load("naive_bayes.pkl")
log_reg_model = joblib.load("logistic_regression.pkl") 

st.title("Email Spam Classifier 🚀")

email_text = st.text_area("Enter the email content:")

model_choice = st.selectbox("Choose a model:", ["Naïve Bayes", "Logistic Regression"])

if st.button("Classify Email"):
    if email_text:
        email_tfidf = vectorizer.transform([email_text])
        if model_choice == "Naïve Bayes":
            model = nb_model
        else:
            model = log_reg_model
        prediction = model.predict(email_tfidf)[0]

        if prediction == 0:
            result = "✅ Not Spam"
        else:
            result = "🚨 Spam Email" 
        st.subheader(f"Prediction: {result}")
    else:
        st.warning("Please enter an email to classify.")
