
import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# UI
st.title("📩 Spam Detection App")
st.write("Enter a message and we'll predict whether it's Spam or Not Spam.")

# Input box
message = st.text_area("Your Message", "")

# Prediction
if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        data = vectorizer.transform([message])
        prediction = model.predict(data)[0]
        if prediction == 1:
            st.error("🚨 This is SPAM!")
        else:
            st.success("✅ This is NOT spam.")
