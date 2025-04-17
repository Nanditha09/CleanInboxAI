import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

# Load trained model and vectorizer
model = joblib.load('xgb_spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'\d{10,}', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Streamlit UI
st.title("📩 Spam Detection App")
st.write("Enter a message to classify it as Spam or Not Spam.")

# Input box
message = st.text_area("Your Message", "")

# Prediction
if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        processed_message = preprocess_text(message)
        data_tfidf = vectorizer.transform([processed_message])

        # Detect presence of URLs, email addresses, and phone numbers
        has_url = int(bool(re.search(r'http[s]?://|www\.', message)))
        has_email = int(bool(re.search(r'\S+@\S+', message)))
        has_phone = int(bool(re.search(r'\d{10,}', message)))

        # Combine TF-IDF features with the new binary features
        additional_features = np.array([[has_url, has_email, has_phone]])
        data_combined = hstack([data_tfidf, additional_features])

        prediction = model.predict(data_combined)[0]
        if prediction == 1:
            st.error("🚨 This is SPAM!")
        else:
            st.success("✅ This is NOT spam.")
