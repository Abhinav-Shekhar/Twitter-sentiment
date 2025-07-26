import streamlit as st
import pickle

# Load model and vectorizer
with open("sentiment_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# Streamlit app
st.set_page_config(page_title="Financial Sentiment Classifier", page_icon="📊")

st.title("📈 Financial News Sentiment Classifier")
st.write("Enter a financial news headline or tweet to classify its sentiment.")

# User input
user_input = st.text_area("📝 Enter News or Tweet", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        X = vectorizer.transform([user_input])
        pred = model.predict(X)[0]
        st.success(f"🔍 Predicted Sentiment Label: **{pred}**")