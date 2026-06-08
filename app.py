import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("tuned_logistic_regression_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# App Title
st.title("🎬 IMDb Movie Review Sentiment Analysis")

st.write("Enter a movie review and predict whether it is Positive or Negative.")

# User Input
review = st.text_area("Movie Review")

# Prediction
if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        review_tfidf = vectorizer.transform([review])

        prediction = model.predict(review_tfidf)

        if prediction[0] == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")