import streamlit as st
import sklearn
import helper
import pickle
import nltk

# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Load pre-trained model and vectorizer
model = pickle.load(open("models/model.pkl", 'rb'))
vectorizer = pickle.load(open("models/vectorizer.pkl", 'rb'))

# Streamlit UI
st.title("Sentiment Analysis App")
st.write("Analyze the sentiment of a given review as **Positive** or **Negative**.")

# Text input for the review
text = st.text_input("Please enter your review:")

if st.button("Predict"):
    if text.strip():  # Ensure text is not empty
        # Preprocess the text
        token = helper.preprocessing_step(text)
        
        # Vectorize the preprocessed text
        vectorized_data = vectorizer.transform([token])
        
        # Predict sentiment
        prediction = model.predict(vectorized_data)[0]  # Get the first prediction
        
        # Convert prediction to label
        sentiment = "Positive" if prediction == 1 else "Negative"
        
        # Display the result
        st.success(f"The sentiment of the review is: **{sentiment}**")
    else:
        st.error("Please enter a valid review to analyze.")
