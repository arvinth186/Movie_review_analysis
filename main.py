import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import streamlit as st

# load the IMDB dataset

word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

# Load the trained model with tanh activation function
model = load_model('simple_rnn_imdb_model.h5')

# Function to decode the review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

# Function to add the padding to the input review
def preprocess_review(review):
    words = review.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Predict function

def predict_sentiment(review):
    preprocessed_input = preprocess_review(review)
    prediction = model.predict(preprocessed_input)
    
    sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'

    return sentiment, prediction[0][0]


st.title("IMDB Movie Review Sentiment Analysis")

user_input = st.text_area("Enter your movie review here:")

if st.button("Predict Sentiment"):
    if user_input:
        sentiment, confidence = predict_sentiment(user_input)
        # create the success message for positve and fail for negative
        if sentiment == 'Positive':
            st.success(f"The review is predicted to be: {sentiment}  (Confidence: {confidence:.2f})")
        else:
            st.error(f"The review is predicted to be: {sentiment} (Confidence: {confidence:.2f})")
    else:
        st.write("Please enter a movie review to analyze.")