import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

with open("model/model.pkl", "rb") as model_file:
    clf = pickle.load(model_file)

with open("model/vectorizer.pkl", "rb") as vectorizer_file:
    cv = pickle.load(vectorizer_file)

st.title("Twitter Sentiment Classification")
st.subheader("Classify tweets as Hate Speech, Offensive Language, or Neutral Language")

user_input = st.text_area("Enter a tweet", "")

if st.button("Classify"):
    if user_input:
        transformed_input = cv.transform([user_input]).toarray()
        prediction = clf.predict(transformed_input)[0]
        st.write(f"Prediction: **{prediction}**")
    else:
        st.write("Please enter a tweet to classify.")

st.sidebar.title("About")
st.sidebar.info("This app classifies tweets into three categories: Hate Speech, Offensive Language, or Neutral Language. Built using a Decision Tree Classifier and Streamlit.")
