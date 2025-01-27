import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load model and vectorizer
with open("model/model.pkl", "rb") as model_file:
    clf = pickle.load(model_file)

with open("model/vectorizer.pkl", "rb") as vectorizer_file:
    cv = pickle.load(vectorizer_file)

# App title and styling
st.set_page_config(page_title="Twitter Sentiment Classifier", page_icon=":bird:", layout="centered")
st.markdown(
    """
    <style>
    .main { 
        background-color: #f9f9f9; 
    }
    .stTextArea textarea {
        font-size: 1.1rem;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 1rem;
        border-radius: 8px;
        padding: 10px 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    /* Set the background color for the entire page */
    .stApp {
        background-color: #e6f7ff; /* Light blue color */
        padding: 20px; /* Optional: Add padding for aesthetics */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Main content
st.title("🐦 Twitter Sentiment Classification")

st.subheader("Classify tweets as Hate Speech, Offensive Language, or Neutral Language")

# Input section
# st.markdown("### Enter the Tweet Below")
user_input = st.text_area("Enter your tweet", placeholder="Type your tweet here...", height=150)

# Classification button
if st.button("🚀 Classify"):
    if user_input.strip():
        transformed_input = cv.transform([user_input]).toarray()
        prediction = clf.predict(transformed_input)[0]
        
        # Display result with styling
        st.markdown(
            f"""
            ### Prediction: 
            **:speech_balloon: {prediction}**
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("⚠️ Please enter a tweet to classify.")

# Sidebar
st.sidebar.title("About the App")
st.sidebar.info(
    """
    - **Purpose**: This app classifies tweets into three categories:
      1. Hate Speech
      2. Offensive Language
      3. Neutral Language
    - **Technology Used**: 
      - Decision Tree Classifier
      - Streamlit Framework
    """
)

# Footer
# st.markdown(
#     """
#     <hr>
#     <footer style="text-align: center;">
#         Built with ❤️ using Streamlit
#     </footer>
#     """,
#     unsafe_allow_html=True,
# )
