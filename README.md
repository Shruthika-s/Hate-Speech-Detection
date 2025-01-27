# Twitter Sentiment Classification Web App

This project is a *Streamlit-based web application* for classifying tweets into three categories:

- *Hate Speech*
- *Offensive Language*
- *Neutral Language*

The app leverages a trained *Decision Tree Classifier* and a *CountVectorizer* for natural language processing and sentiment classification. The backend is implemented in Python, and the frontend is built using Streamlit.

---

## Features

- *Real-Time Predictions*: Classify any input tweet into one of the three categories.
- *Interactive UI*: Simple and user-friendly interface using Streamlit.
- *Pre-Trained Model*: A pre-trained model and vectorizer are used to ensure quick and accurate predictions.

---

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip

### Clone the Repository
bash
$ git clone https://github.com/yourusername/twitter-classification-streamlit.git
$ cd twitter-classification-streamlit


### Install Dependencies
bash
$ pip install -r requirements.txt


---

## Usage

### 1. Train the Model (Optional)
If you want to train the model from scratch:
bash
$ python train_model.py

This will generate model.pkl and vectorizer.pkl in the model/ directory.

### 2. Run the Application
Launch the Streamlit app:
bash
$ streamlit run app.py


### 3. Classify Tweets
- Open the app in your browser (default: http://localhost:8501).
- Enter a tweet in the text area and click *Classify* to see the prediction.

---

## Technologies Used

- *Python 3.8+*
- *Streamlit*: Web framework for building interactive UIs
- *scikit-learn*: Machine learning library for training the model
- *nltk*: Natural Language Toolkit for text preprocessing
- *pandas*: Data manipulation library

---

## Future Enhancements

- Add support for more advanced models like Random Forest or Deep Learning.
- Incorporate additional NLP preprocessing techniques.
- Visualize tweet sentiment distribution with charts and graphs.
- Integrate a database to log user inputs and predictions.

---

## Acknowledgments

- Dataset sourced from [Twitter Sentiment Dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis).
- Inspired by real-world applications of sentiment analysis.
