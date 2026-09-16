😊 Sentiment Analysis

A Machine Learning-based Sentiment Analysis web application built with Streamlit. It uses TF-IDF Vectorization and a trained Logistic Regression model to analyze user-provided text and predict emotions.

🚀 Features

Interactive Streamlit web interface

Text preprocessing

TF-IDF text vectorization

Logistic Regression prediction

Emotion prediction from user input

Confidence score using model probabilities

🧠 Model

The project uses:

TF-IDF Vectorizer to convert text into numerical features.

Logistic Regression to predict the emotion category.

The trained model and vectorizer are stored as .joblib files.

📁 Project Structure

sentiment_project/
├── app.py
├── logistic_regression_model(2).joblib
├── tfidf_vectorizer.joblib
└── README.md

🛠️ Technologies Used

Python

Streamlit

Scikit-learn

NLTK

Joblib

TF-IDF

Logistic Regression

⚙️ Installation

Install the required packages:

pip install streamlit scikit-learn joblib nltk

▶️ Run the Application

streamlit run app.py

The application will open in your browser.

✍️ How to Use

Enter a sentence in the text area.

Click Predict Sentiment.

The model processes the text and displays the predicted emotion and confidence score.

📌 Emotion Classes

The trained model contains 6 output classes. The exact class-to-emotion names should match the emotion_numbers mapping used during model training.

