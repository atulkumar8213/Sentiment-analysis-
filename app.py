import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords


# =========================================================
# Page Configuration
# =========================================================
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="centered"
)


# =========================================================
# Load NLTK Stopwords
# =========================================================
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))


# =========================================================
# Load Model and TF-IDF Vectorizer
# =========================================================
@st.cache_resource
def load_model():
    model = joblib.load("logistic_regression_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")

    return model, vectorizer


model, vectorizer = load_model()


# =========================================================
# Text Preprocessing
# =========================================================
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove stopwords
    words = text.split()
    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# =========================================================
# Emotion Labels
# =========================================================
emotion_labels = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}


# =========================================================
# Custom CSS
# =========================================================
st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        border: 1px solid #ddd;
    }

    .emotion {
        font-size: 32px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Header
# =========================================================
st.markdown(
    '<div class="main-title">😊 Emotion Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enter a sentence and let the Machine Learning model '
    'detect the emotion.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# Text Input
# =========================================================
user_input = st.text_area(
    "Enter your text",
    height=150,
    placeholder=(
        "Example: I am very happy today because "
        "I got a new job!"
    )
)


# =========================================================
# Prediction
# =========================================================
if st.button(
    "🔍 Predict Emotion",
    use_container_width=True
):

    if not user_input.strip():

        st.warning("⚠️ Please enter some text first.")

    else:

        # Preprocess
        processed_text = preprocess_text(user_input)

        # TF-IDF transformation
        text_vector = vectorizer.transform(
            [processed_text]
        )

        # Model prediction
        prediction = model.predict(text_vector)[0]

        # Convert numerical label to emotion
        emotion = emotion_labels.get(
            int(prediction),
            str(prediction)
        )

        # Probability if available
        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(
                text_vector
            )[0]

            probability = max(probabilities) * 100


        # =================================================
        # Result
        # =================================================
        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="emotion">{emotion}</div>',
            unsafe_allow_html=True
        )

        if probability is not None:
            st.write(
                f"Confidence: **{probability:.2f}%**"
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:

    st.header("📌 About")

    st.write(
        """
        This application uses:

        **Machine Learning Model**
        - Logistic Regression

        **Text Representation**
        - TF-IDF Vectorizer

        **Preprocessing**
        - Lowercase conversion
        - Punctuation removal
        - Stopword removal

        **Emotion Classes**
        - Sadness
        - Joy
        - Love
        - Anger
        - Fear
        - Surprise
        """
    )

    st.divider()

    st.caption(
        "NLP Emotion Detection Project"
    )

