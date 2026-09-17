
import streamlit as st
import joblib
import string
import nltk
from pathlib import Path

# ---------------------------------------------------------
# Er. Atul Kumar | NLP Sentiment Analysis
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Sentiment Analyzer | Er. Atul Kumar",
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "logistic_regression_model.joblib"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.joblib"

# Download English stopwords when they are not available
try:
    STOP_WORDS = set(nltk.corpus.stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(nltk.corpus.stopwords.words("english"))


@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def preprocess_text(text: str) -> str:
    """Match the preprocessing used in the training notebook."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    return " ".join(words)


def predict_emotion(text: str):
    model, vectorizer = load_assets()
    cleaned = preprocess_text(text)
    vector = vectorizer.transform([cleaned])
    prediction_id = int(model.predict(vector)[0])

    # The training notebook creates IDs from the first occurrence
    # order in the emotion column. For this dataset the mapping is:
    # 0 sadness, 1 anger, 2 love, 3 surprise, 4 fear, 5 joy.
    id_to_emotion = {
        0: "sadness",
        1: "anger",
        2: "love",
        3: "surprise",
        4: "fear",
        5: "joy",
    }

    emotion = id_to_emotion.get(prediction_id, "unknown")
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(max(model.predict_proba(vector)[0])) * 100

    return emotion, confidence, cleaned


# -------------------- Custom CSS --------------------
st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }

    .hero {
        padding: 2rem 1.5rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #111827 0%, #312e81 55%, #4f46e5 100%);
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 35px rgba(0,0,0,.16);
    }

    .hero h1 {
        margin-bottom: .35rem;
        font-size: 2.35rem;
    }

    .hero p {
        margin: .25rem 0;
        opacity: .92;
    }

    .developer {
        margin-top: 1rem;
        font-size: .95rem;
        font-weight: 600;
        opacity: .95;
    }

    .result-card {
        padding: 1.4rem;
        border-radius: 18px;
        border: 1px solid rgba(99,102,241,.25);
        background: rgba(99,102,241,.08);
        text-align: center;
        margin-top: 1rem;
    }

    .emotion {
        font-size: 2rem;
        font-weight: 800;
        text-transform: capitalize;
    }

    .small-note {
        color: #6b7280;
        font-size: .9rem;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Header --------------------
st.markdown(
    """
    <div class="hero">
        <h1>😊 AI Sentiment Analyzer</h1>
        <p>Understand the emotion expressed in your text using NLP.</p>
        <div class="developer">Developed & Deployed by Er. Atul Kumar</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("### Enter your text")
user_input = st.text_area(
    "Text",
    height=150,
    placeholder="Example: I am very happy today because I completed my project!",
    label_visibility="collapsed",
)

col1, col2 = st.columns([2, 1])
with col1:
    predict_clicked = st.button("🔍 Predict Emotion", use_container_width=True)
with col2:
    clear_clicked = st.button("🗑️ Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

if predict_clicked:
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            emotion, confidence, cleaned = predict_emotion(user_input)

            emoji_map = {
                "joy": "😊",
                "sadness": "😢",
                "anger": "😠",
                "fear": "😨",
                "love": "❤️",
                "surprise": "😮",
            }
            icon = emoji_map.get(emotion, "🤖")

            st.markdown(
                f"""
                <div class="result-card">
                    <div style="font-size:3rem">{icon}</div>
                    <div class="emotion">{emotion}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if confidence is not None:
                st.progress(min(confidence / 100, 1.0))
                st.metric("Model confidence", f"{confidence:.2f}%")

            with st.expander("View processed text"):
                st.write(cleaned)

        except FileNotFoundError as e:
            st.error(
                f"Required model file not found: {Path(e.filename).name}. "
                "Keep both .joblib files in the same folder as app.py."
            )
        except Exception as e:
            st.error(f"Prediction error: {e}")

# -------------------- About --------------------
with st.expander("ℹ️ About this project"):
    st.write(
        """
        This Streamlit frontend uses the trained Logistic Regression model
        and TF-IDF vectorizer from the supplied NLP project.

        The model predicts one of six emotions:
        sadness, anger, love, surprise, fear, and joy.
        """
    )

st.markdown("---")
st.caption("© 2026 Er. Atul Kumar | NLP Sentiment Analysis Project")
