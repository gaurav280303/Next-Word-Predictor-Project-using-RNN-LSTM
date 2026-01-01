import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NextWord  Predictor (LSTM & RNN)",
    page_icon="🧠",
    layout="centered"
)

# ---------------- STYLING ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(180deg, #0a0f1c, #0f172a);
    color: #e5e7eb;
}

.main {
    padding: 3rem 2rem;
    max-width: 900px;
    margin: auto;
}

/* Headings */
.title {
    font-size: 3.6rem;
    font-weight: 800;
    letter-spacing: -1px;
    color: #ffffff;
    margin-bottom: 0.3rem;
}

.subtitle {
    font-size: 1.2rem;
    color: #9ca3af;
    margin-bottom: 0.2rem;
}

.author {
    font-size: 1rem;
    color: #60a5fa;
    margin-bottom: 3rem;
}

/* Input */
textarea {
    font-size: 1.1rem !important;
    background: #111827 !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #374151 !important;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    border: none;
    padding: 0.8rem;
    font-size: 1.1rem;
    font-weight: 600;
    border-radius: 12px;
    margin-top: 1rem;
}

/* Prediction */
.prediction-box {
    margin-top: 2.5rem;
    padding: 1.8rem;
    border-radius: 14px;
    background: rgba(255,255,255,0.06);
    text-align: center;
}

.prediction-text {
    font-size: 2rem;
    font-weight: 700;
    color: #22c55e;
}

/* Footer */
.footer {
    margin-top: 3rem;
    text-align: center;
    color: #9ca3af;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_assets():
    model = load_model("lstm_model (1).h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len

model, tokenizer, max_len = load_assets()

# ---------------- HEADER ----------------
st.markdown("""
<div class="title">NextWord</div>
<div class="subtitle">LSTM & RNN Word Prediction System</div>
<div class="author">Built by <b>GAURAV SINGH</b></div>
""", unsafe_allow_html=True)

# ---------------- INPUT ----------------
text = st.text_area(
    "",
    placeholder="Type a sentence and let the model predict the next word...",
    height=120
)

predict = st.button("Predict Next Word")

# ---------------- PREDICTION ----------------
if predict and text.strip():
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = seq[-max_len:]
    seq = np.pad(seq, (max_len - len(seq), 0))
    seq = np.expand_dims(seq, axis=0)

    pred = model.predict(seq)
    word = tokenizer.index_word.get(np.argmax(pred), "")

    st.markdown(f"""
    <div class="prediction-box">
        <div class="prediction-text">{word}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
LSTM • RNN • NLP • Deep Learning
</div>
""", unsafe_allow_html=True)

