import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NextWord Prediction Engine",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0b0f19, #111827);
    color: #e5e7eb;
}

.main {
    padding: 2.5rem 4rem;
}

/* Title */
.title {
    font-size: 3.4rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.3rem;
}

.subtitle {
    font-size: 1.2rem;
    color: #9ca3af;
    margin-bottom: 0.3rem;
}

.author {
    font-size: 1rem;
    color: #60a5fa;
    font-weight: 600;
    margin-bottom: 2rem;
}

/* Cards */
.card {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 18px;
    padding: 28px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Labels */
.label {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 10px;
}

/* Prediction Text */
.prediction {
    font-size: 1.8rem;
    font-weight: 700;
    color: #22c55e;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    border: none;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
}

/* Footer */
.footer {
    margin-top: 25px;
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
<div class="title">NextWord Prediction Engine</div>
<div class="subtitle">LSTM & RNN Based Language Modeling System</div>
<div class="author">Built by <b>Gaurav Singh</b></div>
""", unsafe_allow_html=True)

# ---------------- MAIN LAYOUT ----------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label">Enter Input Text</div>', unsafe_allow_html=True)

    user_input = st.text_area(
        "",
        placeholder="Type a sentence and let the model predict the next word...",
        height=140
    )

    predict_btn = st.button("Predict Next Word")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label">Prediction Output</div>', unsafe_allow_html=True)

    if predict_btn and user_input.strip():
        seq = tokenizer.texts_to_sequences([user_input])[0]
        seq = seq[-max_len:]
        seq = np.pad(seq, (max_len - len(seq), 0))
        seq = np.expand_dims(seq, axis=0)

        pred = model.predict(seq)
        word = tokenizer.index_word.get(np.argmax(pred), "")

        st.markdown(f"<div class='prediction'>{word}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='prediction'>Waiting for input...</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
LSTM • RNN • NLP • Deep Learning • TensorFlow
</div>
""", unsafe_allow_html=True)
