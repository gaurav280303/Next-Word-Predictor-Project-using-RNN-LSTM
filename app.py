import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NextWord | LSTM Language Model",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: #e5e7eb;
}

.main {
    padding: 2.5rem;
}

.title {
    font-size: 3rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.2rem;
}

.subtitle {
    font-size: 1.1rem;
    color: #9ca3af;
    margin-bottom: 2rem;
}

.card {
    background: rgba(255,255,255,0.05);
    border-radius: 18px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.08);
}

.label {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 10px;
}

.result {
    font-size: 1.6rem;
    font-weight: 700;
    color: #22c55e;
}

.footer {
    margin-top: 20px;
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
<div class="subtitle">LSTM & RNN Based Language Prediction System</div>
<div class="subtitle">Built by <b>Gaurav Singh</b></div>
""", unsafe_allow_html=True)

# ---------------- MAIN GRID ----------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label">Enter Text</div>', unsafe_allow_html=True)

    user_input = st.text_area(
        "",
        placeholder="Type a sentence and let the model predict the next word...",
        height=140
    )

    predict = st.button("Predict Next Word")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label">Prediction</div>', unsafe_allow_html=True)

    if predict and user_input.strip():
        seq = tokenizer.texts_to_sequences([user_input])[0]
        seq = seq[-max_len:]
        seq = np.pad(seq, (max_len - len(seq), 0))
        seq = np.expand_dims(seq, axis=0)

        pred = model.predict(seq)
        word = tokenizer.index_word.get(np.argmax(pred), "")

        st.markdown(f"<div class='result'>{word}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result'>Waiting for input...</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
LSTM-based Sequence Modeling • TensorFlow • NLP • Deep Learning
</div>
""", unsafe_allow_html=True)
