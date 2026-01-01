import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NextWord Prediction | LSTM & RNN",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #f9fafb, #eef2f7);
}

.main {
    padding: 2rem 4rem;
}

.title {
    font-size: 3rem;
    font-weight: 700;
    color: #111827;
}

.subtitle {
    font-size: 1.2rem;
    color: #4b5563;
    margin-bottom: 20px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.05);
}

.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 10px;
}

.footer {
    margin-top: 30px;
    text-align: center;
    font-size: 0.9rem;
    color: #6b7280;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_and_assets():
    model = load_model("lstm_model (1).h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len

model, tokenizer, max_len = load_model_and_assets()

# ---------------- HEADER ----------------
st.markdown("""
<div class="title">NextWord Prediction</div>
<div class="subtitle">
LSTM & RNN Based Language Modeling System
</div>
""", unsafe_allow_html=True)

# ---------------- MAIN LAYOUT ----------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🔤 Enter Text</div>", unsafe_allow_html=True)

    user_input = st.text_area(
        "Type a sentence and let the model predict the next word:",
        height=120,
        placeholder="Deep learning models are capable of..."
    )

    predict_btn = st.button("Predict Next Word")

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🧠 Prediction</div>", unsafe_allow_html=True)

    if predict_btn and user_input.strip():
        seq = tokenizer.texts_to_sequences([user_input])[0]
        seq = seq[-max_len:]
        seq = np.pad(seq, (max_len - len(seq), 0))
        seq = np.expand_dims(seq, axis=0)

        pred = model.predict(seq)
        word = tokenizer.index_word.get(np.argmax(pred), "")

        st.success(f"**{word}**")
    else:
        st.info("Waiting for input...")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- BOTTOM INFO ----------------
st.markdown("""
<div class="card">
<b>Model Overview</b><br>
• Architecture: LSTM-based Recurrent Neural Network  
• Framework: TensorFlow / Keras  
• Task: Next-word prediction using sequence learning  
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
Built by <b>Gaurav Singh</b> • AI & Data Science Enthusiast  
</div>
""", unsafe_allow_html=True)
