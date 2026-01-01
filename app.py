import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NextWord Prediction | LSTM & RNN",
    page_icon="📘",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #f8f9fa;
}

.main {
    padding: 2rem;
}

h1, h2, h3 {
    color: #1f2937;
}

.header {
    padding: 30px 0;
    border-bottom: 1px solid #e5e7eb;
}

.section {
    background-color: #ffffff;
    padding: 30px;
    border-radius: 10px;
    margin-top: 25px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
}

.footer {
    text-align: center;
    margin-top: 60px;
    color: #6b7280;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="header">
    <h1>NextWord Prediction</h1>
    <h3>LSTM & RNN Based Language Model</h3>
    <p>
        A deep learning system that predicts the next word in a sentence
        using sequence modeling techniques.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_and_assets():
    model = load_model("lstn_model (1).h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


model, tokenizer, max_len = load_model_and_assets()

# ---------------- PREDICTION FUNCTION ----------------
def predict_next_word(text):
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = np.array(seq)

    if len(seq) > max_len:
        seq = seq[-max_len:]

    seq = np.pad(seq, (max_len - len(seq), 0), mode='constant')
    seq = np.expand_dims(seq, axis=0)

    prediction = model.predict(seq)
    predicted_index = np.argmax(prediction)

    return tokenizer.index_word.get(predicted_index, "")

# ---------------- USER INPUT ----------------
st.markdown("<div class='section'>", unsafe_allow_html=True)

st.subheader("🔍 Predict the Next Word")

user_input = st.text_input(
    "Enter a sentence:",
    placeholder="Machine learning models are able to"
)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = predict_next_word(user_input)
        st.success(f"Predicted Next Word: **{result}**")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- MODEL INFO ----------------
st.markdown("""
<div class="section">
<h3>📘 Model Information</h3>

<ul>
<li>Architecture: LSTM-based Recurrent Neural Network</li>
<li>Framework: TensorFlow / Keras</li>
<li>Sequence Padding: Pre-padding</li>
<li>Tokenizer: Keras Tokenizer</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ---------------- ABOUT YOU ----------------
st.markdown("""
<div class="section">
<h3>👨‍💻 About the Developer</h3>

<p>
<b>Gaurav Singh</b><br>
AI & Data Science Enthusiast<br><br>
Focused on building intelligent systems using deep learning and real-world NLP applications.
</p>
</div>
""", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
Built with Python, Streamlit & TensorFlow • © 2025
</div>
""", unsafe_allow_html=True)
