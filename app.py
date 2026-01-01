import streamlit as st
import numpy as np
import pickle

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
    <h3>LSTM & RNN Based Language Modeling System</h3>
    <p>
        A deep learning application that predicts the next word in a sentence 
        using sequence learning techniques.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    return model, tokenizer

model, tokenizer = load_model()

# ---------------- PREDICTION SECTION ----------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("🔍 Text Prediction")

user_input = st.text_input(
    "Enter a sentence to predict the next word:",
    placeholder="Example: Machine learning models are used to"
)

def predict_next_word(text):
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = np.array(seq).reshape(1, -1)
    prediction = model.predict(seq)
    return tokenizer.index_word[np.argmax(prediction)]

if st.button("Predict Next Word"):
    if user_input.strip() != "":
        output = predict_next_word(user_input)
        st.success(f"Predicted Next Word: **{output}**")
    else:
        st.warning("Please enter some text to continue.")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- MODEL INFO ----------------
st.markdown("""
<div class="section">
<h3>📘 Model Overview</h3>

<ul>
<li>Architecture: Recurrent Neural Network (RNN) with LSTM layers</li>
<li>Objective: Predict the most probable next word based on context</li>
<li>Training Data: Text corpus processed using tokenization and sequence modeling</li>
<li>Frameworks Used: TensorFlow / Keras</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ---------------- ABOUT YOU ----------------
st.markdown("""
<div class="section">
<h3>👨‍💻 About the Developer</h3>

<p>
<b>Gaurav Singh</b><br>
Aspiring Data Scientist | AI & Machine Learning Enthusiast  
<br><br>
Focused on building intelligent, practical, and scalable AI-driven applications 
that combine strong theoretical foundations with real-world usability.
</p>
</div>
""", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
Built with Python, Streamlit, and Deep Learning • © 2025
</div>
""", unsafe_allow_html=True)
