import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NextWord Prediction | LSTM & RNN",
    page_icon="📘",
    layout="wide"
)

# ---------------- DEBUG: SHOW FILES ----------------
st.write("Files available in directory:")
st.write(os.listdir())

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

# ---------------- UI HEADER ----------------
st.markdown("""
<h1>NextWord Prediction</h1>
<h3>LSTM & RNN Based Language Model</h3>
<hr>
""", unsafe_allow_html=True)

# ---------------- PREDICTION FUNCTION ----------------
def predict_next_word(text):
    seq = tokenizer.texts_to_sequences([text])[0]

    if len(seq) > max_len:
        seq = seq[-max_len:]

    seq = np.pad(seq, (max_len - len(seq), 0), mode="constant")
    seq = np.expand_dims(seq, axis=0)

    prediction = model.predict(seq)
    predicted_index = np.argmax(prediction)

    return tokenizer.index_word.get(predicted_index, "")

# ---------------- USER INPUT ----------------
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

# ---------------- INFO ----------------
st.markdown("""
### 📘 Model Info
- Architecture: LSTM (Recurrent Neural Network)
- Framework: TensorFlow / Keras
- Tokenization: Keras Tokenizer
- Padding: Pre-padding
""")

st.markdown("""
---
👨‍💻 **Developed by Gaurav Singh**  
AI & Data Science Enthusiast  
""")
