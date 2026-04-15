import streamlit as st
from transformers import pipeline

# Load model once (cached)
@st.cache_resource
def load_summarizer():
    return pipeline(
        task="summarization",
        model="sshleifer/distilbart-cnn-12-6",
        framework="pt"
    )

# ✅ FIX: Initialize summarizer
summarizer = load_summarizer()

# UI
st.title("🤖 AI Text Summarizer")
st.write("Enter a long text below, and get a concise summary!")

# Input
long_text = st.text_area("Enter text to summarize:", height=200)

# Sliders
max_length = st.slider("Max Summary Length", 50, 300, 130)
min_length = st.slider("Min Summary Length", 20, 100, 30)

# Button
if st.button("Summarize"):
    if long_text.strip():
        try:
            with st.spinner("Generating summary..."):
                summary = summarizer(
                    long_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )

            st.subheader("📄 Summary:")
            st.success(summary[0]['summary_text'])

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("⚠️ Please enter some text to summarize.")
