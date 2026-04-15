import streamlit as st
from transformers import pipeline

# Load model (cached for performance)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# ✅ Initialize summarizer
summarizer = load_summarizer()

# UI
st.set_page_config(page_title="AI Text Summarizer", layout="centered")

st.title("🤖 AI Text Summarizer")
st.write("Enter a long text below, and get a concise summary!")

# Input text
long_text = st.text_area("Enter text to summarize:", height=200)

# Controls
max_length = st.slider("Max Summary Length", min_value=50, max_value=300, value=130)
min_length = st.slider("Min Summary Length", min_value=20, max_value=100, value=30)

# Button action
if st.button("Summarize"):
    if long_text.strip():
        with st.spinner("Generating summary... ⏳"):
            try:
                # ⚡ Limit input size (important for model)
                safe_text = long_text[:1024]

                summary = summarizer(
                    safe_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )

                st.subheader("📌 Summary:")
                st.success(summary[0]['summary_text'])

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("⚠️ Please enter some text to summarize.")
