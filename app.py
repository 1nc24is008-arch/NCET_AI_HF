import streamlit as st
from transformers import pipeline

# Load model (cached)
@st.cache_resource
def load_summarizer():
    return pipeline(
        task="summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

# ✅ IMPORTANT: initialize summarizer
summarizer = load_summarizer()

# UI
st.title("🤖 AI Text Summarizer")
st.write("Enter a paragraph or sentence to summarize.")

# Input
long_text = st.text_area("Enter text:", height=200)

# Controls
max_length = st.slider("Max Summary Length", 50, 300, 130)
min_length = st.slider("Min Summary Length", 20, 100, 30)

# Button
if st.button("Summarize"):
    if long_text.strip():
        with st.spinner("Generating summary..."):
            summary = summarizer(
                long_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )

        st.subheader("📄 Summary:")
        st.success(summary[0]['summary_text'])
    else:
        st.warning("⚠️ Please enter some text.")
