import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/bart-tiny")

summarizer = load_summarizer()

st.title("Sentence Summarizer")
st.write("Enter a sentence to summarize:")
sentence = st.text_area("", height=100)

max_length = st.slider("Max Summary Length", min_value=10, max_value=50, value=20)
min_length = st.slider("Min Summary Length", min_value=5, max_value=20, value=10)

if st.button("Summarize"):
    if sentence.strip():
        with st.spinner("Generating summary... "):
            summary = summarizer(sentence, max_length=max_length, min_length=min_length, do_sample=False)
        st.subheader("Summary:")
        st.success(summary[0]['summary_text'])
    else:
        st.warning("Please enter a sentence to summarize.")
