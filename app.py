import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_summarizer():
    """Load the sentence summarization model."""
    return pipeline("summarization", model="sshleifer/bart-tiny")

# Initialize the summarizer
summarizer = load_summarizer()

# Set up the Streamlit app
st.title("Sentence Summarizer")
st.write("Enter a sentence to summarize:")

# Get the sentence from the user
sentence = st.text_area("", height=100)

# Get the max and min summary lengths from the user
max_length = st.slider("Max Summary Length", min_value=10, max_value=50, value=20)
min_length = st.slider("Min Summary Length", min_value=5, max_value=20, value=10)

# Summarize the sentence when the button is clicked
if st.button("Summarize"):
    if sentence.strip():
        with st.spinner("Generating summary... "):
            # Use the summarizer to generate a summary
            summary = summarizer(sentence, max_length=max_length, min_length=min_length, do_sample=False)
        st.subheader("Summary:")
        st.success(summary[0]['summary_text'])
    else:
        st.warning("Please enter a sentence to summarize.")
