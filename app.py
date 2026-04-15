import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

st.title("Sentence Summarizer")
st.write("Enter a sentence to summarize:")
sentence = st.text_area("", height=100)

if st.button("Summarize"):
    if sentence.strip():
        embeddings = model.encode([sentence])
        summary = sentence
        st.subheader("Summary:")
        st.success(summary)
    else:
        st.warning("Please enter a sentence to summarize.")
