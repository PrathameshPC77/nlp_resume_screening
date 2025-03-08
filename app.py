import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

# Function to rank resumes
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    job_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_vector], resume_vectors).flatten()

    return cosine_similarities

# Streamlit UI
st.title("AI-Powered Resume Screening & Ranking System")

st.header("Job Description")
job_description = st.text_area("Enter the job description")

st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    resumes = [extract_text_from_pdf(file) for file in uploaded_files]

    scores = rank_resumes(job_description, resumes)

    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)

    st.write(results)
