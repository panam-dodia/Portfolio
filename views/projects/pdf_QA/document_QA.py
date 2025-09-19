import torch
import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader

# Load model (either from HuggingFace Hub or local path)
model_path = r"views\projects\pdf_QA\models\adc3b06f79f797d1c575d5479d6f5efe54a9e3b4"
# question_answer = pipeline("question-answering", model=model_path)

question_answer = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Function to extract text from PDF
def read_pdf_context(uploaded_file):
    try:
        pdf_reader = PdfReader(uploaded_file)
        context = ""
        for page in pdf_reader.pages:
            context += page.extract_text() + "\n"
        return context
    except Exception as e:
        return f"An error occurred while reading PDF: {e}"

# Streamlit UI
st.title("📄 PDF Question and Answer")
st.write("Upload a PDF and ask a question. The model will try to find the answer from the document.")

uploaded_file = st.file_uploader("Upload Your PDF", type=["pdf"])
question = st.text_area("Input Your Question", height=150)

if uploaded_file is not None and question.strip() != "":
    with st.spinner("Processing..."):
        context = read_pdf_context(uploaded_file)
        if context.startswith("An error occurred"):
            st.error(context)
        else:
            answer = question_answer(question=question, context=context)
            st.subheader("Answer:")
            st.success(answer["answer"])
