import streamlit as st
from PyPDF2 import PdfReader
import pdfplumber
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os
import faiss
from docx import Document
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import pandas as pd


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def create_faiss_index(dimension):
    index = faiss.IndexFlatL2(dimension)  # L2 distance
    return index


def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text


def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)


def extract_text_from_image_pdf(pdf_file):
    model = ocr_predictor(pretrained=True)
    doc = DocumentFile.from_pdf(pdf_file)
    result = model(doc)
    return result.pages[0].to_string()


def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()


def calculate_similarity(faiss_index, job_desc_embedding, resume_embeddings):
    distances, indices = faiss_index.search(job_desc_embedding.reshape(1, -1), len(resume_embeddings))
    return indices[0], distances[0]


def main():
    
    result_data = {'Name': [], 'Similarity': []}
    
    st.set_page_config(page_title="ATS with BERT & Cosin_similarity", page_icon=':books:')
    st.title("Resume Screening System using BERT & Cosin_similarity")
    st.header('Upload Resumes and Match with Job Description')

    
    job_description_input = st.text_area(
        "Enter Job Description", 
        """
        We are looking for a Data Analyst who can collect, process, and perform statistical analyses on large datasets. 
        You should be proficient in SQL, Excel, and Python. The candidate must have strong analytical skills, 
        experience in building reports, and the ability to communicate data-driven insights to stakeholders.
        """
    )

    
    with st.sidebar:
        st.subheader("Your Resumes")
        uploaded_files = st.file_uploader("Upload your resumes (PDF, DOCX, or Image PDFs)", accept_multiple_files=True, type=["pdf", "docx","png", "jpg", "jpeg"])

    
    if st.button("Process and Match Resumes"):
        if uploaded_files and job_description_input:
            st.info("Processing job description...")

            
            job_desc_embedding = embed_text(job_description_input).cpu().numpy().reshape(1, -1)
            resume_embeddings = []
            resume_names = []

            
            faiss_index = create_faiss_index(dimension=768)

            for file in uploaded_files:
                st.info(f"Processing {file.name}...")

                
                file_extension = os.path.splitext(file.name)[1].lower()
                if file_extension == ".pdf":
                    try:
                        
                        resume_text = extract_text_from_pdf(file)
                        if not resume_text:
                            
                            resume_text = extract_text_from_image_pdf(file)
                    except:
                        
                        resume_text = extract_text_from_image_pdf(file)
                elif file_extension == ".docx":
                    
                    resume_text = extract_text_from_docx(file)
                else:
                    st.warning(f"Unsupported file format for {file.name}. Skipping...")
                    continue

                if resume_text:
                    
                    resume_embedding = embed_text(resume_text).cpu().numpy()
                    resume_embeddings.append(resume_embedding)
                    resume_names.append(file.name)
                else:
                    st.warning(f"No readable text found in {file.name}. Skipping...")

            
            if resume_embeddings:
                resume_embeddings_np = torch.stack([torch.tensor(e) for e in resume_embeddings]).cpu().numpy()
                faiss_index.add(resume_embeddings_np)

                
                indices, distances = calculate_similarity(faiss_index, job_desc_embedding, resume_embeddings_np)

                
                matching_results = sorted(zip(indices, distances), key=lambda x: x[1])

                
                st.subheader("Matching Results")
                for idx, dist in matching_results:
                    resume_name = resume_names[idx]
                    similarity_score = 100 - dist
                    st.write(f"Resume: {resume_name}, Similarity Score: {similarity_score:.2f}%")
                    
                    
                    result_data['Name'].append(resume_name)
                    result_data['Similarity'].append(similarity_score)

           
            if result_data['Name']:  
                df = pd.DataFrame(result_data)
                st.subheader("Similarity Scores for Resumes")
                st.dataframe(df)
            else:
                st.warning("No valid resumes processed.")

        else:
            st.warning("Please upload at least one resume and enter a job description.")

if __name__ == "__main__":
    main()
