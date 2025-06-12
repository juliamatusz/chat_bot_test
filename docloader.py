import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(file_path):
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return "\n".join([page.extract_text() or "" for page in reader.pages])

def split_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            text = load_pdf(os.path.join(folder_path, filename))
            # Split the PDF text into chunks
            chunks = split_text(text)
            # Add each chunk as a separate document (better for RAG)
            for i, chunk in enumerate(chunks):
                documents.append({"filename": f"{filename}_chunk{i}", "text": chunk})
    return documents

# def load_documents_from_folder(folder_path):
#     documents = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".pdf"):
#             text = load_pdf(os.path.join(folder_path, filename))
#             documents.append({"filename": filename, "text": text})
#     return documents