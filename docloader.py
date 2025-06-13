import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re 
import unicodedata
import nltk
from nltk.tokenize import sent_tokenize

def ensure_punkt():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        # Choose a download directory, e.g., ./nltk_data inside the app folder
        download_dir = os.path.join(os.getcwd(), 'nltk_data')
        os.makedirs(download_dir, exist_ok=True)
        nltk.data.path.append(download_dir)
        nltk.download('punkt', download_dir=download_dir)

# Call this early, e.g., at module load or in your Streamlit app's startup
ensure_punkt()

char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
token_splitter = False

def load_pdf(path: str) -> str:
    """Load a PDF file and concatenate all pages' text."""
    reader = PyPDF2.PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        # preserve a newline to avoid sentence merging across pages
        pages.append(text)
    return "\n".join(pages)

def clean_text(text: str) -> str:
    """
    Normalize unicode, strip control characters,
    and collapse whitespace to clean PDF extraction artifacts.
    """
    text = unicodedata.normalize("NFKC", text)
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_text(text: str,
               use_sentences: bool = True,
               use_token_splitter: bool = False) -> list[str]:
    """
    Split text into coherent chunks.
    1. Sentence-aware splitting to avoid cutting mid-sentence.
    2. Token/char-based splitting to enforce chunk size and overlap.
    """
    # 1) Sentence-aware blocks
    if use_sentences:
        sentences = sent_tokenize(text)
        blocks = []
        current = []
        cur_len = 0
        for sent in sentences:
            # If the block would exceed ~1000 chars, start a new one
            if cur_len + len(sent) > 1000 and current:
                blocks.append(' '.join(current))
                current = [sent]
                cur_len = len(sent)
            else:
                current.append(sent)
                cur_len += len(sent)
        if current:
            blocks.append(' '.join(current))
    else:
        blocks = [text]

    # 2) Further split each block by token or character
    splitter = token_splitter if use_token_splitter else char_splitter
    chunks = []
    for block in blocks:
        chunks.extend(splitter.split_text(block))

    return chunks

def load_documents_from_folder(folder_path: str) -> list[dict]:
    """
    Walk a folder, load each PDF, clean & split into chunks,
    and return list of {{filename, text}} documents for embedding.
    """
    docs = []
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith('.pdf'):
            continue
        path = os.path.join(folder_path, fname)
        raw = load_pdf(path)
        clean = clean_text(raw)
        chunks = split_text(clean,
                            use_sentences=True,
                            use_token_splitter=False)
        for i, chunk in enumerate(chunks):
            docs.append({
                'filename': f"{fname}_chunk{i}",
                'text': chunk
            })
    return docs

# def load_documents_from_folder(folder_path):
#     documents = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".pdf"):
#             text = load_pdf(os.path.join(folder_path, filename))
#             documents.append({"filename": filename, "text": text})
#     return documents