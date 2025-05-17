import streamlit as st
import fitz
import os
import tempfile

from docloader import load_documents_from_folder
from embedder import create_index, retrieve_docs
from chat_openrouter import ChatOpenRouter
from langchain.prompts import ChatPromptTemplate

st.title("Test Chat with PDF Context")


with st.sidebar:
    st.title("Upload PDFs")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        with tempfile.TemporaryDirectory() as tmpdir:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(tmpdir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
            documents = load_documents_from_folder(tmpdir)
            st.session_state.documents = documents
            st.session_state.index = create_index(documents)
            st.success(f"Loaded and indexed {len(documents)} documents.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

template = """
Write short answers, up to 4 sentences. When you don't know just write: I don't know.
Question: {question}
Context: {context}
Answer:
"""

def answer_with_context(question, index, model):
    top_docs = retrieve_docs(question, index)
    context = "\n\n".join([doc["text"] for doc in top_docs])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

if user_input := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            if "index" in st.session_state:
                response = answer_with_context(
                    user_input,
                    st.session_state.index,
                    ChatOpenRouter(model=st.secrets["MODEL"])
                )
            else:
                response = "Please upload PDF files to provide context."

            message_placeholder.markdown(response)
        except Exception as e:
            response = f"Error: {e}"
            st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})