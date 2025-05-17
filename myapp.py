import streamlit as st
import random
import time
import openai
import fitz
from chat_openrouter import ChatOpenRouter
from docloader import load_documents_from_folder
from embedder import create_index, retrieve_docs
from chat_openrouter import ChatOpenRouter
from langchain.prompts import ChatPromptTemplate
import tempfile
import os

st.write("Test chat.")

api_key = st.secrets["API_KEY"]
base_url = st.secrets["BASE_URL"]
model_name = st.secrets["MODEL"] 

client = openai.OpenAI(api_key=api_key, base_url=base_url)

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

def answear_question(question, documents, model):
    context = "\n\n".join([doc["text"] for doc in documents])
    propmt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,  # Streaming response
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"Error: {e}"
            st.error(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
