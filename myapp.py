import streamlit as st
import os
import tempfile

from docloader import load_documents_from_folder
from embedder import create_index, retrieve_docs
from chat_openrouter import ChatOpenRouter
from langchain.prompts import ChatPromptTemplate

st.title("🧭 Dopasuj kandydata do swoich poglądów")


with st.sidebar:
    st.title("Dodaj PDFy")
    uploaded_files = st.file_uploader("Dodaj pliki PDF", type=["pdf"], accept_multiple_files=True)

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
    st.session_state.messages = [{"role": "assistant", "content": "Porozmawiajmy o poglądach kandydatów!"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

template = """
Na podstawie poniższego kontekstu i pytania użytkownika wskaż jednego lub kilku kandydatów na prezydenta, których poglądy są najbardziej zgodne z preferencjami użytkownika. 

Podaj tylko imiona i nazwiska kandydatów oraz krótko wyjaśnij, dlaczego zostali zaproponowani. Nie twórz własnych opinii – bazuj wyłącznie na kontekście. Jeśli brak danych – napisz: „Nie wiem”.

Pytanie użytkownika:
{question}

Kontekst:
{context}

Rekomendowani kandydaci:
"""

def answer_with_context(question, index, model):
    top_docs = retrieve_docs(question, index)
    context = "\n\n".join([doc["text"] for doc in top_docs])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

def extract_pure_text(response):
    if isinstance(response, dict):
        return response.get("content", str(response))
    elif hasattr(response, "content"):
        return response.content
    else:
        return str(response)

if user_input := st.chat_input("Co jest dla Ciebie ważne?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            if "index" in st.session_state:
                response_to_clear = answer_with_context(
                    user_input,
                    st.session_state.index,
                    ChatOpenRouter(model=st.secrets["MODEL"])
                )
                response = extract_pure_text(response_to_clear)
            else:
                response = "Dodaj pliki PDF, aby zapewnić kontekst."
            message_placeholder.markdown(response)
        except Exception as e:
            response = f"Error: {e}"
            st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
