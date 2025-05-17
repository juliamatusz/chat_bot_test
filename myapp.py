import streamlit as st
import random
import time
import openai
import fitz
from chat_openrouter import ChatOpenRouter

st.write("Test chat.")

api_key = st.secrets["API_KEY"]
base_url = st.secrets["BASE_URL"]
model_name = st.secrets["MODEL"] 

client = openai.OpenAI(api_key=api_key, base_url=base_url)

with st.sidebar:
    st.title("Upload PDFs")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        st.session_state.uploaded_texts = []

        for uploaded_file in uploaded_files:
            text = ""
            try:
                with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                    for page in doc:
                        text += page.get_text()
                st.session_state.uploaded_texts.append({
                    "filename": uploaded_file.name,
                    "content": text
                })
                st.success(f"{uploaded_file.name} loaded ({len(doc)} pages)")

            except Exception as e:
                st.error(f"Failed to read {uploaded_file.name}: {e}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
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