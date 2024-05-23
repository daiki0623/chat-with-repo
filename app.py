from typing import Iterator
import streamlit as st
from add_document import add_documents
import settings

from dotenv import load_dotenv
from utils import my_logger
from chat_assistant import ChatAssistant



def main():
    st.title("langchain-streamlit-app")

    with st.sidebar:
        repo_url = st.text_input(label="Enter repository URL", value="https://github.com/langchain-ai/langchain")
        if st.button("Scan", type="primary"):
            add_documents(repo_url)

    models = tuple(d["model"] for d in settings.MODEL_OPTIONS)
    selected_model = st.selectbox("Select Model", models)
    on_rag = st.toggle("RAG")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("What's up?")

    if prompt:
        st.session_state.messages.append({"role": settings.ROLE_USER, "content": prompt})

        with st.chat_message(settings.ROLE_USER):
            st.markdown(prompt)

        with st.chat_message(settings.ROLE_ASSISTANT):
            assistant = ChatAssistant()
            response_generator: Iterator = assistant.respond(selected_model, prompt, on_rag)
            response = st.write_stream(response_generator)

        st.session_state.messages.append({"role": settings.ROLE_ASSISTANT, "content": response})

if __name__ == "__main__":
    load_dotenv()
    logger = my_logger.set_logger(__name__)

    main()