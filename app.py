import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# ------------------ PDF PROCESSING ------------------

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def get_text_chunks(raw_text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(raw_text)


def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )
    return FAISS.from_texts(text_chunks, embeddings)


# ------------------ CONVERSATION CHAIN ------------------

def get_conversation_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    chat_history = InMemoryChatMessageHistory()

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Answer the question using ONLY the context below.\n\n{context}"
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    def ask(question: str):
        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        messages = prompt.format_messages(
            context=context,
            history=chat_history.messages,
            question=question
        )

        response = llm.invoke(messages)

        chat_history.add_user_message(question)
        chat_history.add_ai_message(response.content)

        return response.content

    return ask


# ------------------ STREAMLIT APP ------------------

def main():
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📑")
    st.title("Chat with Multiple PDFs 📑")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_question = st.chat_input("Ask a question about your documents")

    if user_question and st.session_state.conversation:
        answer = st.session_state.conversation(user_question)
        st.session_state.messages.append(("user", user_question))
        st.session_state.messages.append(("assistant", answer))

    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.write(content)

    with st.sidebar:
        st.subheader("Your documents")

        pdf_docs = st.file_uploader(
            "Upload PDFs and click Process",
            accept_multiple_files=True
        )

        if st.button("Process") and pdf_docs:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

            st.success("Documents processed successfully!")


if __name__ == "__main__":
    main()
