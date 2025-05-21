import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

def load_qa_chain():
    if st.session_state.qa_chain is not None:
        return st.session_state.qa_chain

    try:
        # Load API key
        load_dotenv(override=True)
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            st.error("GROQ_API_KEY not found in environment!")
            return None

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load existing vector store
        vector_store = FAISS.load_local(
            "vector_store",
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Set up chatbot
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3
        )
        
        # Configure memory correctly
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # Match the chain's output key
        )
        
        # Create chain
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=False  # Change to False to disable source tracking
        )
        return st.session_state.qa_chain
        
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        return None

def main():
    st.title("Document Q&A Bot")

    # Initialize session state
    init_session_state()
    
    # Load QA chain
    qa_chain = load_qa_chain()
    if not qa_chain:
        st.stop()

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your documents..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = qa_chain({"question": prompt})
                answer = response["answer"]
                st.markdown(answer)  # Display only the answer without sources
                
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
