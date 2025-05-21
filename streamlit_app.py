import streamlit as st
from chatbot_groq_web import init_chatbot, load_api_key, cleanup_vector_store
import os

# Set page configuration
st.set_page_config(page_title="StanleyAI Chatbot", layout="wide")

# Initialize session state
if "qa_chain" not in st.session_state:
    try:
        # Load API key
        api_key = load_api_key()
        os.environ["GROQ_API_KEY"] = api_key
        
        # Initialize chatbot
        cleanup_vector_store()
        st.session_state.qa_chain = init_chatbot(force_refresh=True)
    except Exception as e:
        st.error(f"Error initializing chatbot: {e}")
        st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat header
st.title("StanleyAI Chatbot")

# Add refresh button
if st.button("Refresh Vector Store"):
    try:
        cleanup_vector_store()
        st.session_state.qa_chain = init_chatbot(force_refresh=True)
        st.success("Vector store refreshed successfully!")
    except Exception as e:
        st.error(f"Error refreshing vector store: {e}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get bot response
    try:
        response = st.session_state.qa_chain({"question": prompt})
        with st.chat_message("assistant"):
            st.write(response["answer"])
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    except Exception as e:
        st.error(f"Error getting response: {e}")
