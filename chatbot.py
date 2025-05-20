import os
import sqlite3
from flask import Flask, render_template, request, jsonify
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory  # Fixed import
from dotenv import load_dotenv

# Add import error handling
try:
    app = Flask(__name__)
except ImportError as e:
    print("Error: Required packages not installed. Please run:")
    print("pip install flask langchain-core langchain-community langchain-groq python-dotenv")
    raise e

# Custom loader for SQLite database
class SQLiteDocumentLoader(BaseLoader):
    def __init__(self, db_path):
        super().__init__()
        self.db_path = db_path

    def load(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT url, title, content FROM pages")
        documents = []
        for url, title, content in cursor.fetchall():
            documents.append(Document(
                page_content=content,
                metadata={"url": url, "title": title}
            ))
        conn.close()
        return documents

# Initialize chatbot components
def init_chatbot():
    # Load documents from database
    db_path = "ask_content.db"
    loader = SQLiteDocumentLoader(db_path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Set up conversational model and memory
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create conversational retrieval chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    return qa_chain

def validate_api_key():
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set. Please check your .env file.")
    return api_key

# Load environment variables
load_dotenv(override=True)
GROQ_API_KEY = validate_api_key()
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize the chain
try:
    qa_chain = init_chatbot()
except Exception as e:
    print(f"Error initializing chatbot: {e}")
    raise

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message")
        if not user_input:
            return jsonify({"error": "No message provided"}), 400
        response = qa_chain.invoke({"question": user_input})
        return jsonify({"response": response["answer"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)