import os
import sys
import logging
from flask_cors import CORS
import sqlite3
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Added import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from templates.create_documents_db import create_database
import shutil  # Add this import at the top

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_api_key():
    # Ensure .env is loaded
    load_dotenv(override=True)
    
    # First try environment variable
    api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        print("\nGROQ_API_KEY not found. Checking .env file...")
        
        # Try to read directly from .env
        env_path = Path(__file__).parent / '.env'
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith('GROQ_API_KEY='):
                        api_key = line.split('=')[1].strip()
                        break
    
    if not api_key:
        raise ValueError("""
GROQ_API_KEY not found! Please:
1. Create or edit .env file in the project directory
2. Add the line: GROQ_API_KEY=your_actual_api_key_here
3. Make sure there are no extra spaces or quotes
4. Save the file and try again""")
    
    return api_key

# Custom loader for SQLite database with documents
class SQLiteDocumentLoader(BaseLoader):
    def __init__(self, db_path):
        super().__init__()
        self.db_path = db_path

    def load(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT file_path, file_type, content, title FROM documents")
            documents = []
            for file_path, file_type, content, title in cursor.fetchall():
                documents.append(Document(
                    page_content=content,
                    metadata={"file_path": file_path, "file_type": file_type, "title": title}
                ))
            conn.close()
            return documents
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            print("Make sure you've initialized the database with: python templates/create_documents_db.py")
            return []

def ensure_database():
    """Ensure database exists and contains documents"""
    db_path = Path(__file__).parent / "documents.db"  # Changed from data/documents.db
    files_dir = Path(__file__).parent / "files"
    
    if not db_path.exists() or db_path.stat().st_size == 0:
        print("Initializing document database...")
        create_database(files_dir=str(files_dir), db_path=str(db_path))
    return db_path

def cleanup_vector_store():
    """Safely remove vector store directory"""
    vector_store_path = Path(__file__).parent / "vector_store"
    if vector_store_path.exists():
        try:
            shutil.rmtree(vector_store_path)
            print(f"Removed existing vector store at {vector_store_path}")
            return True
        except Exception as e:
            print(f"Error removing vector store: {e}")
            return False
    return True

# Initialize chatbot components
def init_chatbot(force_refresh=False):
    """Initialize chatbot with option to force vector store refresh"""
    # Initialize database
    db_path = ensure_database()
    vector_store_path = Path(__file__).parent / "vector_store"
    
    # Force refresh or missing vector store triggers reindexing
    if force_refresh or not vector_store_path.exists():
        # Load documents
        loader = SQLiteDocumentLoader(str(db_path))
        try:
            documents = loader.load()
            if not documents:
                print("No documents found. Please add files to the 'files' directory.")
                return None
                
            print(f"Loaded {len(documents)} documents for indexing")
            
            # Split documents into chunks with better context preservation
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # Smaller chunks for better context
                chunk_overlap=100,
                separators=["\n\n", "\n", " | ", ".", "!", "?", ",", " ", ""],
                length_function=len,
                keep_separator=True
            )
            chunks = text_splitter.split_documents(documents)
            print(f"Created {len(chunks)} chunks for embedding")

            # Create embeddings and vector store
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            vector_store = FAISS.from_documents(chunks, embeddings)
            vector_store.save_local(str(vector_store_path))
            print("Vector store created and saved successfully")
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None
    else:
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            vector_store = FAISS.load_local(
                str(vector_store_path),
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return None

    # Set up conversational model and memory
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create conversational retrieval chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        verbose=True  # Added for debugging
    )
    return qa_chain

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
CORS(app)  # Enable CORS

# Set up API key before anything else
try:
    api_key = load_api_key()
    os.environ["GROQ_API_KEY"] = api_key
    print("API key loaded successfully")
except Exception as e:
    print(f"Error: {str(e)}")
    sys.exit(1)

# Initialize the chain
try:
    qa_chain = init_chatbot()
    if not qa_chain:
        sys.exit(1)
except Exception as e:
    print(f"Error initializing chatbot: {e}")
    sys.exit(1)

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    if not qa_chain:
        logger.error("QA chain not initialized")
        return jsonify({"error": "Chatbot not initialized"}), 500
        
    try:
        user_input = request.json.get("message")
        logger.debug(f"Received message: {user_input}")
        
        if not user_input:
            return jsonify({"error": "No message provided"}), 400
            
        logger.debug("Invoking QA chain...")
        response = qa_chain({"question": user_input})  # Use direct call instead of invoke
        logger.debug(f"Got response: {response}")
        
        if not response or "answer" not in response:
            return jsonify({"error": "No response from chatbot"}), 500
            
        return jsonify({"response": response["answer"]})
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Add route to force refresh
@app.route("/refresh", methods=["POST"])
def refresh():
    """Force refresh of the vector store"""
    global qa_chain
    try:
        if not cleanup_vector_store():
            return jsonify({"error": "Failed to cleanup vector store"}), 500
        qa_chain = init_chatbot(force_refresh=True)
        if not qa_chain:
            return jsonify({"error": "Failed to initialize chatbot"}), 500
        return jsonify({"message": "Vector store refreshed successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    try:
        api_key = load_api_key()
        os.environ["GROQ_API_KEY"] = api_key
        qa_chain = init_chatbot()
        if qa_chain:
            logger.info("Starting Flask server...")
            app.run(debug=True, port=5000)  # Enable debug mode for development
    except Exception as e:
        logger.error(f"Startup error: {str(e)}", exc_info=True)
        sys.exit(1)