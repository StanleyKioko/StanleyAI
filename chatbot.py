import os
import sys
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

def check_dependencies():
    required_packages = {
        'langchain_community': 'langchain-community',
        'langchain_groq': 'langchain-groq',
        'sentence_transformers': 'sentence-transformers',
        'faiss': 'faiss-cpu',
        'torch': 'torch'
    }
    
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print("Missing required packages. Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    return True

# Only import if dependencies are met
if check_dependencies():
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.chains import ConversationalRetrievalChain
    from langchain_groq import ChatGroq
    from langchain.memory import ConversationBufferMemory

def validate_api_key():
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("Error: GROQ_API_KEY not found in .env file")
        print("Please copy .env.template to .env and add your API key")
        return False
    return True

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    print("Error: GROQ_API_KEY not found in environment variables")
    sys.exit(1)

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

def initialize_chatbot():
    if not validate_api_key():
        return None
    
    # Check if FAQ file exists
    if not os.path.exists("faq.txt"):
        print("Error: faq.txt file not found!")
        return None
    
    try:
        # Step 1: Load and process the FAQ document
        loader = TextLoader("faq.txt")
        documents = loader.load()

        # Step 2: Split documents into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Step 3: Create embeddings and store in FAISS vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vector_store = FAISS.from_documents(chunks, embeddings)

        # Step 4: Set up the conversational model and memory
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

        # Step 5: Create the conversational retrieval chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            verbose=True
        )

        return qa_chain
    except Exception as e:
        print(f"Initialization error: {str(e)}")
        return None

# Step 6: Chatbot interaction loop
def run_chatbot():
    print("Welcome to the FAQ Chatbot (Powered by Groq)! Type 'exit' to quit.")
    while True:
        question = input("You: ")
        if question.lower() == "exit":
            print("Goodbye!")
            break
        # Get response from the chain
        try:
            response = qa_chain.invoke({"question": question})
            print(f"Bot: {response['answer']}")
        except Exception as e:
            if "invalid_api_key" in str(e):
                print("Error: Invalid API key. Please check your GROQ_API_KEY in .env file")
                break
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    if not check_dependencies():
        sys.exit(1)
        
    qa_chain = initialize_chatbot()
    if qa_chain:
        run_chatbot()
    else:
        print("Failed to initialize chatbot. Please check the errors above.")