import csv
import os
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

def ensure_data_directory():
    """Create data directory if it doesn't exist"""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir

def create_sample_data(csv_path):
    """Create sample cadastre data if file doesn't exist"""
    sample_data = [
        {
            "Reference": "CAD001",
            "Uso principal": "Residential",
            "Superficie construida": "150.5"
        },
        {
            "Reference": "CAD002",
            "Uso principal": "Commercial",
            "Superficie construida": "320.0"
        },
        {
            "Reference": "CAD003",
            "Uso principal": "Industrial",
            "Superficie construida": "1500.0"
        }
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Reference", "Uso principal", "Superficie construida"])
        writer.writeheader()
        writer.writerows(sample_data)
    print(f"Created sample data file: {csv_path}")

def load_cadastre_data(csv_file):
    """Load and process cadastre data from CSV file"""
    documents = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            # Skip error entries
            if row['Uso principal'] == 'Error':
                continue
                
            # Create content from row data
            content = f"Reference: {row['Reference']}\n"
            content += f"Main use: {row['Uso principal']}\n"
            content += f"Built area: {row['Superficie construida']}"
            
            # Create document
            doc = Document(
                page_content=content,
                metadata={
                    "reference": row['Reference'],
                    "use": row['Uso principal'],
                    "area": row['Superficie construida']
                }
            )
            documents.append(doc)
            
    return documents

def create_vector_store(documents, store_path="vector_store"):
    """Create FAISS vector store from documents"""
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create and save vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(store_path)
    
    return vector_store

def main():
    # Setup paths
    data_dir = ensure_data_directory()
    csv_file = data_dir / "cadastre_results.csv"
    store_path = data_dir / "vector_store"
    
    # Create sample data if needed
    if not csv_file.exists():
        create_sample_data(csv_file)
    
    print("Loading cadastre data...")
    documents = load_cadastre_data(str(csv_file))
    print(f"Loaded {len(documents)} documents")
    
    print("Creating vector store...")
    vector_store = create_vector_store(documents, str(store_path))
    print(f"Vector store created and saved to {store_path}")

if __name__ == "__main__":
    main()
