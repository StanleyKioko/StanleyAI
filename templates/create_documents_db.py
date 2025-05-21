import os
import sqlite3
from pathlib import Path
import pandas as pd
from typing import List, Optional
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)

def process_excel_content(file_path: str) -> str:
    """Process Excel files to extract structured content"""
    try:
        df = pd.read_excel(file_path)
        # Convert DataFrame to a readable format
        content = []
        # Add column names as context
        content.append("Columns: " + ", ".join(df.columns.tolist()))
        # Add each row as a structured sentence
        for idx, row in df.iterrows():
            row_content = []
            for col in df.columns:
                if pd.notna(row[col]):  # Only include non-empty values
                    row_content.append(f"{col}: {row[col]}")
            content.append(" | ".join(row_content))
        return "\n".join(content)
    except Exception as e:
        print(f"Error processing Excel file {file_path}: {e}")
        return ""

def get_loader_for_file(file_path: str):
    """Return appropriate loader based on file extension"""
    ext = Path(file_path).suffix.lower()
    if ext in ['.xlsx', '.xls']:
        return None  # We'll handle Excel files separately
    return {
        '.pdf': PyPDFLoader,
        '.csv': CSVLoader,
        '.doc': UnstructuredWordDocumentLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.txt': TextLoader
    }.get(ext)

def create_database(files_dir=None, db_path=None):
    """Create and initialize the documents database"""
    # Use default paths if none provided
    if files_dir is None:
        files_dir = Path(__file__).parent.parent / "files"
    if db_path is None:
        db_path = Path(__file__).parent.parent / "documents.db"
    
    # Ensure directories exist
    files_dir = Path(files_dir)
    files_dir.mkdir(exist_ok=True)
    db_path.parent.mkdir(exist_ok=True)
    
    # Initialize database
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Drop existing table to ensure clean schema
    cursor.execute('DROP TABLE IF EXISTS documents')
    
    # Create documents table with all required columns
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT NOT NULL,
        file_type TEXT,
        content TEXT,
        title TEXT,
        metadata TEXT DEFAULT '{}',
        embedding BLOB,
        chunk_size INTEGER DEFAULT 1000,
        processed BOOLEAN DEFAULT 0,
        last_indexed TIMESTAMP,
        searchable BOOLEAN DEFAULT 1,
        last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Process files
    processed = 0
    for file_path in files_dir.glob('**/*'):  # Recursively scan subdirectories
        if file_path.is_file():
            try:
                content = ""
                metadata = "{}"
                
                # Handle Excel files separately
                if file_path.suffix.lower() in ['.xlsx', '.xls']:
                    content = process_excel_content(str(file_path))
                else:
                    loader_class = get_loader_for_file(str(file_path))
                    if loader_class:
                        loader = loader_class(str(file_path))
                        docs = loader.load()
                        content = "\n".join(doc.page_content for doc in docs)
                        metadata = str(docs[0].metadata if docs else {})
                
                if content:  # Only insert if we have content
                    cursor.execute('''
                    INSERT INTO documents 
                    (file_path, file_type, content, title, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    ''', (
                        str(file_path),
                        file_path.suffix,
                        content,
                        file_path.stem,
                        metadata
                    ))
                    processed += 1
                    print(f"Processed: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"Database created at {db_path}")
    print(f"Processed {processed} documents")
    return db_path

if __name__ == "__main__":
    create_database()