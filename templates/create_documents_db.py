import os
import sqlite3
from pathlib import Path
import shutil

def create_database(files_dir=None, db_path=None):
    """Create and initialize the documents database"""
    # Use default paths if none provided
    if files_dir is None:
        files_dir = Path(__file__).parent.parent / "files"
    if db_path is None:
        db_path = Path(__file__).parent.parent / "documents.db"  # Changed from data/documents.db
    
    # Ensure files directory exists
    files_dir = Path(files_dir)
    files_dir.mkdir(exist_ok=True)
    
    # Create parent directory for database if needed
    db_path = Path(db_path)
    db_path.parent.mkdir(exist_ok=True)
    
    # Initialize database
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create documents table with additional fields
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY,
        file_path TEXT,
        file_type TEXT,
        content TEXT,
        title TEXT,
        embedding TEXT,
        metadata TEXT
    )
    ''')
    
    # Process any existing files
    processed = 0
    for file_path in files_dir.glob('*.*'):
        # Simple text extraction for now
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                cursor.execute('''
                INSERT INTO documents (file_path, file_type, content, title)
                VALUES (?, ?, ?, ?)
                ''', (str(file_path), file_path.suffix, content, file_path.stem))
                processed += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"Database created at {db_path}")
    print(f"Processed {processed} documents")
    return db_path

if __name__ == "__main__":
    create_database()