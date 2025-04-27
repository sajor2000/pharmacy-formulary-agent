#!/usr/bin/env python3
"""
Memory-Optimized PDF Processor for Pharmacy Formulary Agent
---------------------------------------------------------
Processes remaining PDF formulary documents with enhanced memory management.
"""

import os
import time
import gc  # Garbage collection
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

# Add the code_reference directory to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'code_reference'))

# Import from the original codebase
from code_reference.document_processor import DocumentProcessor

def get_processed_files():
    """Get list of already processed files from Pinecone using batched queries to save memory"""
    try:
        from pinecone import Pinecone
        import os
        
        # Get Pinecone API key
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Connect to the 'form' index
        index = pc.Index("form")
        
        # Use multiple small queries with different random vectors to sample the database
        processed_files = set()
        batch_size = 20  # Small batch size to avoid memory issues
        num_batches = 5  # Multiple batches to get a good sample
        
        print("Sampling Pinecone database to find processed files...")
        for i in range(num_batches):
            # Create a random query vector
            query_vector = np.random.rand(1024).tolist()
            
            # Query with small batch size
            results = index.query(
                namespace="formulary",
                vector=query_vector,
                top_k=batch_size,
                include_metadata=True
            )
            
            # Extract source files
            for match in results.matches:
                if hasattr(match, 'metadata') and match.metadata and 'source' in match.metadata:
                    processed_files.add(match.metadata['source'])
            
            print(f"Batch {i+1}/{num_batches} complete. Found {len(processed_files)} unique files so far.")
            
            # Force garbage collection after each batch
            gc.collect()
        
        return list(processed_files)
    except Exception as e:
        print(f"Error getting processed files: {e}")
        return []

def get_insurance_from_filename(filename):
    """Extract insurance provider from filename"""
    # Common insurance providers in the filenames
    providers = {
        "UHC": "UnitedHealthcare",
        "BCBS": "Blue Cross Blue Shield",
        "Cigna": "Cigna",
        "Express": "Express Scripts",
        "Humana": "Humana",
        "Meridian": "Meridian",
        "Wellcare": "Wellcare",
        "County": "CountyCare"
    }
    
    for key, value in providers.items():
        if key in filename:
            return value
    
    return "Unknown"

def chunk_text(text, chunk_size=3000, overlap=200):
    """Split text into chunks of specified size with overlap"""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        # If this is not the last chunk, try to find a good break point
        if end < text_len:
            # Try to find a period, newline, or space to break at
            for char in ['. ', '\n', ' ']:
                pos = text.rfind(char, start, end)
                if pos != -1:
                    end = pos + 1  # Include the breaking character
                    break
        
        chunks.append(text[start:end])
        start = end - overlap if end < text_len else text_len
    
    return chunks

def process_pdf_file(processor, filename, chunk_size=3000):
    """Process a single PDF file with memory optimization"""
    pdf_path = os.path.join(processor.pdf_dir, filename)
    print(f"\nProcessing {filename}...")
    file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # Convert to MB
    print(f"File size: {file_size:.2f} MB")
    
    file_embeddings = []
    
    try:
        # Extract text
        text = processor.extract_text_from_pdf(pdf_path)
        text_length = len(text)
        print(f"Extracted {text_length} characters of text")
        
        # Force garbage collection after text extraction
        gc.collect()
        
        # Extract tables
        tables = processor.extract_tables_from_pdf(pdf_path)
        print(f"Extracted {len(tables)} tables")
        
        # Force garbage collection after table extraction
        gc.collect()
        
        # Chunk the text to avoid token limits
        if text:
            chunks = chunk_text(text, chunk_size)
            print(f"Split text into {len(chunks)} chunks")
            
            # Clear text variable to free memory
            del text
            gc.collect()
            
            for chunk_idx, chunk in enumerate(chunks):
                print(f"Creating embedding for text chunk {chunk_idx+1}/{len(chunks)}...")
                try:
                    chunk_embedding = processor.get_embedding(chunk)
                    if isinstance(chunk_embedding, list) and len(chunk_embedding) > 0:
                        file_embeddings.append({
                            'content': chunk[:200] + '...',  # Shorter preview to save memory
                            'embedding': chunk_embedding,
                            'metadata': {
                                'source': filename, 
                                'type': 'text_chunk', 
                                'chunk_idx': chunk_idx,
                                'insurance': get_insurance_from_filename(filename)
                            }
                        })
                        
                        # Clear variables to free memory
                        del chunk_embedding
                        
                        # Force garbage collection after each embedding
                        if chunk_idx % 5 == 0:  # Do garbage collection every 5 chunks
                            gc.collect()
                            
                except Exception as e:
                    print(f"Error creating embedding for chunk {chunk_idx+1}: {e}")
                
                # Clear chunk variable to free memory
                del chunk
        
        # Embeddings for tables
        for i, table in enumerate(tables):
            if isinstance(table.get('data'), pd.DataFrame) and not table['data'].empty:
                print(f"Creating embedding for table {i+1}...")
                table_str = table['data'].to_string()
                try:
                    table_embedding = processor.get_embedding(table_str)
                    if isinstance(table_embedding, list) and len(table_embedding) > 0:
                        file_embeddings.append({
                            'content': f"Table from page {table['page']}:\n{table_str[:200]}...",
                            'embedding': table_embedding,
                            'metadata': {
                                'source': filename,
                                'type': 'table',
                                'page': table['page'],
                                'insurance': get_insurance_from_filename(filename)
                            }
                        })
                        
                        # Clear variables to free memory
                        del table_embedding
                        del table_str
                        
                except Exception as e:
                    print(f"Error creating embedding for table {i+1}: {e}")
        
        # Clear tables variable to free memory
        del tables
        
        print(f"Created {len(file_embeddings)} embeddings for {filename}")
        return file_embeddings
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return []

def process_remaining_pdfs(batch_size=2, delay_between_batches=15, chunk_size=3000):
    """Process remaining PDF files with enhanced memory management"""
    processor = DocumentProcessor()
    
    # Get list of all PDF files
    try:
        all_pdf_files = [f for f in os.listdir(processor.pdf_dir) if f.endswith('.pdf')]
    except FileNotFoundError:
        print(f"Error: Directory '{processor.pdf_dir}' not found.")
        print(f"Creating directory: {processor.pdf_dir}")
        os.makedirs(processor.pdf_dir)
        print(f"Please add your PDF formulary documents to the '{processor.pdf_dir}' directory and run this script again.")
        return []
    
    # Get list of already processed files from Pinecone
    processed_files = get_processed_files()
    
    # Determine which files still need processing
    remaining_files = list(set(all_pdf_files) - set(processed_files))
    
    if not remaining_files:
        print("All PDF files have already been processed!")
        return []
    
    total_files = len(remaining_files)
    print(f"Found {total_files} PDF files that still need processing:")
    for i, filename in enumerate(sorted(remaining_files), 1):
        file_path = os.path.join(processor.pdf_dir, filename)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        print(f"{i}. {filename} ({file_size:.2f} MB)")
    
    # Sort files by size (process smaller files first)
    remaining_files.sort(key=lambda f: os.path.getsize(os.path.join(processor.pdf_dir, f)))
    print("\nFiles sorted by size (smallest first) to optimize processing:")
    for i, filename in enumerate(remaining_files, 1):
        file_path = os.path.join(processor.pdf_dir, filename)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        print(f"{i}. {filename} ({file_size:.2f} MB)")
    
    # Process in small batches
    total_embeddings_count = 0
    num_batches = (total_files + batch_size - 1) // batch_size
    
    for batch_num in range(1, num_batches + 1):
        start_idx = (batch_num - 1) * batch_size
        end_idx = min(start_idx + batch_size, total_files)
        batch = remaining_files[start_idx:end_idx]
        
        print(f"\n=== Processing Batch {batch_num}/{num_batches} ===")
        print(f"Files: {batch}")
        
        # Process this batch
        batch_embeddings = []
        for filename in batch:
            # Process each file individually
            file_embeddings = process_pdf_file(processor, filename, chunk_size)
            
            # Store embeddings immediately after processing each file
            if file_embeddings:
                print(f"\nStoring {len(file_embeddings)} embeddings for {filename} in Pinecone...")
                success = processor.store_in_pinecone(file_embeddings)
                if success:
                    print(f"Successfully stored embeddings for {filename} in Pinecone")
                    total_embeddings_count += len(file_embeddings)
                else:
                    print(f"Failed to store embeddings for {filename} in Pinecone")
            
            # Clear file_embeddings to free memory
            del file_embeddings
            gc.collect()
            
            print(f"\nMemory cleanup after processing {filename}")
            print("Forcing garbage collection...")
            gc.collect()
        
        # Delay between batches to avoid rate limits and allow memory cleanup
        if batch_num < num_batches:
            print(f"\nWaiting {delay_between_batches} seconds before processing next batch...")
            print("This delay helps avoid rate limits and allows memory cleanup.")
            time.sleep(delay_between_batches)
            
            # Force garbage collection during the delay
            gc.collect()
    
    print(f"\n=== Processing Complete ===")
    print(f"Total embeddings created and stored: {total_embeddings_count}")
    return total_embeddings_count

if __name__ == "__main__":
    print("=== Memory-Optimized PDF Processor for Pharmacy Formulary Agent ===")
    print("This script processes remaining PDFs with enhanced memory management.\n")
    
    # Process missing PDFs with small batch size and longer delays
    process_remaining_pdfs(batch_size=2, delay_between_batches=15, chunk_size=3000)
