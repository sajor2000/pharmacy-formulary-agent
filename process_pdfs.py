#!/usr/bin/env python3
"""
Process PDFs for Pharmacy Formulary Agent
---------------------------------------
Processes PDF formulary documents in small batches to avoid memory issues.
"""

import os
import time
import pandas as pd
import numpy as np
from document_processor import DocumentProcessor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

def chunk_text(text, chunk_size=4000, overlap=200):
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

def process_pdfs(batch_size=3, delay_between_batches=10, chunk_size=3000):
    """Process PDFs in small batches to avoid memory issues"""
    processor = DocumentProcessor()
    
    # Create data directory if it doesn't exist
    if not os.path.exists(processor.pdf_dir):
        os.makedirs(processor.pdf_dir)
        print(f"Created directory: {processor.pdf_dir}")
        print("Please add your PDF formulary documents to this directory and run this script again.")
        return []
    
    # Get list of all PDF files
    all_pdf_files = [f for f in os.listdir(processor.pdf_dir) if f.endswith('.pdf')]
    
    if not all_pdf_files:
        print(f"No PDF files found in {processor.pdf_dir}. Please add your PDF formulary documents and run this script again.")
        return []
    
    total_files = len(all_pdf_files)
    print(f"Found {total_files} PDF files to process:")
    for i, filename in enumerate(sorted(all_pdf_files), 1):
        file_path = os.path.join(processor.pdf_dir, filename)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        print(f"{i}. {filename} ({file_size:.2f} MB)")
    
    # Process in small batches
    all_embeddings = []
    num_batches = (total_files + batch_size - 1) // batch_size
    
    for batch_num in range(1, num_batches + 1):
        start_idx = (batch_num - 1) * batch_size
        end_idx = min(start_idx + batch_size, total_files)
        batch = all_pdf_files[start_idx:end_idx]
        
        print(f"\n=== Processing Batch {batch_num}/{num_batches} ===")
        print(f"Files: {batch}")
        
        # Process this batch
        batch_embeddings = []
        for filename in batch:
            pdf_path = os.path.join(processor.pdf_dir, filename)
            print(f"\nProcessing {filename}...")
            
            try:
                # Extract text
                text = processor.extract_text_from_pdf(pdf_path)
                
                # Extract tables
                tables = processor.extract_tables_from_pdf(pdf_path)
                
                # Create embeddings
                file_embeddings = []
                
                # Chunk the text to avoid token limits
                if text:
                    chunks = chunk_text(text, chunk_size)
                    print(f"Split text into {len(chunks)} chunks")
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        print(f"Creating embedding for text chunk {chunk_idx+1}/{len(chunks)}...")
                        try:
                            chunk_embedding = processor.get_embedding(chunk)
                            if isinstance(chunk_embedding, list) and len(chunk_embedding) > 0:
                                file_embeddings.append({
                                    'content': chunk[:500] + '...',  # Just store preview
                                    'embedding': chunk_embedding,
                                    'metadata': {
                                        'source': filename, 
                                        'type': 'text_chunk', 
                                        'chunk_idx': chunk_idx,
                                        'insurance': get_insurance_from_filename(filename)
                                    }
                                })
                                
                                # Clear memory after each embedding
                                del chunk_embedding
                                
                        except Exception as e:
                            print(f"Error creating embedding for chunk {chunk_idx+1}: {e}")
                
                # Embeddings for tables
                for i, table in enumerate(tables):
                    if isinstance(table.get('data'), pd.DataFrame) and not table['data'].empty:
                        print(f"Creating embedding for table {i+1}...")
                        table_str = table['data'].to_string()
                        try:
                            table_embedding = processor.get_embedding(table_str)
                            if isinstance(table_embedding, list) and len(table_embedding) > 0:
                                file_embeddings.append({
                                    'content': f"Table from page {table['page']}:\n{table_str[:500]}...",
                                    'embedding': table_embedding,
                                    'metadata': {
                                        'source': filename,
                                        'type': 'table',
                                        'page': table['page'],
                                        'insurance': get_insurance_from_filename(filename)
                                    }
                                })
                                
                                # Clear memory after each embedding
                                del table_embedding
                                
                        except Exception as e:
                            print(f"Error creating embedding for table {i+1}: {e}")
                
                batch_embeddings.extend(file_embeddings)
                print(f"Created {len(file_embeddings)} embeddings for {filename}")
                
                # Clear memory
                del file_embeddings
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        # Store batch embeddings in Pinecone
        if batch_embeddings:
            print(f"\nStoring {len(batch_embeddings)} embeddings from batch {batch_num} in Pinecone...")
            success = processor.store_in_pinecone(batch_embeddings)
            if success:
                print(f"Successfully stored batch {batch_num} embeddings in Pinecone")
                all_embeddings.extend(batch_embeddings)
            else:
                print(f"Failed to store batch {batch_num} embeddings in Pinecone")
        
        # Clear memory
        del batch_embeddings
        
        # Delay between batches to avoid rate limits and allow memory cleanup
        if batch_num < num_batches:
            print(f"\nWaiting {delay_between_batches} seconds before processing next batch...")
            print("This delay helps avoid rate limits and allows memory cleanup.")
            time.sleep(delay_between_batches)
    
    print(f"\n=== Processing Complete ===")
    print(f"Total embeddings created and stored: {len(all_embeddings)}")
    return all_embeddings

if __name__ == "__main__":
    print("=== Processing PDFs for Pharmacy Formulary Agent ===")
    print("This script processes PDFs in small batches to avoid memory issues.\n")
    print("Optimized for Mac M3 chip with 24GB RAM\n")
    
    # Process PDFs with optimized batch size for Mac M3 with 24GB RAM
    # Larger batch size and chunk size for more efficient processing
    process_pdfs(batch_size=5, delay_between_batches=5, chunk_size=5000)
