#!/usr/bin/env python3
"""
M3 Memory-Optimized PDF Processor
---------------------------------
A highly optimized PDF processor for Mac M3 chips with 24GB RAM.
Includes memory management, parallel processing, and efficient chunking.
"""

import os
import gc
import time
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm
from pinecone import Pinecone
from document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('m3_optimized_processor')

# Load environment variables
load_dotenv()

def chunk_text(text, chunk_size):
    """Split text into chunks of approximately equal size"""
    # Simple chunking by character count
    chunks = []
    
    # If text is short enough, return as a single chunk
    if len(text) <= chunk_size:
        return [text]
    
    # Otherwise, split into chunks of approximately chunk_size
    # Try to split at paragraph or sentence boundaries when possible
    start = 0
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to find a paragraph break
        paragraph_break = text.rfind('\n\n', start, end)
        if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
            end = paragraph_break + 2
        else:
            # Try to find a newline
            newline = text.rfind('\n', start, end)
            if newline != -1 and newline > start + chunk_size // 2:
                end = newline + 1
            else:
                # Try to find a sentence break (period followed by space)
                sentence_break = text.rfind('. ', start, end)
                if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                    end = sentence_break + 2
                else:
                    # Last resort: find a space
                    space = text.rfind(' ', start, end)
                    if space != -1:
                        end = space + 1
        
        chunks.append(text[start:end])
        start = end
    
    return chunks

def process_pdf_file(pdf_file, pdf_dir, chunk_size, index, namespace):
    """Process a single PDF file and store embeddings in Pinecone"""
    try:
        # Extract filename without extension for metadata
        filename = os.path.splitext(os.path.basename(pdf_file))[0]
        
        # Initialize document processor
        processor = DocumentProcessor(pdf_dir)
        
        # Process the PDF
        logger.info(f"Processing {pdf_file}...")
        
        # Extract text from PDF
        text = processor.extract_text_from_pdf(pdf_file)
        if not text:
            logger.warning(f"No text extracted from {pdf_file}")
            return False
        
        # Get insurance provider from filename if possible
        insurance_provider = None
        if " " in filename:
            parts = filename.split(" ")
            if len(parts) >= 2:
                insurance_provider = parts[1]  # Assuming format like "2023 BCBS Formulary.pdf"
        
        # Chunk the text
        logger.info(f"Chunking text from {pdf_file} with chunk size {chunk_size}...")
        chunks = chunk_text(text, chunk_size)
        logger.info(f"Created {len(chunks)} chunks from {pdf_file}")
        
        # Create embeddings and store in Pinecone
        logger.info(f"Creating embeddings for {pdf_file}...")
        
        # Process chunks in smaller batches to manage memory
        batch_size = 5  # Small batch size to prevent memory issues
        total_chunks = len(chunks)
        
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size} for {pdf_file}")
            
            # Create embeddings for this batch
            batch_embeddings = []
            for j, chunk in enumerate(batch_chunks):
                logger.info(f"Creating embedding for text chunk {i+j+1}/{total_chunks}...")
                embedding = processor.get_embedding(chunk)
                
                if embedding is None:
                    logger.warning(f"Failed to create embedding for chunk {i+j+1} in {pdf_file}")
                    continue
                
                # Prepare metadata
                metadata = {
                    'source': filename,
                    'content': chunk,
                    'type': 'text',
                }
                
                if insurance_provider:
                    metadata['insurance'] = insurance_provider
                
                # Store in Pinecone
                batch_embeddings.append({
                    'id': f"{filename}_chunk_{i+j+1}",
                    'values': embedding,
                    'metadata': metadata
                })
            
            # Upsert batch to Pinecone
            if batch_embeddings:
                try:
                    index.upsert(vectors=batch_embeddings, namespace=namespace)
                    logger.info(f"Added {len(batch_embeddings)} embeddings from batch {i//batch_size + 1}")
                except Exception as e:
                    logger.error(f"Error upserting to Pinecone: {e}")
            
            # Force garbage collection after each batch
            gc.collect()
            
            # Small delay between batches to allow system to recover
            time.sleep(1)
        
        logger.info(f"Completed processing {pdf_file}")
        return True
    
    except Exception as e:
        logger.error(f"Error processing {pdf_file}: {e}")
        return False
    finally:
        # Force garbage collection
        gc.collect()

def get_already_processed_files(index, namespace):
    """Get list of already processed files from Pinecone"""
    try:
        # Query Pinecone for existing vectors
        stats = index.describe_index_stats()
        
        if namespace in stats.namespaces:
            # Get a sample of vectors to extract source filenames
            query_vector = [0.0] * 1024  # Assuming 1024-dimensional vectors
            results = index.query(
                namespace=namespace,
                vector=query_vector,
                top_k=100,
                include_metadata=True
            )
            
            # Extract unique source filenames
            processed_files = set()
            for match in results.matches:
                if hasattr(match, 'metadata') and 'source' in match.metadata:
                    processed_files.add(match.metadata['source'])
            
            return processed_files
        else:
            return set()
    
    except Exception as e:
        logger.error(f"Error getting processed files: {e}")
        return set()

def process_pdfs_with_memory_optimization(pdf_dir="data", chunk_size=2000, max_workers=2):
    """
    Process PDF files with memory optimization for Mac M3 with 24GB RAM
    
    Args:
        pdf_dir: Directory containing PDF files
        chunk_size: Size of text chunks for embedding
        max_workers: Maximum number of parallel workers
    """
    try:
        # Initialize Pinecone
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            logger.error("PINECONE_API_KEY not found in environment variables")
            return
        
        # Initialize Pinecone with new API
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Check if 'form' index exists, if not create it
        index_name = "form"
        namespace = "formulary"
        dimension = 1024
        
        # Get list of indexes
        existing_indexes = pc.list_indexes()
        
        if index_name not in existing_indexes.names():
            logger.info(f"Creating index '{index_name}'...")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine"
            )
            logger.info(f"Index '{index_name}' created")
        
        # Connect to the index
        index = pc.Index(index_name)
        
        # Create data directory if it doesn't exist
        os.makedirs(pdf_dir, exist_ok=True)
        
        # Get list of PDF files
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return
        
        # Get already processed files
        processed_files = get_already_processed_files(index, namespace)
        logger.info(f"Found {len(processed_files)} already processed files in Pinecone")
        
        # Filter out already processed files
        remaining_files = []
        for pdf_file in pdf_files:
            filename = os.path.splitext(os.path.basename(pdf_file))[0]
            if filename not in processed_files:
                remaining_files.append(pdf_file)
        
        logger.info(f"Found {len(remaining_files)} new PDF files to process")
        
        if not remaining_files:
            logger.info("No new PDF files to process")
            return
        
        # Sort files by size (smallest first) to optimize memory usage
        remaining_files.sort(key=lambda f: os.path.getsize(os.path.join(pdf_dir, f)))
        
        # Process files in parallel with limited workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for pdf_file in remaining_files:
                future = executor.submit(
                    process_pdf_file,
                    os.path.join(pdf_dir, pdf_file),
                    pdf_dir,
                    chunk_size,
                    index,
                    namespace
                )
                futures.append(future)
            
            # Wait for all futures to complete with progress tracking
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing PDFs"):
                try:
                    result = future.result()
                    # Force garbage collection after each file
                    gc.collect()
                except Exception as e:
                    logger.error(f"Error in PDF processing: {e}")
        
        logger.info("PDF processing completed")
    
    except Exception as e:
        logger.error(f"Error in PDF processing: {e}")
    finally:
        # Final garbage collection
        gc.collect()

if __name__ == "__main__":
    print("=== M3 Memory-Optimized PDF Processor ===")
    print("Optimized for Mac M3 with 24GB RAM")
    
    # M3-optimized parameters
    CHUNK_SIZE = 2000  # Smaller chunks to manage memory
    MAX_WORKERS = 2    # Limit parallel processing to prevent memory issues
    
    process_pdfs_with_memory_optimization(
        chunk_size=CHUNK_SIZE,
        max_workers=MAX_WORKERS
    )
