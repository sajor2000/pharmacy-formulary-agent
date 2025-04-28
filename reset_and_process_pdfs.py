#!/usr/bin/env python3
"""
Reset Pinecone Index and Process All PDFs from Scratch
-----------------------------------------------------
This script:
1. Clears all data from the Pinecone index
2. Processes all PDFs using the table-aware LlamaParse processor
3. Optimizes memory usage for M3 Mac (24GB RAM)
4. Uses parallel processing where appropriate
"""

import os
import time
import gc
import psutil
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

# Import our custom processors and helpers
from llama_parse_processor import LlamaParseProcessor
from pinecone_helper import reset_pinecone_index, initialize_pinecone

# Load environment variables
load_dotenv()

# Index name for Pinecone
INDEX_NAME = "form"  # Using the existing Pinecone index

def clear_pinecone_index():
    """Clear all data from the Pinecone index"""
    try:
        print("=== Clearing Pinecone Index ===")
        # Connect to the existing index
        index, pc = initialize_pinecone(INDEX_NAME)
        if index:
            # Delete all vectors instead of recreating the index
            print(f"Connected to existing index '{INDEX_NAME}'. Deleting all vectors...")
            try:
                # Get all namespaces
                stats = index.describe_index_stats()
                namespaces = stats.get('namespaces', {})
                
                # Delete vectors in each namespace
                for namespace in namespaces:
                    print(f"Deleting vectors in namespace: {namespace}")
                    index.delete(delete_all=True, namespace=namespace)
                
                # Also delete vectors in default namespace
                index.delete(delete_all=True)
                print("All vectors deleted successfully.")
                return True
            except Exception as e:
                print(f"Error deleting vectors: {e}")
                # Continue anyway
                return True
        else:
            print("Could not connect to Pinecone index. Aborting.")
            return False
    except Exception as e:
        print(f"Error clearing Pinecone index: {e}")
        return False

def get_optimal_batch_size_and_workers():
    """Determine optimal batch size and worker count based on available memory and CPU cores"""
    # Get available memory in GB
    available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    
    # Reserve 4GB for system and other processes
    usable_memory = max(1, available_memory - 4)
    
    # Estimate memory per PDF (conservative estimate)
    memory_per_pdf = 1.5  # GB
    
    # Calculate batch size (at least 1, at most 4)
    batch_size = max(1, min(4, int(usable_memory / memory_per_pdf)))
    
    print(f"Available memory: {available_memory:.2f} GB")
    print(f"Optimal batch size: {batch_size} PDFs")
    
    return batch_size

def get_optimal_worker_count():
    """Determine optimal worker count based on CPU cores"""
    # Get CPU core count
    cpu_count = os.cpu_count()
    
    # For M3 Mac, use at most 75% of cores to avoid overheating
    worker_count = max(1, int(cpu_count * 0.75))
    
    print(f"CPU cores: {cpu_count}")
    print(f"Optimal worker count: {worker_count}")
    
    return worker_count

def process_pdf_batch(pdf_files, processor, namespace="formulary"):
    """Process a batch of PDF files"""
    total_chunks = 0
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join("data", pdf_file)
        print(f"Processing {pdf_file}...")
        chunks = processor.process_pdf(pdf_path, namespace)
        total_chunks += chunks
        
        # Force garbage collection after each file
        gc.collect()
        
        # Small delay between files
        time.sleep(1)
    
    return total_chunks

def process_all_pdfs_with_table_awareness():
    """Process all PDFs with table awareness using optimal parallelism"""
    print("=== Processing All PDFs with Table Awareness ===")
    
    # Check if LlamaParse API key is available
    if not os.getenv("LLAMAPARSE_API_KEY") or os.getenv("LLAMAPARSE_API_KEY") == "YOUR_LLAMAPARSE_API_KEY":
        print("⚠️ WARNING: LlamaParse API key not found or not set in .env file.")
        print("For best table extraction, please add your LlamaParse API key.")
        print("Continuing with enhanced PyMuPDF extraction as fallback...")
    else:
        print("✅ LlamaParse API key found. Will use LlamaParse for optimal table extraction.")
    
    # Get all PDF files in the data directory
    pdf_dir = "data"
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in the data directory.")
        return 0
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Get optimal batch size and worker count
    batch_size = get_optimal_batch_size_and_workers()
    worker_count = get_optimal_worker_count()
    
    # Initialize processor with the correct index name
    processor = LlamaParseProcessor(index_name=INDEX_NAME)
    
    # Sort PDFs by size (process largest files first)
    pdf_files_with_size = [(f, os.path.getsize(os.path.join(pdf_dir, f))) for f in pdf_files]
    pdf_files_with_size.sort(key=lambda x: x[1], reverse=True)
    sorted_pdf_files = [f[0] for f in pdf_files_with_size]
    
    # Create batches
    batches = [sorted_pdf_files[i:i+batch_size] for i in range(0, len(sorted_pdf_files), batch_size)]
    
    print(f"Processing {len(batches)} batches with up to {batch_size} PDFs per batch")
    
    # Process batches sequentially (better for memory management)
    total_chunks = 0
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)}: {batch}")
        chunks = process_pdf_batch(batch, processor)
        total_chunks += chunks
        
        # Force garbage collection between batches
        gc.collect()
        
        # Delay between batches to allow memory to stabilize
        if i < len(batches) - 1:
            print(f"Waiting 10 seconds before processing next batch...")
            time.sleep(10)
    
    print(f"Successfully processed {len(pdf_files)} PDFs with {total_chunks} total chunks")
    return total_chunks

def main():
    """Main function to process all PDFs"""
    print("=== Process All PDFs with Table Awareness ===")
    print(f"Time started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Clear Pinecone index (only deletes vectors, not the index itself)
    if not clear_pinecone_index():
        print("WARNING: Could not clear index, but will try to continue processing PDFs anyway...")
    
    # Process all PDFs with table awareness
    total_chunks = process_all_pdfs_with_table_awareness()
    
    print(f"Time completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successfully processed all PDFs with {total_chunks} total chunks")
    print("Your Pinecone index is now ready with table-aware embeddings!")

if __name__ == "__main__":
    main()
