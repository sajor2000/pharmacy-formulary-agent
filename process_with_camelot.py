"""
Process all pharmacy formulary PDFs using Camelot

This script processes all PDF files in the data directory using the
Camelot processor, which provides excellent table extraction capabilities
without requiring external dependencies like Poppler.
"""

import os
import time
import gc
import psutil
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
from camelot_processor import CamelotProcessor

# Load environment variables
load_dotenv()

def main():
    """Main function to process all PDFs with Camelot"""
    print("=== Process All PDFs with Camelot ===")
    print(f"Time started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check memory and CPU resources
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    cpu_cores = psutil.cpu_count(logical=False)
    
    print(f"Available memory: {available_gb:.2f} GB")
    print(f"CPU cores: {cpu_cores}")
    
    # Determine optimal batch size based on available memory
    optimal_batch_size = 1
    if available_gb > 8:
        optimal_batch_size = 3
    elif available_gb > 4:
        optimal_batch_size = 2
    
    print(f"Optimal batch size: {optimal_batch_size} PDFs")
    
    # Initialize the processor
    processor = CamelotProcessor(index_name="finalpharm")
    
    # Process all PDFs in batches
    pdf_dir = "data"
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) 
                if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(pdf_dir, f))]
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return 0
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process PDFs in batches
    total_chunks = processor.process_pdfs_in_batches(
        pdf_dir=pdf_dir,
        batch_size=optimal_batch_size,
        delay_between_batches=15  # 15 seconds between batches
    )
    
    print(f"Total chunks processed: {total_chunks}")
    print(f"Time finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return total_chunks

if __name__ == "__main__":
    main()
