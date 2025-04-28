#!/usr/bin/env python3
"""
Process CountyCare PDF with Table-Aware Processing
-------------------------------------------------
This script processes the CountyCare formulary PDF using the enhanced
LlamaParse processor that preserves table structure for better RAG results.
"""

import os
import sys
from dotenv import load_dotenv
from llama_parse_processor import LlamaParseProcessor

# Load environment variables
load_dotenv()

def main():
    print("=== Processing CountyCare Formulary with Table-Aware Processing ===")
    
    # Initialize the LlamaParse processor
    processor = LlamaParseProcessor()
    
    # Path to CountyCare PDF
    countycare_pdf = os.path.join("data", "CountyCare.0209.1Q2024.pdf")
    
    # Check if file exists
    if not os.path.exists(countycare_pdf):
        print(f"Error: CountyCare PDF not found at {countycare_pdf}")
        return
    
    print(f"Processing {countycare_pdf} with table structure preservation...")
    
    # Process the PDF with table awareness
    chunks = processor.process_pdf(countycare_pdf, namespace="formulary")
    
    print(f"Successfully processed CountyCare formulary with {chunks} chunks")
    print("Tables have been preserved in their original structure for better retrieval")

if __name__ == "__main__":
    main()
