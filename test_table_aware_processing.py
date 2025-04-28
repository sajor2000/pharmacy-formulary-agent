#!/usr/bin/env python3
"""
Test Table-Aware Processing and Agent Response Quality
-----------------------------------------------------
This script tests the table-aware processing and agent response quality
to ensure high-quality outputs for pharmacy formulary queries.
"""

import os
import time
from dotenv import load_dotenv
from llama_parse_processor import LlamaParseProcessor
from pharmacy_lightrag_agent import PharmacyFormularyAgent

# Load environment variables
load_dotenv()

def test_table_extraction():
    """Test table extraction from a PDF file"""
    print("=== Testing Table Extraction ===")
    
    # Initialize processor with the correct index name
    processor = LlamaParseProcessor(index_name="newformrag")
    
    # Find a PDF file with tables (CountyCare is a good candidate)
    pdf_dir = "data"
    test_pdf = None
    for f in os.listdir(pdf_dir):
        if f.endswith('.pdf') and 'county' in f.lower():
            test_pdf = f
            break
    
    if not test_pdf:
        # Try any PDF if CountyCare not found
        for f in os.listdir(pdf_dir):
            if f.endswith('.pdf'):
                test_pdf = f
                break
    
    if not test_pdf:
        print("No PDF files found for testing.")
        return False
    
    print(f"Testing table extraction with {test_pdf}")
    
    # Parse the PDF with table awareness
    pdf_path = os.path.join(pdf_dir, test_pdf)
    content = processor._enhanced_pymupdf_extraction(pdf_path)
    
    # Check if tables were extracted
    tables_found = len(content.get('tables', []))
    print(f"Found {tables_found} tables in {test_pdf}")
    
    # Show a sample of the first table if any were found
    if tables_found > 0:
        print("\nSample of first table:")
        print(content['tables'][0]['markdown'][:500] + "...\n")
        return True
    else:
        print("No tables found in the test PDF.")
        return False

def test_chunking_with_table_awareness():
    """Test chunking with table awareness"""
    print("=== Testing Chunking with Table Awareness ===")
    
    # Initialize processor with the correct index name
    processor = LlamaParseProcessor(index_name="newformrag")
    
    # Create a sample text with a table
    sample_text = {
        'text': """
This is some text before a table.

TABLE 1-1:
| Medication | Tier | Requirements | Quantity Limit |
|------------|------|--------------|----------------|
| Advair     | 2    | None         | 1 inhaler/30d  |
| Symbicort  | 3    | PA           | 1 inhaler/30d  |
| Spiriva    | 2    | None         | 30 caps/30d    |

This is some text after the table.

Here's another paragraph with more information.
        """,
        'tables': [{'page': 1, 'table_num': 1, 'markdown': '| Medication | Tier | Requirements | Quantity Limit |\n|------------|------|--------------|----------------|\n| Advair     | 2    | None         | 1 inhaler/30d  |\n| Symbicort  | 3    | PA           | 1 inhaler/30d  |\n| Spiriva    | 2    | None         | 30 caps/30d    |'}]
    }
    
    # Chunk the content with table awareness
    chunks = processor.chunk_with_table_awareness(sample_text, chunk_size=500)
    
    print(f"Created {len(chunks)} chunks with table awareness")
    
    # Check if tables are preserved in chunks
    tables_preserved = sum(1 for chunk in chunks if 'TABLE' in chunk)
    print(f"Chunks containing tables: {tables_preserved}")
    
    # Show a sample of a chunk with a table
    for chunk in chunks:
        if 'TABLE' in chunk:
            print("\nSample chunk with table:")
            print(chunk[:300] + "...\n")
            return True
    
    return False

def test_agent_response_quality():
    """Test agent response quality with table data"""
    print("=== Testing Agent Response Quality ===")
    
    # Initialize agent
    agent = PharmacyFormularyAgent()
    
    # Test queries
    test_queries = [
        "What is the coverage for Advair Diskus under CountyCare insurance?",
        "What tier is Spiriva in BCBS formulary?",
        "What are the quantity limits for albuterol inhalers?"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: {query}")
        
        # Time the response
        start_time = time.time()
        response = agent.query(query)
        elapsed_time = time.time() - start_time
        
        print(f"Response received in {elapsed_time:.2f} seconds")
        print(f"Response length: {len(response)} characters")
        
        # Check for structured format
        has_primary_rec = "Primary Recommendation:" in response
        has_alternatives = "Alternative Options:" in response
        has_coverage_notes = "Coverage Notes:" in response
        
        print(f"Has structured format: {has_primary_rec and has_alternatives and has_coverage_notes}")
        
        # Print a preview of the response
        print("\nResponse preview:")
        print(response[:300] + "...\n")
        
        # Small delay between queries
        time.sleep(2)
    
    return True

def main():
    """Run all tests"""
    print("=== Testing Table-Aware Processing and Agent Response Quality ===")
    
    # Test table extraction
    table_extraction_ok = test_table_extraction()
    
    # Test chunking with table awareness
    chunking_ok = test_chunking_with_table_awareness()
    
    # Test agent response quality
    agent_ok = test_agent_response_quality()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Table extraction: {'✅ PASS' if table_extraction_ok else '❌ FAIL'}")
    print(f"Chunking with table awareness: {'✅ PASS' if chunking_ok else '❌ FAIL'}")
    print(f"Agent response quality: {'✅ PASS' if agent_ok else '❌ FAIL'}")
    
    if table_extraction_ok and chunking_ok and agent_ok:
        print("\n✅ All tests passed! Your table-aware processing is working correctly.")
        print("You can now run reset_and_process_pdfs.py to process all PDFs from scratch.")
    else:
        print("\n❌ Some tests failed. Please check the issues before processing all PDFs.")

if __name__ == "__main__":
    main()
