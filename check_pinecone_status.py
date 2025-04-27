#!/usr/bin/env python3
"""
Check Pinecone Status
---------------------
Script to check the status of the Pinecone index and which files have been processed.
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

def main():
    """Check Pinecone index status and processed files"""
    try:
        # Initialize Pinecone
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            print("ERROR: PINECONE_API_KEY not found in environment variables")
            return
        
        # Initialize Pinecone with new API
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Check if 'form' index exists
        index_name = "form"
        
        # Get list of indexes
        existing_indexes = pc.list_indexes()
        
        if index_name not in existing_indexes.names():
            print(f"ERROR: Index '{index_name}' does not exist")
            return
        
        # Connect to the index
        index = pc.Index(index_name)
        
        # Get index stats
        stats = index.describe_index_stats()
        
        print("\n=== Pinecone Index Stats ===")
        print(f"Total vector count: {stats.total_vector_count}")
        
        print("\nNamespaces:")
        for ns, ns_stats in stats.namespaces.items():
            print(f"  {ns}: {ns_stats.vector_count} vectors")
        
        # Get list of processed files
        namespace = "formulary"
        if namespace in stats.namespaces and stats.namespaces[namespace].vector_count > 0:
            # Query Pinecone for existing vectors to get source filenames
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
            
            print("\nProcessed Files:")
            for i, filename in enumerate(sorted(processed_files), 1):
                print(f"  {i}. {filename}")
            
            # Get list of PDF files in data directory
            pdf_dir = "data"
            if os.path.exists(pdf_dir):
                # Get actual PDF files with extensions
                pdf_files_with_ext = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
                # Get filenames without extensions for comparison
                pdf_files_no_ext = [os.path.splitext(f)[0] for f in pdf_files_with_ext]
                
                # Convert processed files to remove .pdf if present
                processed_files_clean = set()
                for filename in processed_files:
                    if filename.lower().endswith('.pdf'):
                        processed_files_clean.add(filename[:-4])
                    else:
                        processed_files_clean.add(filename)
                
                # Find unprocessed files (comparing without extensions)
                unprocessed_files_no_ext = set(pdf_files_no_ext) - processed_files_clean
                
                # Map back to filenames with extensions for display
                unprocessed_files = []
                for f in pdf_files_with_ext:
                    if os.path.splitext(f)[0] in unprocessed_files_no_ext:
                        unprocessed_files.append(f)
                
                print("\nUnprocessed Files:")
                if unprocessed_files:
                    for i, filename in enumerate(sorted(unprocessed_files), 1):
                        print(f"  {i}. {filename}")
                else:
                    print("  All files have been processed!")
            else:
                print(f"\nERROR: Directory '{pdf_dir}' does not exist")
        else:
            print(f"\nNamespace '{namespace}' is empty or does not exist")
            print("No files have been processed yet")
    
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
