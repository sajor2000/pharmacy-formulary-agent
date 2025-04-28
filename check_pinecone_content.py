"""
Check what content is available in the Pinecone index
"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import json

# Load environment variables
load_dotenv()

def main():
    """Check what content is available in the Pinecone index"""
    print("Checking Pinecone index content...")
    
    # Initialize Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    
    # Connect to the index
    index = pc.Index("finalpharm")
    
    # Get index stats
    stats = index.describe_index_stats()
    print(f"Total vectors in index: {stats.total_vector_count}")
    print(f"Dimension: {stats.dimension}")
    
    # Create a simple query to find CountyCare content
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create embedding for the query
    response = openai_client.embeddings.create(
        input="CountyCare formulary coverage",
        model="text-embedding-3-large"
    )
    embedding = response.data[0].embedding
    
    # Query Pinecone
    results = index.query(
        vector=embedding,
        top_k=5,
        include_metadata=True
    )
    
    # Print the results
    print("\nTop 5 results for 'CountyCare formulary coverage':")
    for i, match in enumerate(results.matches):
        print(f"\nResult {i+1} (Score: {match.score:.4f}):")
        if hasattr(match, 'metadata') and match.metadata:
            if 'source' in match.metadata:
                print(f"Source: {match.metadata['source']}")
            if 'insurance_provider' in match.metadata:
                print(f"Insurance: {match.metadata['insurance_provider']}")
            if 'has_table' in match.metadata:
                print(f"Has table: {match.metadata['has_table']}")
            if 'text' in match.metadata:
                # Truncate text for display
                text = match.metadata['text']
                if len(text) > 200:
                    text = text[:200] + "..."
                print(f"Text snippet: {text}")
        else:
            print("No metadata available")
    
    # Try a more specific query for ICS-LABA inhalers
    print("\n\nSearching for ICS-LABA inhalers in CountyCare...")
    response = openai_client.embeddings.create(
        input="CountyCare ICS-LABA inhalers Advair Symbicort Breo",
        model="text-embedding-3-large"
    )
    embedding = response.data[0].embedding
    
    # Query Pinecone
    results = index.query(
        vector=embedding,
        top_k=5,
        include_metadata=True
    )
    
    # Print the results
    print("\nTop 5 results for 'CountyCare ICS-LABA inhalers':")
    for i, match in enumerate(results.matches):
        print(f"\nResult {i+1} (Score: {match.score:.4f}):")
        if hasattr(match, 'metadata') and match.metadata:
            if 'source' in match.metadata:
                print(f"Source: {match.metadata['source']}")
            if 'insurance_provider' in match.metadata:
                print(f"Insurance: {match.metadata['insurance_provider']}")
            if 'has_table' in match.metadata:
                print(f"Has table: {match.metadata['has_table']}")
            if 'text' in match.metadata:
                # Truncate text for display
                text = match.metadata['text']
                if len(text) > 200:
                    text = text[:200] + "..."
                print(f"Text snippet: {text}")
        else:
            print("No metadata available")

if __name__ == "__main__":
    main()
