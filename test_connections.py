#!/usr/bin/env python3
"""
Test script to verify API connections for the Pharmacy Formulary Agent
"""

import os
import time
from dotenv import load_dotenv
import openai
from pinecone import Pinecone

# Load environment variables
load_dotenv()

def test_openai_connection():
    """Test connection to OpenAI API"""
    try:
        # Get OpenAI API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return False, "OpenAI API key not found in .env file"
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Test with a simple embedding request
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input="This is a test"
        )
        
        # Check if we got a valid response
        if hasattr(response, 'data') and len(response.data) > 0:
            embedding_length = len(response.data[0].embedding)
            return True, f"OpenAI connection successful. Embedding dimension: {embedding_length}"
        else:
            return False, "OpenAI connection failed: Invalid response format"
    
    except Exception as e:
        return False, f"OpenAI connection failed: {str(e)}"

def test_pinecone_connection():
    """Test connection to Pinecone API"""
    try:
        # Get Pinecone API key from environment
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            return False, "Pinecone API key not found in .env file"
        
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        
        # List indexes to verify connection
        indexes = pc.list_indexes()
        index_names = [index.name for index in indexes]
        
        # Check if 'form' index exists
        form_index_exists = "form" in index_names
        
        if form_index_exists:
            # Try to connect to the 'form' index
            try:
                index = pc.Index("form")
                stats = index.describe_index_stats()
                namespace_count = len(stats.namespaces) if hasattr(stats, 'namespaces') else 0
                vector_count = stats.total_vector_count if hasattr(stats, 'total_vector_count') else 0
                return True, f"Pinecone connection successful. 'form' index exists with {vector_count} vectors across {namespace_count} namespaces."
            except Exception as e:
                return False, f"Pinecone connection successful but error accessing 'form' index: {str(e)}"
        else:
            return False, "Pinecone connection successful but 'form' index does not exist. Please create it with 1024 dimensions."
    
    except Exception as e:
        return False, f"Pinecone connection failed: {str(e)}"

if __name__ == "__main__":
    print("=== Testing API Connections for Pharmacy Formulary Agent ===\n")
    
    # Test OpenAI connection
    print("Testing OpenAI connection...")
    openai_success, openai_message = test_openai_connection()
    print(f"{'✅' if openai_success else '❌'} {openai_message}\n")
    
    # Test Pinecone connection
    print("Testing Pinecone connection...")
    pinecone_success, pinecone_message = test_pinecone_connection()
    print(f"{'✅' if pinecone_success else '❌'} {pinecone_message}\n")
    
    # Overall status
    if openai_success and pinecone_success:
        print("✅ All connections successful! You can now run the processing scripts.")
    else:
        print("❌ Some connections failed. Please fix the issues before running the processing scripts.")
