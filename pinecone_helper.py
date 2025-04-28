#!/usr/bin/env python3
"""
Pinecone Helper Functions for Pharmacy Formulary Agent
-----------------------------------------------------
Helper functions for working with Pinecone v6.0.2
"""

import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec, PodSpec

# Load environment variables
load_dotenv()

# Get API keys from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

def initialize_pinecone(index_name="newformrag"):
    """Initialize Pinecone with proper configuration for v6.0.2"""
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if the environment URL is a full URL or just a region name
        host = None
        if PINECONE_ENVIRONMENT and PINECONE_ENVIRONMENT.startswith('http'):
            host = PINECONE_ENVIRONMENT
            print(f"Using host URL directly: {host}")
        
        # Try to connect to the existing index first
        try:
            print(f"Trying to connect to existing index: {index_name}")
            if host:
                index = pc.Index(index_name, host=host)
            else:
                index = pc.Index(index_name)
            
            # Test the connection
            stats = index.describe_index_stats()
            print(f"Successfully connected to existing index: {index_name}")
            print(f"Index stats: {stats}")
            return index, pc
        except Exception as e1:
            print(f"Could not connect to existing index: {e1}")
        
        # If we can't connect to the existing index, check if we need to create it
        indexes = [index.name for index in pc.list_indexes()]
        if index_name not in indexes:
            print(f"Creating new Pinecone index: {index_name}")
            
            # Try different approaches for creating the index
            try:
                # First try with no spec (simplest approach)
                pc.create_index(
                    name=index_name,
                    dimension=3072,
                    metric="cosine"
                )
            except Exception as e_simple:
                print(f"Simple index creation failed: {e_simple}")
                try:
                    # Try with ServerlessSpec
                    pc.create_index(
                        name=index_name,
                        dimension=3072,
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud="gcp",  # Try GCP instead of AWS
                            region="us-central1"
                        )
                    )
                except Exception as e_serverless:
                    print(f"Serverless index creation failed: {e_serverless}")
                    # Try with PodSpec as last resort
                    pc.create_index(
                        name=index_name,
                        dimension=3072,
                        metric="cosine",
                        spec=PodSpec(
                            environment=PINECONE_ENVIRONMENT
                        )
                    )
            
            # Wait for index to be ready
            print("Waiting for index to be ready...")
            time.sleep(10)
        
        # Get the index
        index = pc.Index(index_name)
        return index, pc
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        # Try direct connection using the environment URL
        try:
            print("Trying direct connection using environment URL...")
            # Use the environment URL directly
            if PINECONE_ENVIRONMENT and PINECONE_ENVIRONMENT.startswith('http'):
                host = PINECONE_ENVIRONMENT
                print(f"Using host: {host}")
                pc = Pinecone(api_key=PINECONE_API_KEY)
                index = pc.Index(index_name, host=host)
                return index, pc
        except Exception as e3:
            print(f"All Pinecone initialization approaches failed: {e3}")
            return None, None

def reset_pinecone_index(index_name="pharmacy-formulary"):
    """Reset Pinecone index by deleting and recreating it"""
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists
        indexes = [index.name for index in pc.list_indexes()]
        if index_name in indexes:
            print(f"Deleting existing index '{index_name}'...")
            pc.delete_index(index_name)
            
            # Wait for deletion to complete
            print("Waiting for index deletion to complete...")
            time.sleep(10)
        
        # Create a new index
        print(f"Creating new index '{index_name}'...")
        try:
            # First try ServerlessSpec
            pc.create_index(
                name=index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                )
            )
        except Exception as e:
            print(f"Error creating index with ServerlessSpec: {e}")
            # Try PodSpec as fallback
            pc.create_index(
                name=index_name,
                dimension=1024,
                metric="cosine",
                spec=PodSpec(
                    environment=PINECONE_ENVIRONMENT
                )
            )
        
        # Wait for index to be ready
        print("Waiting for index to be ready...")
        time.sleep(10)
        
        print("Pinecone index reset successfully!")
        return True
    except Exception as e:
        print(f"Error resetting Pinecone index: {e}")
        return False

if __name__ == "__main__":
    # Test Pinecone initialization
    index, pc = initialize_pinecone()
    if index:
        print("Pinecone initialization successful!")
        stats = index.describe_index_stats()
        print(f"Index stats: {stats}")
    else:
        print("Pinecone initialization failed!")
