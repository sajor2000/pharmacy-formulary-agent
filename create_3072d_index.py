"""
Create a new Pinecone index with 3072 dimensions for text-embedding-3-large
"""
from pinecone import Pinecone, ServerlessSpec
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone
api_key = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=api_key)

# New index name
INDEX_NAME = "form3072"

# Check if index already exists
existing_indexes = [idx.name for idx in pc.list_indexes()]
if INDEX_NAME in existing_indexes:
    print(f"Index '{INDEX_NAME}' already exists. Deleting it first...")
    pc.delete_index(INDEX_NAME)
    # Wait for deletion to complete
    time.sleep(10)

# Create new serverless index with 3072 dimensions
print(f"Creating new index '{INDEX_NAME}' with 3072 dimensions...")
try:
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Successfully created index '{INDEX_NAME}'")
except Exception as e:
    print(f"Error creating index: {e}")

# List all indexes to confirm
indexes = pc.list_indexes()
print("\nAvailable Pinecone indexes:")
for idx in indexes:
    print(f"- {idx.name}")
    
    # Try to get index details
    try:
        index = pc.Index(idx.name)
        stats = index.describe_index_stats()
        print(f"  Dimension: {stats.dimension}")
        print(f"  Vector count: {stats.total_vector_count}")
    except Exception as e:
        print(f"  Error getting details: {e}")
    print()
