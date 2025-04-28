"""
Check Pinecone indexes and their dimensions
"""
from pinecone import Pinecone
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone
api_key = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=api_key)

# List indexes
indexes = pc.list_indexes()
print('Available Pinecone indexes:')
for idx in indexes:
    print(f'- {idx.name}')
    
    # Try to get index details
    try:
        index = pc.Index(idx.name)
        stats = index.describe_index_stats()
        print(f'  Dimension: {stats.dimension}')
        print(f'  Vector count: {stats.total_vector_count}')
        print(f'  Namespaces: {list(stats.namespaces.keys()) if stats.namespaces else "None"}')
    except Exception as e:
        print(f'  Error getting details: {e}')
    print()
