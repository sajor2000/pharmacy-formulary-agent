"""
Test Pinecone connection with different approaches
"""
import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone
import requests

# Load environment variables
load_dotenv()

# Get API key and environment from .env
api_key = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENVIRONMENT')

print(f"Testing Pinecone connection")
print(f"API Key (first 5 chars): {api_key[:5]}...")
print(f"Environment: {pinecone_env}")
print("-" * 50)

# Method 1: Standard Pinecone client
print("\n1. Testing standard Pinecone client")
try:
    pc = Pinecone(api_key=api_key)
    indexes = pc.list_indexes()
    print(f"Success! Found {len(indexes)} indexes:")
    for idx in indexes:
        print(f"  - {idx.name}")
except Exception as e:
    print(f"Error with standard client: {e}")

# Method 2: Direct connection with host URL
print("\n2. Testing direct connection with host URL")
try:
    pc = Pinecone(api_key=api_key)
    # Use the environment URL directly as host
    host = pinecone_env
    print(f"Connecting to host: {host}")
    index = pc.Index(host=host)
    stats = index.describe_index_stats()
    print(f"Success! Index stats: {stats.total_vector_count} vectors")
except Exception as e:
    print(f"Error with direct host connection: {e}")

# Method 3: REST API direct call
print("\n3. Testing REST API direct call")
try:
    headers = {
        "Api-Key": api_key,
        "Content-Type": "application/json"
    }
    
    # Extract the index name from the URL
    # URL format: https://indexname-xxxxx.svc.environment.pinecone.io
    host_parts = pinecone_env.split('.')
    if len(host_parts) > 0:
        domain_parts = host_parts[0].split('/')
        if len(domain_parts) > 2:
            index_name = domain_parts[2].split('-')[0]
            print(f"Extracted index name: {index_name}")
            
            # Try to get index stats via REST API
            stats_url = f"{pinecone_env}/describe_index_stats"
            print(f"Calling: {stats_url}")
            response = requests.post(stats_url, headers=headers, json={})
            
            if response.status_code == 200:
                print(f"Success! Response: {response.json()}")
            else:
                print(f"API call failed with status {response.status_code}: {response.text}")
        else:
            print("Could not extract index name from URL")
    else:
        print("Invalid URL format")
except Exception as e:
    print(f"Error with REST API call: {e}")

# Method 4: Try with environment variable as region
print("\n4. Testing with environment as region")
try:
    # Extract region from URL if possible
    region = "us-east-1"  # Default
    if "aped-4627" in pinecone_env:
        region = "us-east-1"  # This appears to be an AWS us-east-1 endpoint
    
    print(f"Using region: {region}")
    pc = Pinecone(api_key=api_key)
    
    # Try to connect to the index by name
    index_name = "newformrag"
    print(f"Connecting to index: {index_name}")
    
    # This will use the API to resolve the host
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    print(f"Success! Index stats: {stats.total_vector_count} vectors")
except Exception as e:
    print(f"Error with region approach: {e}")

print("\nTroubleshooting completed!")
