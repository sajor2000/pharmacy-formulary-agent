#!/usr/bin/env python3
"""
Test script to verify Pinecone connection using environment variables
"""

import os
import logging
from dotenv import load_dotenv
from pinecone import Pinecone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_pinecone_connection():
    """Test connection to Pinecone using environment variables"""
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        logger.error("❌ PINECONE_API_KEY is not set")
        return False
    
    logger.info(f"Using API key: {api_key[:10]}...{api_key[-5:]}")
    
    try:
        # Initialize Pinecone client
        logger.info("Initializing Pinecone client...")
        pc = Pinecone(api_key=api_key)
        
        # List indexes
        logger.info("Listing Pinecone indexes...")
        indexes = pc.list_indexes()
        logger.info(f"✅ Successfully connected to Pinecone. Available indexes: {indexes}")
        
        # Check if finalpharm index exists
        index_names = [index.name for index in indexes]
        logger.info(f"Index names: {index_names}")
        
        if "finalpharm" in index_names:
            logger.info("✅ finalpharm index exists")
            
            # Connect to the index
            logger.info("Connecting to finalpharm index...")
            index = pc.Index("finalpharm")
            
            # Get stats
            logger.info("Getting index stats...")
            stats = index.describe_index_stats()
            logger.info(f"✅ Connected to finalpharm index. Stats: {stats}")
            
            # Try a simple query
            logger.info("Testing a simple query...")
            results = index.query(
                vector=[0.1] * 3072,  # Create a dummy vector of the right dimension
                top_k=1,
                include_metadata=True
            )
            logger.info(f"✅ Query successful. Results: {results}")
            
            return True
        else:
            logger.error(f"❌ finalpharm index not found. Available indexes: {index_names}")
            return False
    
    except Exception as e:
        logger.error(f"❌ Error connecting to Pinecone: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting Pinecone connection test...")
    success = test_pinecone_connection()
    if success:
        logger.info("✅ All Pinecone tests passed!")
    else:
        logger.error("❌ Pinecone tests failed!")
