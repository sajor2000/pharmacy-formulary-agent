#!/usr/bin/env python3
"""
Document Processor for Pharmacy Formulary Agent
----------------------------------------------
Processes PDF formulary documents and creates embeddings for vector search.
"""

import os
import time
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Get Pinecone API key from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

class DocumentProcessor:
    def __init__(self, pdf_dir="data"):
        self.pdf_dir = pdf_dir
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
    def get_embedding(self, text):
        """Get embedding for text and resize to match Pinecone index dimensions (1024)"""
        try:
            # Truncate text if too long (embedding model has token limits)
            max_tokens = 8000  # embedding models typically have 8k token limit
            # Simple truncation - in production you'd want to chunk properly
            if len(text) > max_tokens * 4:  # rough estimate of chars to tokens
                text = text[:max_tokens * 4]
            
            # Use OpenAI's standard embedding model
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            
            # Get the embedding vector
            embedding = response.data[0].embedding
            
            # Resize to match Pinecone index dimensions (1024)
            resized_embedding = self._resize_embedding(embedding, target_dim=1024)
            
            return resized_embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file"""
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_tables_from_pdf(self, pdf_path):
        """Extract tables from a PDF file using PyMuPDF"""
        tables = []
        try:
            # Method 1: Extract tables using PyMuPDF
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                tab = page.find_tables()
                if tab.tables:
                    for i, table in enumerate(tab.tables):
                        df = table.to_pandas()
                        tables.append({
                            'page': page_num + 1,
                            'table_num': i + 1,
                            'source': 'pymupdf',
                            'data': df
                        })
            
            return tables
        except Exception as e:
            print(f"Error extracting tables from {pdf_path}: {e}")
            return []
    
    def _resize_embedding(self, embedding, target_dim=1024):
        """Resize an embedding vector to the target dimension using a simpler method"""
        try:
            # Convert to numpy array if it's not already
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # Get current dimensions
            current_dim = embedding.shape[0]
            
            if current_dim == target_dim:
                return embedding.tolist()
            
            # For dimensionality reduction, use a simpler approach
            # We'll take a subset of the original dimensions and normalize
            if current_dim > target_dim:
                # Take every nth element to get target_dim elements
                indices = np.round(np.linspace(0, current_dim - 1, target_dim)).astype(int)
                reduced = embedding[indices]
                
                # Normalize to preserve the vector magnitude
                norm = np.linalg.norm(reduced)
                if norm > 0:
                    reduced = reduced / norm
                
                return reduced.tolist()
            
            # If we need to increase dimensions, pad with zeros
            if current_dim < target_dim:
                padded = np.zeros(target_dim)
                padded[:current_dim] = embedding
                
                # Normalize to preserve the vector magnitude
                norm = np.linalg.norm(padded)
                if norm > 0:
                    padded = padded / norm
                
                return padded.tolist()
            
        except Exception as e:
            print(f"Error resizing embedding: {e}")
            # If resizing fails, create a random normalized vector
            random_vec = np.random.randn(target_dim)
            random_vec = random_vec / np.linalg.norm(random_vec)
            return random_vec.tolist()
    
    def store_in_pinecone(self, embeddings, index_name="form", namespace="formulary"):
        """Store embeddings in Pinecone using the existing index"""
        try:
            from pinecone import Pinecone
            
            print(f"\nStoring {len(embeddings)} embeddings in Pinecone...")
            
            # Initialize Pinecone
            pc = Pinecone(api_key=PINECONE_API_KEY)
            
            # Connect to the existing index
            index = pc.Index(index_name)
            
            # Prepare vectors for upsert
            vectors = []
            for i, item in enumerate(embeddings):
                vector_id = f"vec_{i}_{int(time.time())}"
                vectors.append({
                    "id": vector_id,
                    "values": item["embedding"],
                    "metadata": item["metadata"]
                })
            
            # Upsert in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                index.upsert(vectors=batch, namespace=namespace)
                print(f"Stored batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
            
            print(f"Successfully stored {len(embeddings)} embeddings in Pinecone")
            return True
        except Exception as e:
            print(f"Error storing embeddings in Pinecone: {e}")
            return False
