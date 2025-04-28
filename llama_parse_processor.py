#!/usr/bin/env python3
"""
LlamaParse Document Processor for Pharmacy Formulary Agent
---------------------------------------------------------
Enhanced processor that preserves table structure using LlamaParse
for better RAG performance with tabular formulary data.
"""

import os
import time
import json
import requests
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv
from tqdm import tqdm

# Import our Pinecone helper
from pinecone_helper import initialize_pinecone

# Load environment variables
load_dotenv()

# Get API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# Optional: LlamaParse API key (if you have one)
LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY", "")

class LlamaParseProcessor:
    def __init__(self, pdf_dir="data", index_name="pharmacy-formulary"):
        self.pdf_dir = pdf_dir
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.index_name = index_name
        
        # Initialize Pinecone using our helper
        self.index, _ = initialize_pinecone(self.index_name)
        if not self.index:
            print("Failed to initialize Pinecone index. Some functionality may be limited.")
        
    def get_embedding(self, text):
        """Get embedding for text and resize to match Pinecone index dimensions (3072)"""
        try:
            # Truncate text if too long (embedding model has token limits)
            max_tokens = 8000  # embedding models typically have 8k token limit
            # Simple truncation - in production you'd want to chunk properly
            if len(text) > max_tokens * 4:  # rough estimate of chars to tokens
                text = text[:max_tokens * 4]
            
            # Use OpenAI's standard embedding model
            response = self.client.embeddings.create(
                model="text-embedding-3-large",  # Use the latest model which outputs 3072 dimensions
                input=text
            )
            
            # Get the embedding vector
            embedding = response.data[0].embedding
            
            # The embedding is already 3072 dimensions, no need to resize
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def _resize_embedding(self, embedding, target_dim=3072):
        """Resize an embedding vector to the target dimension"""
        try:
            # Convert to numpy array if it's not already
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # Get current dimensions
            current_dim = embedding.shape[0]
            
            if current_dim == target_dim:
                return embedding.tolist()
            
            # If current dimension is larger, use dimensionality reduction
            if current_dim > target_dim:
                # Simple approach: take every nth element
                indices = np.round(np.linspace(0, current_dim - 1, target_dim)).astype(int)
                resized = embedding[indices]
                return resized.tolist()
            
            # If current dimension is smaller, use padding
            if current_dim < target_dim:
                # Simple approach: repeat the vector
                repeats = int(np.ceil(target_dim / current_dim))
                padded = np.tile(embedding, repeats)[:target_dim]
                return padded.tolist()
                
        except Exception as e:
            print(f"Error resizing embedding: {e}")
            return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
    
    def parse_pdf_with_llamaparse(self, pdf_path):
        """
        Use LlamaParse to extract structured content from PDF, preserving tables
        
        If LlamaParse API key is not available, falls back to PyMuPDF
        but with enhanced table handling
        """
        if LLAMAPARSE_API_KEY:
            return self._parse_with_llamaparse_api(pdf_path)
        else:
            return self._enhanced_pymupdf_extraction(pdf_path)
    
    def _parse_with_llamaparse_api(self, pdf_path):
        """Parse PDF using LlamaParse API to preserve table structure"""
        try:
            print(f"Using LlamaParse API for {os.path.basename(pdf_path)} to extract tables and text...")
            # LlamaParse API endpoint
            url = "https://api.llamaparse.com/v1/parse"
            
            # Prepare the file for upload
            files = {'file': open(pdf_path, 'rb')}
            
            # Set headers with API key
            headers = {
                'Authorization': f'Bearer {LLAMAPARSE_API_KEY}'
            }
            
            # Add parameters to optimize for table extraction
            params = {
                'extract_tables': 'true',
                'mode': 'markdown',  # Get markdown output for better table formatting
                'include_metadata': 'true'  # Get additional metadata
            }
            
            # Make the API request
            response = requests.post(url, headers=headers, files=files, params=params)
            
            if response.status_code == 200:
                result = response.json()
                # Count tables found
                tables = result.get('tables', [])
                print(f"✅ LlamaParse successfully extracted {len(tables)} tables from {os.path.basename(pdf_path)}")
                
                # LlamaParse returns markdown with preserved table structure
                return {
                    'text': result.get('text', ''),
                    'markdown': result.get('markdown', ''),
                    'tables': tables
                }
            else:
                print(f"⚠️ LlamaParse API error: {response.status_code} - {response.text}")
                print("Falling back to enhanced PyMuPDF extraction...")
                # Fall back to enhanced PyMuPDF extraction
                return self._enhanced_pymupdf_extraction(pdf_path)
                
        except Exception as e:
            print(f"⚠️ Error using LlamaParse API: {e}")
            print("Falling back to enhanced PyMuPDF extraction...")
            # Fall back to enhanced PyMuPDF extraction
            return self._enhanced_pymupdf_extraction(pdf_path)
    
    def _enhanced_pymupdf_extraction(self, pdf_path):
        """Enhanced PDF extraction with better table handling using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            markdown_tables = []
            
            for page_num, page in enumerate(doc):
                # Extract regular text
                page_text = page.get_text()
                
                # Extract tables
                tab = page.find_tables()
                if tab.tables:
                    for i, table in enumerate(tab.tables):
                        try:
                            # Convert table to pandas DataFrame
                            df = table.to_pandas()
                            
                            # Convert DataFrame to markdown table
                            markdown_table = df.to_markdown(index=False)
                            
                            # Add table identifier and page number
                            table_identifier = f"\n\nTABLE {page_num+1}-{i+1}:\n{markdown_table}\n\n"
                            
                            # In newer PyMuPDF versions, we need to handle table areas differently
                            try:
                                # Try getting the table rect if available
                                if hasattr(table, 'rect'):
                                    rect = table.rect
                                    table_text = page.get_text("text", clip=rect)
                                    if table_text in page_text:
                                        page_text = page_text.replace(table_text, table_identifier)
                                    else:
                                        # If exact replacement fails, append the table
                                        page_text += table_identifier
                                else:
                                    # If rect not available, just append the table
                                    page_text += table_identifier
                            except AttributeError:
                                # If rect attribute is not available, just append the table
                                page_text += table_identifier
                        except Exception as e:
                            print(f"Error processing table {i} on page {page_num+1}: {e}")
                            # Still add a placeholder for the table
                            page_text += f"\n\nTABLE {page_num+1}-{i+1}: [Table processing error]\n\n"
                        
                        # Store the markdown table
                        markdown_tables.append({
                            'page': page_num + 1,
                            'table_num': i + 1,
                            'markdown': markdown_table,
                            'dataframe': df
                        })
                
                full_text += page_text
            
            return {
                'text': full_text,
                'markdown': full_text,  # In this case, text already has markdown tables
                'tables': markdown_tables
            }
            
        except Exception as e:
            print(f"Error in enhanced PyMuPDF extraction: {e}")
            # Fall back to basic text extraction
            return {'text': self.extract_basic_text(pdf_path), 'markdown': '', 'tables': []}
    
    def extract_basic_text(self, pdf_path):
        """Basic text extraction fallback"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            print(f"Error extracting basic text from {pdf_path}: {e}")
            return ""
    
    def chunk_with_table_awareness(self, content, chunk_size=2000):
        """
        Chunk text while preserving table structure
        This ensures tables don't get split in the middle and prioritizes formulary tables
        """
        # If content is just a string, handle it differently
        if isinstance(content, str):
            chunks = self._chunk_text(content, chunk_size)
            return chunks, 0  # No tables
        
        text = content.get('text', '')
        tables = content.get('tables', [])
        table_count = len(tables)
        
        # If no tables, use regular chunking
        if not tables:
            chunks = self._chunk_text(text, chunk_size)
            return chunks, 0  # No tables
        
        # Find table markers in the text
        chunks = []
        current_pos = 0
        
        # Look for table markers like "TABLE X-Y:"
        import re
        table_markers = re.finditer(r'TABLE \d+-\d+:', text)
        
        # Keywords that indicate a formulary table
        formulary_keywords = ['tier', 'drug', 'medication', 'formulary', 'quantity', 'limit', 
                             'prior auth', 'pa', 'authorization', 'preferred', 'non-preferred', 
                             'brand', 'generic', 'specialty', 'inhaler', 'respiratory']
        
        formulary_table_count = 0
        
        for match in table_markers:
            marker_start = match.start()
            marker_text = match.group(0)
            
            # If we're far enough from the last chunk, add text before the table
            if marker_start - current_pos > 200:  # Minimum text size to create a chunk
                # Add text before the table as a chunk
                pre_table_text = text[current_pos:marker_start].strip()
                if pre_table_text:
                    pre_table_chunks = self._chunk_text(pre_table_text, chunk_size)
                    chunks.extend(pre_table_chunks)
            
            # Find the end of the table (next table or end of text)
            next_match = re.search(r'TABLE \d+-\d+:', text[marker_start+1:])
            if next_match:
                table_end = marker_start + 1 + next_match.start()
            else:
                # Look for double newlines after the table
                newlines_match = re.search(r'\n\n', text[marker_start+100:])  # Skip the table header
                if newlines_match:
                    table_end = marker_start + 100 + newlines_match.end()
                else:
                    table_end = len(text)
            
            # Get the table text
            table_text = text[marker_start:table_end].strip()
            
            # Check if this is a formulary table
            is_formulary_table = any(keyword in table_text.lower() for keyword in formulary_keywords)
            
            if is_formulary_table:
                formulary_table_count += 1
                # Rename the table to indicate it's a formulary table
                table_text = table_text.replace(marker_text, f"FORMULARY TABLE {formulary_table_count}:")
                print(f"Found formulary table #{formulary_table_count} in document")
            
            # Add the table as a single chunk
            if table_text:
                # Add metadata to the chunk
                metadata = {
                    "has_table": True,
                    "is_formulary_table": is_formulary_table,
                    "table_number": formulary_table_count if is_formulary_table else 0
                }
                
                # Store metadata in Pinecone when we add this chunk
                chunks.append(table_text)
            
            current_pos = table_end
        
        # Add any remaining text
        if current_pos < len(text):
            remaining_text = text[current_pos:].strip()
            if remaining_text:
                remaining_chunks = self._chunk_text(remaining_text, chunk_size)
                chunks.extend(remaining_chunks)
        
        return chunks, table_count
    
    def _chunk_text(self, text, chunk_size):
        """Split text into chunks of approximately equal size"""
        chunks = []
        
        # If text is short enough, return as a single chunk
        if len(text) <= chunk_size:
            return [text]
        
        # Otherwise, split into chunks of approximately chunk_size
        # Try to split at paragraph or sentence boundaries when possible
        start = 0
        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to find a paragraph break
            paragraph_break = text.rfind('\n\n', start, end)
            if paragraph_break != -1:
                end = paragraph_break + 2  # Include the newlines
            else:
                # Try to find a sentence break (period followed by space)
                sentence_break = text.rfind('. ', start, end)
                if sentence_break != -1:
                    end = sentence_break + 2  # Include the period and space
                else:
                    # If no natural breaks found, try to find a space
                    space = text.rfind(' ', start, end)
                    if space != -1:
                        end = space + 1  # Include the space
            
            chunks.append(text[start:end])
            start = end
        
        return chunks
    
    def process_pdf(self, pdf_path, namespace="formulary", batch_size=10):
        """
        Process a PDF file with table awareness and store in Pinecone
        
        Args:
            pdf_path: Path to the PDF file
            namespace: Namespace in Pinecone for this document
            batch_size: Number of chunks to process at once (for memory efficiency)
        
        Returns:
            Number of chunks processed
        """
        try:
            # Extract filename without extension for metadata
            filename = os.path.basename(pdf_path)
            file_id = os.path.splitext(filename)[0]
            
            print(f"Processing {filename} with table-aware extraction...")
            
            # Parse PDF with table structure preservation
            content = self.parse_pdf_with_llamaparse(pdf_path)
            
            # Free up memory after extraction
            if isinstance(content, dict) and 'text' in content:
                content_size_mb = len(content['text']) / (1024 * 1024)
                print(f"Extracted content size: {content_size_mb:.2f} MB")
            
            # Chunk the content with table awareness
            chunks, table_count = self.chunk_with_table_awareness(content, chunk_size=2000)
            
            # Free up memory after chunking
            del content
            import gc
            gc.collect()
            
            print(f"Created {len(chunks)} chunks with table awareness. Found {table_count} tables.")
            
            # Process and store chunks in batches for memory efficiency
            total_chunks = len(chunks)
            for batch_start in range(0, total_chunks, batch_size):
                batch_end = min(batch_start + batch_size, total_chunks)
                current_batch = chunks[batch_start:batch_end]
                
                print(f"Processing batch {batch_start//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}")
                
                # Process and store each chunk in the current batch
                for i, chunk in enumerate(tqdm(current_batch, desc=f"Embedding batch {batch_start//batch_size + 1}")):
                    # Adjust the chunk index to be global
                    chunk_idx = batch_start + i
                    # Create a unique ID for this chunk
                    chunk_id = f"{file_id}_chunk_{chunk_idx}"
                    
                    # Check if this chunk contains a table
                    has_table = "TABLE" in chunk or "FORMULARY TABLE" in chunk
                    is_formulary_table = "FORMULARY TABLE" in chunk
                    
                    # Get embedding for the chunk
                    embedding = self.get_embedding(chunk)
                    
                    if embedding and self.index:
                        # Store in Pinecone with metadata
                        self.index.upsert(
                            vectors=[
                                {
                                    'id': chunk_id,
                                    'values': embedding,
                                    'metadata': {
                                        'file': filename,
                                        'file_id': file_id,
                                        'chunk': chunk_idx,
                                        'text': chunk[:1000],  # Store first 1000 chars as metadata
                                        'has_table': has_table,
                                        'is_formulary_table': is_formulary_table,
                                        'insurance_provider': self._extract_insurance_provider(filename),
                                        'document_type': 'formulary'
                                    }
                                }
                            ],
                            namespace=namespace
                        )
                        
                        # Free up memory after each embedding
                        del embedding
                
                # Force garbage collection after each batch
                gc.collect()
                    
            # Small delay to avoid rate limits
            time.sleep(0.1)
            
            return len(chunks)
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return 0
    
    def process_all_pdfs(self, max_workers=2, namespace="formulary"):
        """
        Process all PDFs in the directory with table awareness
        
        Args:
            max_workers: Maximum number of parallel workers
            namespace: Namespace in Pinecone
            
        Returns:
            Total number of chunks processed
        """
        # Get all PDF files in the directory
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
        
        # Get already processed files
        processed_files = self.get_already_processed_files(namespace)
        print(f"Found {len(processed_files)} already processed files")
        
        # Filter out already processed files
        new_files = [f for f in pdf_files if os.path.splitext(f)[0] not in processed_files]
        
        if not new_files:
            print("No new files to process")
            return 0
        
        print(f"Processing {len(new_files)} new files: {new_files}")
        
        # Process files sequentially for better memory management
        total_chunks = 0
        for pdf_file in new_files:
            pdf_path = os.path.join(self.pdf_dir, pdf_file)
            chunks = self.process_pdf(pdf_path, namespace)
            total_chunks += chunks
            
            # Force garbage collection after each file
            import gc
            gc.collect()
            
            # Small delay between files
            time.sleep(1)
        
        return total_chunks
    
    def get_already_processed_files(self, namespace="formulary"):
        """Get list of already processed files from Pinecone"""
        try:
            if not self.index:
                return []
                
            # Query Pinecone for all vectors in the namespace
            stats = self.index.describe_index_stats()
            
            # Check if namespace exists
            if namespace not in stats.get('namespaces', {}):
                return []
                
            # Get all vectors in the namespace
            query_response = self.index.query(
                vector=[0] * 3072,  # Dummy vector
                top_k=10000,  # Get all vectors
                include_metadata=True,
                namespace=namespace
            )
            
            # Extract unique filenames from metadata
            processed_files = set()
            for match in query_response.matches:
                if match.metadata and 'file' in match.metadata:
                    processed_files.add(match.metadata['file'])
            
            return list(processed_files)
        except Exception as e:
            print(f"Error getting processed files: {e}")
            return []

    def process_all_pdfs(self, max_workers=2, namespace="formulary"):
        """
        Process all PDFs in the directory with table awareness
        
        Args:
            max_workers: Maximum number of parallel workers
            namespace: Namespace in Pinecone
            
        Returns:
            Total number of chunks processed
        """
        # Get all PDF files in the directory
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
        
        # Get already processed files
        processed_files = self.get_already_processed_files(namespace)
        print(f"Found {len(processed_files)} already processed files")
        
        # Filter out already processed files
        new_files = [f for f in pdf_files if f not in processed_files]
        
        if not new_files:
            print("No new files to process")
            return 0
        
        # Sort files by size (process smaller files first to see results faster)
        file_sizes = [(f, os.path.getsize(os.path.join(self.pdf_dir, f))) for f in new_files]
        file_sizes.sort(key=lambda x: x[1])  # Sort by file size
        sorted_files = [f[0] for f in file_sizes]
        
        print(f"Processing {len(sorted_files)} new files")
        for i, f in enumerate(sorted_files):
            size_mb = file_sizes[i][1] / (1024 * 1024)
            print(f"  {i+1}. {f} - {size_mb:.2f} MB")
        
        # Determine optimal batch size based on available memory
        available_memory = self._get_available_memory()
        batch_size = max(1, min(10, int(available_memory / 2)))
        print(f"Using batch size of {batch_size} chunks per batch based on available memory: {available_memory:.2f} GB")
        
        # Process files sequentially for better memory management
        total_chunks = 0
        for pdf_file in sorted_files:
            pdf_path = os.path.join(self.pdf_dir, pdf_file)
            chunks = self.process_pdf(pdf_path, namespace, batch_size=batch_size)
            total_chunks += chunks
            
            # Force garbage collection after each file
            import gc
            gc.collect()
            
            # Small delay between files
            time.sleep(2)
            
            # Print memory usage after each file
            print(f"Memory available after processing {pdf_file}: {self._get_available_memory():.2f} GB")
        
        return total_chunks
        
    def _get_available_memory(self):
        """Get available memory in GB"""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024 * 1024 * 1024)
        except ImportError:
            # If psutil is not available, assume 4GB
            return 4.0

    def _extract_insurance_provider(self, filename):
        """Extract insurance provider from filename"""
        filename = filename.lower()
        
        # Map of keywords to insurance providers
        provider_keywords = {
        'bcbs': 'Blue Cross Blue Shield',
        'bluecross': 'Blue Cross Blue Shield',
        'blue': 'Blue Cross Blue Shield',
        'aetna': 'Aetna',
        'cigna': 'Cigna',
        'united': 'UnitedHealthcare',
        'unitedhealthcare': 'UnitedHealthcare',
        'uhc': 'UnitedHealthcare',
        'humana': 'Humana',
        'kaiser': 'Kaiser Permanente',
        'medicare': 'Medicare',
        'medicaid': 'Medicaid',
        'countycare': 'CountyCare',
        'county': 'CountyCare',
        'anthem': 'Anthem',
        'tricare': 'Tricare',
        'molina': 'Molina',
        'health net': 'Health Net',
        'healthnet': 'Health Net',
        'centene': 'Centene',
        'wellcare': 'WellCare'
    }
    
        # Check for provider keywords in filename
        for keyword, provider in provider_keywords.items():
            if keyword in filename:
                return provider
        
        # If no match, return Unknown
        return 'Unknown'

    def get_already_processed_files(self, namespace="formulary"):
        """Get list of already processed files from Pinecone"""
        try:
            if not self.index:
                return []
                
            # Query Pinecone for all vectors in the namespace
            stats = self.index.describe_index_stats()
            
            # Check if namespace exists
            if namespace not in stats.get('namespaces', {}):
                return []
                
            # Get all vectors in the namespace
            query_response = self.index.query(
                vector=[0] * 3072,  # Dummy vector
                top_k=10000,  # Get all vectors
                include_metadata=True,
                namespace=namespace
            )
            
            # Extract unique filenames from metadata
            processed_files = set()
            for match in query_response.matches:
                if match.metadata and 'file' in match.metadata:
                    processed_files.add(match.metadata['file'])
            
            return list(processed_files)
        except Exception as e:
            print(f"Error getting processed files: {e}")
            return []

if __name__ == "__main__":
    print("=== LlamaParse Document Processor for Pharmacy Formulary ===")
    print("Enhanced processor with table structure preservation")
    
    # Process PDFs with table awareness
    processor = LlamaParseProcessor()
    total_chunks = processor.process_all_pdfs(max_workers=2)
    
    print(f"Processed {total_chunks} chunks with table awareness")
