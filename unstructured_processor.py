"""
Unstructured PDF Processor for Pharmacy Formulary Agent

This module uses Unstructured.io to extract tables and text from PDF files,
with special handling for formulary tables. It processes PDFs in memory-efficient
batches and stores the extracted content in Pinecone.
"""

import os
import gc
import time
import json
import psutil
import logging
import tempfile
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from tabulate import tabulate
import markdown
from datetime import datetime

# Import Unstructured components
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    Table, Text, Title, NarrativeText, ListItem, 
    Element
)

# Import PyMuPDF for fallback PDF extraction
import fitz  # PyMuPDF

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdf_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UnstructuredProcessor:
    """
    A processor that uses Unstructured.io to extract text and tables from PDFs,
    with special handling for formulary tables.
    """
    
    def __init__(self, index_name="form", chunk_size=3000, overlap=200):
        """
        Initialize the processor with the given parameters.
        
        Args:
            index_name: Name of the Pinecone index to use
            chunk_size: Maximum size of text chunks
            overlap: Overlap between chunks
        """
        self.index_name = index_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Initialize OpenAI client
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Initialize Pinecone
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        self.pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
        if not self.pinecone_env:
            raise ValueError("PINECONE_ENVIRONMENT environment variable not set")
        
        # Initialize Pinecone client
        self.pinecone_client = Pinecone(api_key=self.pinecone_api_key)
        
        # Try to connect to existing index
        try:
            self.index = self.pinecone_client.Index(self.index_name)
            logger.info(f"Connected to existing Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error connecting to Pinecone index: {e}")
            raise
        
        # Pharmacy-related keywords to identify formulary tables
        self.formulary_keywords = [
            "formulary", "medication", "drug", "prescription", "dosage", 
            "tier", "coverage", "copay", "copayment", "prior authorization",
            "quantity limit", "step therapy", "generic", "brand", "inhaler",
            "respiratory", "asthma", "COPD", "bronchodilator", "steroid",
            "mcg", "mg", "ml", "spray", "puff", "nebulizer", "inhalation",
            "preferred", "non-preferred", "specialty"
        ]
    
    def extract_elements_from_pdf(self, pdf_path: str) -> List[Element]:
        """
        Extract all elements (text and tables) from a PDF using Unstructured.io
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Unstructured Element objects
        """
        logger.info(f"Extracting elements from {os.path.basename(pdf_path)} using Unstructured.io...")
        
        try:
            # Extract elements with table detection enabled
            elements = partition_pdf(
                pdf_path,
                extract_images_in_pdf=False,  # Skip image extraction to save memory
                infer_table_structure=True,   # Enable table structure inference
                strategy="hi_res"             # Use high resolution strategy for better accuracy
            )
            
            logger.info(f"Extracted {len(elements)} elements from {os.path.basename(pdf_path)}")
            return elements
            
        except Exception as e:
            logger.error(f"Error extracting elements from {os.path.basename(pdf_path)}: {e}")
            logger.info(f"Falling back to PyMuPDF for {os.path.basename(pdf_path)}")
            return self.extract_with_pymupdf(pdf_path)
    
    def extract_with_pymupdf(self, pdf_path: str) -> List[Element]:
        """
        Fallback method to extract text and tables from PDF using PyMuPDF
        when Unstructured.io fails (e.g., when Poppler is not installed)
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of elements (converted to Unstructured-like format)
        """
        logger.info(f"Using PyMuPDF fallback for {os.path.basename(pdf_path)}")
        elements = []
        
        try:
            # Open the PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc):
                # Extract text
                text = page.get_text()
                if text.strip():
                    # Create a NarrativeText element
                    elements.append(NarrativeText(text=text))
                
                # Extract tables
                tables = page.find_tables()
                for table in tables.tables:
                    rows = []
                    for cells in table.cells:
                        row = []
                        for cell in cells:
                            # Get text from the cell
                            rect = fitz.Rect(cell.bbox.x0, cell.bbox.y0, cell.bbox.x1, cell.bbox.y1)
                            cell_text = page.get_text("text", clip=rect)
                            row.append(cell_text.strip())
                        rows.append(row)
                    
                    # Convert to pandas DataFrame for easier handling
                    if rows:
                        df = pd.DataFrame(rows)
                        # Create a Table element
                        table_text = tabulate(df, headers='firstrow', tablefmt='pipe')
                        elements.append(Table(text=table_text))
            
            logger.info(f"Extracted {len(elements)} elements from {os.path.basename(pdf_path)} using PyMuPDF")
            return elements
            
        except Exception as e:
            logger.error(f"Error in PyMuPDF fallback for {os.path.basename(pdf_path)}: {e}")
            return []
    
    def is_formulary_table(self, table_text: str) -> bool:
        """
        Determine if a table is likely a formulary table based on keywords.
        
        Args:
            table_text: Text content of the table
            
        Returns:
            True if the table is likely a formulary table, False otherwise
        """
        table_text_lower = table_text.lower()
        keyword_count = sum(1 for keyword in self.formulary_keywords if keyword.lower() in table_text_lower)
        
        # If at least 3 keywords are present, it's likely a formulary table
        return keyword_count >= 3
    
    def table_to_markdown(self, table: Table) -> str:
        """
        Convert an Unstructured Table element to markdown format.
        
        Args:
            table: Unstructured Table element
            
        Returns:
            Markdown representation of the table
        """
        if not hasattr(table, 'metadata') or not hasattr(table.metadata, 'text_as_html'):
            # If no HTML representation is available, use the text representation
            return f"```\n{table.text}\n```"
        
        try:
            # Try to convert to pandas DataFrame first
            if hasattr(table, 'to_pandas'):
                df = table.to_pandas()
                return tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
            
            # Fallback to text representation
            return f"```\n{table.text}\n```"
        except Exception as e:
            logger.warning(f"Error converting table to markdown: {e}")
            return f"```\n{table.text}\n```"
    
    def chunk_elements(self, elements: List[Element], insurance_provider: str) -> List[Dict[str, Any]]:
        """
        Chunk the extracted elements, keeping tables intact and chunking text.
        
        Args:
            elements: List of Unstructured Element objects
            insurance_provider: Name of the insurance provider
            
        Returns:
            List of chunks with text, metadata, and table information
        """
        chunks = []
        current_chunk = ""
        current_chunk_elements = []
        
        for element in elements:
            # Handle tables specially
            if isinstance(element, Table):
                # If we have accumulated text, create a chunk for it
                if current_chunk:
                    chunks.append({
                        "text": current_chunk,
                        "has_table": False,
                        "is_formulary_table": False,
                        "elements": current_chunk_elements,
                        "insurance_provider": insurance_provider
                    })
                    current_chunk = ""
                    current_chunk_elements = []
                
                # Convert table to markdown
                table_md = self.table_to_markdown(element)
                
                # Check if it's a formulary table
                is_formulary = self.is_formulary_table(element.text)
                
                # Create a chunk for the table
                chunks.append({
                    "text": f"Table:\n{table_md}",
                    "has_table": True,
                    "is_formulary_table": is_formulary,
                    "elements": [element],
                    "insurance_provider": insurance_provider
                })
            
            # Handle text elements
            elif isinstance(element, (Text, Title, NarrativeText, ListItem)):
                # Skip empty elements
                if not element.text.strip():
                    continue
                
                # If adding this element would exceed chunk size, create a new chunk
                if len(current_chunk) + len(element.text) > self.chunk_size:
                    # If current chunk is not empty, add it to chunks
                    if current_chunk:
                        chunks.append({
                            "text": current_chunk,
                            "has_table": False,
                            "is_formulary_table": False,
                            "elements": current_chunk_elements,
                            "insurance_provider": insurance_provider
                        })
                    
                    # Start a new chunk with overlap
                    words = current_chunk.split()
                    overlap_text = " ".join(words[-self.overlap:]) if len(words) > self.overlap else ""
                    current_chunk = overlap_text + " " + element.text
                    current_chunk_elements = [element]
                else:
                    # Add element to current chunk
                    current_chunk += " " + element.text
                    current_chunk_elements.append(element)
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "has_table": False,
                "is_formulary_table": False,
                "elements": current_chunk_elements,
                "insurance_provider": insurance_provider
            })
        
        return chunks
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of texts using OpenAI.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            # Create embeddings in batches to avoid rate limits
            embeddings = []
            batch_size = 100  # OpenAI can handle up to 2048 texts per request, but we'll be conservative
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                response = self.openai_client.embeddings.create(
                    input=batch_texts,
                    model="text-embedding-3-large"
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # Sleep to avoid rate limits
                if i + batch_size < len(texts):
                    time.sleep(0.5)
            
            return embeddings
        
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def upsert_to_pinecone(self, chunks: List[Dict[str, Any]], pdf_path: str) -> int:
        """
        Create embeddings for chunks and upsert to Pinecone.
        
        Args:
            chunks: List of chunks with text and metadata
            pdf_path: Path to the PDF file
            
        Returns:
            Number of chunks upserted
        """
        if not chunks:
            logger.warning(f"No chunks to upsert for {os.path.basename(pdf_path)}")
            return 0
        
        try:
            # Extract text from chunks
            texts = [chunk["text"] for chunk in chunks]
            
            # Create embeddings
            embeddings = self.create_embeddings(texts)
            
            # Prepare vectors for upsert
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Create metadata
                metadata = {
                    "text": chunk["text"],
                    "source": os.path.basename(pdf_path),
                    "has_table": chunk["has_table"],
                    "is_formulary_table": chunk["is_formulary_table"],
                    "insurance_provider": chunk["insurance_provider"],
                    "chunk_id": i
                }
                
                # Create vector
                vector = {
                    "id": f"{os.path.basename(pdf_path)}_chunk_{i}",
                    "values": embedding,
                    "metadata": metadata
                }
                
                vectors.append(vector)
            
            # Upsert vectors in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch_vectors = vectors[i:i+batch_size]
                self.index.upsert(vectors=batch_vectors)
                
                # Sleep to avoid rate limits
                if i + batch_size < len(vectors):
                    time.sleep(0.5)
            
            logger.info(f"Upserted {len(vectors)} vectors to Pinecone for {os.path.basename(pdf_path)}")
            return len(vectors)
        
        except Exception as e:
            logger.error(f"Error upserting to Pinecone: {e}")
            raise
    
    def process_pdf(self, pdf_path: str) -> int:
        """
        Process a single PDF file and store the extracted content in Pinecone.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Number of chunks processed
        """
        logger.info(f"Processing {os.path.basename(pdf_path)}...")
        
        try:
            # Extract insurance provider from filename
            filename = os.path.basename(pdf_path)
            insurance_provider = "Unknown"
            
            # Try to identify insurance provider from filename
            if "BCBS" in filename:
                insurance_provider = "Blue Cross Blue Shield"
            elif "Aetna" in filename:
                insurance_provider = "Aetna"
            elif "Cigna" in filename:
                insurance_provider = "Cigna"
            elif "UHC" in filename or "UnitedHealthcare" in filename:
                insurance_provider = "UnitedHealthcare"
            elif "Humana" in filename:
                insurance_provider = "Humana"
            elif "Kaiser" in filename:
                insurance_provider = "Kaiser Permanente"
            elif "Medicare" in filename:
                insurance_provider = "Medicare"
            elif "Medicaid" in filename:
                insurance_provider = "Medicaid"
            elif "CountyCare" in filename:
                insurance_provider = "CountyCare"
            
            # Extract elements from PDF
            elements = self.extract_elements_from_pdf(pdf_path)
            
            if not elements:
                logger.warning(f"No elements extracted from {os.path.basename(pdf_path)}")
                return 0
            
            # Chunk elements
            chunks = self.chunk_elements(elements, insurance_provider)
            
            # Upsert to Pinecone
            num_chunks = self.upsert_to_pinecone(chunks, pdf_path)
            
            # Force garbage collection to free memory
            elements = None
            chunks = None
            gc.collect()
            
            return num_chunks
        
        except Exception as e:
            logger.error(f"Error processing {os.path.basename(pdf_path)}: {e}")
            return 0
    
    def process_pdfs_in_batches(self, pdf_dir: str, batch_size: int = 1, delay_between_batches: int = 10) -> int:
        """
        Process PDF files in batches to manage memory usage.
        
        Args:
            pdf_dir: Directory containing PDF files
            batch_size: Number of PDFs to process in each batch
            delay_between_batches: Delay between batches in seconds
            
        Returns:
            Total number of chunks processed
        """
        # Find all PDF files
        pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) 
                    if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(pdf_dir, f))]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return 0
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Check available memory and adjust batch size if needed
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        logger.info(f"Available memory: {available_gb:.2f} GB")
        
        # Adjust batch size based on available memory
        if available_gb < 2:
            batch_size = 1
            logger.warning(f"Low memory available ({available_gb:.2f} GB). Setting batch size to 1.")
        elif available_gb < 4:
            batch_size = min(batch_size, 2)
            logger.info(f"Setting batch size to {batch_size} based on available memory.")
        
        # Process PDFs in batches
        total_chunks = 0
        num_batches = (len(pdf_files) + batch_size - 1) // batch_size
        
        logger.info(f"Processing {num_batches} batches with up to {batch_size} PDFs per batch")
        
        for i in range(0, len(pdf_files), batch_size):
            batch = pdf_files[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{num_batches}: {[os.path.basename(f) for f in batch]}")
            
            for pdf_file in batch:
                num_chunks = self.process_pdf(pdf_file)
                total_chunks += num_chunks
                
                # Force garbage collection
                gc.collect()
            
            # Delay between batches to allow system to recover
            if i + batch_size < len(pdf_files):
                logger.info(f"Sleeping for {delay_between_batches} seconds before next batch...")
                time.sleep(delay_between_batches)
        
        logger.info(f"Processed {len(pdf_files)} PDF files with {total_chunks} total chunks")
        return total_chunks


def process_all_pdfs(pdf_dir: str = "data", batch_size: int = 2, delay_between_batches: int = 10) -> int:
    """
    Process all PDFs in the specified directory.
    
    Args:
        pdf_dir: Directory containing PDF files
        batch_size: Number of PDFs to process in each batch
        delay_between_batches: Delay between batches in seconds
        
    Returns:
        Total number of chunks processed
    """
    processor = UnstructuredProcessor()
    return processor.process_pdfs_in_batches(pdf_dir, batch_size, delay_between_batches)


if __name__ == "__main__":
    print("=== Processing PDFs with Unstructured.io ===")
    print(f"Time started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Process all PDFs
    total_chunks = process_all_pdfs()
    
    print(f"Total chunks processed: {total_chunks}")
    print(f"Time finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
