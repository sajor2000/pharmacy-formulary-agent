# Pharmacy Formulary AI Agent

This project processes insurance formulary PDF documents and creates a vector database to help healthcare providers find the best medication options based on insurance coverage, with a focus on respiratory medications and inhalers.

## Features

- Processes PDF formulary documents from various insurance providers
- Creates embeddings for vector search using OpenAI
- Stores vectors in Pinecone for efficient retrieval
- Prioritizes lowest-tier medications in recommendations
- Handles large PDFs with text chunking to avoid token limit errors

## Setup Instructions

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Set up environment variables**

Copy the `.env.example` file to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Then edit the `.env` file with your actual API keys:
- `OPENAI_API_KEY`: Your OpenAI API key
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_ENVIRONMENT`: Your Pinecone environment (e.g., "gcp-starter")

3. **Create a Pinecone index**

Create a Pinecone index named "form" with 1024 dimensions.

4. **Add PDF documents**

Create a `data` directory and add your insurance formulary PDFs:

```bash
mkdir data
```

Then copy your PDF files into the `data` directory.

## Processing PDFs

Run the PDF processing script to extract text, create embeddings, and store them in Pinecone:

```bash
python process_pdfs.py
```

This script processes PDFs in small batches to avoid memory issues.

## Memory Usage Considerations

The PDF processing can be memory-intensive. The script includes several optimizations:
- Processes PDFs in small batches (3 at a time)
- Uses text chunking to handle large documents
- Clears memory after each embedding is created
- Includes delays between batches to allow memory cleanup

## Insurance Providers

The system is designed to work with formularies from:
- Blue Cross Blue Shield
- UnitedHealthcare
- Cigna
- Express Scripts
- Humana
- Meridian
- Wellcare
- CountyCare
