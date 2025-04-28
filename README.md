# Pharmacy Formulary RAG Agent

A production-ready, table-aware pharmacy formulary retrieval-augmented generation (RAG) agent that helps nurses find medication coverage information quickly and accurately. The system extracts tables and text from pharmacy formulary PDFs, stores them in a vector database, and delivers structured, nurse-friendly answers in a web UI.

## 🌟 Features

- **Table-Aware Extraction**: Extracts both tables and text from formulary PDFs using Camelot and PyMuPDF
- **Structured Output**: Delivers markdown-formatted, emoji-enhanced responses optimized for nurses
- **Vector Search**: Uses OpenAI's text-embedding-3-large (3072 dimensions) with Pinecone vector database
- **Memory Optimized**: Batch processing designed for Mac M3 hardware (18GB RAM)
- **Production Ready**: Includes Render deployment configuration and web UI
- **CountyCare Focus**: Special handling for CountyCare ICS-LABA inhaler queries

## 📋 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Copy the `.env.example` file to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Then edit the `.env` file with your API keys:
- `OPENAI_API_KEY`: Your OpenAI API key
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_ENVIRONMENT`: Your Pinecone index URL (e.g., "https://yourindex.svc.environment.pinecone.io")
- `LLAMAPARSE_API_KEY`: (Optional) Your LlamaParse API key

### 3. Create a Pinecone Index

Create a Pinecone serverless index named "finalpharm" with 3072 dimensions (for compatibility with OpenAI's text-embedding-3-large):

```bash
python create_3072d_index.py
```

### 4. Add PDF Documents

Create a `data` directory and add your insurance formulary PDFs:

```bash
mkdir -p data
```

## 🔄 Processing PDFs

The system offers multiple PDF processing options:

### Camelot Processor (Recommended)

```bash
python process_with_camelot.py
```

This uses Camelot for table extraction and PyMuPDF for text extraction. No external dependencies required.

### Unstructured.io Processor (Alternative)

```bash
python process_with_unstructured.py
```

Requires Poppler to be installed on your system.

### LlamaParse Processor (Alternative)

```bash
python process_with_llamaparse.py
```

Requires a LlamaParse API key.

## 💾 Memory Usage Optimizations

The PDF processing includes several memory optimizations for Mac M3 hardware:
- Processes PDFs in small batches (2 at a time)
- Uses text chunking with table awareness
- Clears memory after each embedding is created
- Includes delays between batches to allow memory cleanup

## 🏥 Supported Insurance Providers

The system is designed to work with formularies from:
- CountyCare (primary focus)
- Blue Cross Blue Shield
- UnitedHealthcare
- Cigna
- Meridian
- Wellcare
- Other Medicaid/Medicare providers

## 🖥️ Web Interface

Run the web interface locally:

```bash
python app.py
```

Or deploy to Render using the included `render.yaml` configuration.
