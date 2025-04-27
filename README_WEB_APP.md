# Pharmacy Formulary Web Application

This web application provides a user-friendly interface for healthcare providers to query insurance formulary information using the LightRAG framework.

## Features

- **Interactive Chat Interface**: Ask questions about medication coverage in natural language
- **Insurance Provider Filtering**: Easily filter queries by specific insurance providers
- **Common Medications**: Quick access to frequently prescribed medications
- **Tier Information**: Visual indicators for different medication tiers

## Setup Instructions

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Ensure your .env file is configured**

Make sure your `.env` file contains the necessary API keys:
- `OPENAI_API_KEY`: Your OpenAI API key
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_ENVIRONMENT`: Your Pinecone environment

3. **Process PDF formularies (if not already done)**

If you haven't already processed your PDF formularies, run:
```bash
python m3_optimized_processor.py
```

4. **Start the web application**

```bash
python app.py
```

5. **Access the web interface**

Open your browser and go to:
```
http://localhost:5000
```

## Using the Web Application

### Basic Queries
- Type your question in the input field and press Enter or click the Send button
- Example queries:
  - "What tier is Advair in BCBS insurance?"
  - "Are there any lower-tier alternatives to Symbicort?"
  - "Does Spiriva require prior authorization?"
  - "Compare Breo Ellipta and Trelegy for UnitedHealthcare"

### Filtering by Insurance
- Click on an insurance provider badge to filter your queries to that specific provider
- The system will automatically include the selected insurance in your queries
- Click the selected insurance again to clear the filter

### Quick Medication Lookup
- Click on any medication in the "Common Medications" section to quickly ask about its tier
- This will pre-fill the query input with "What tier is [medication]?"

## Advanced Features

### Specialized Query Types

The system supports several specialized query types:

1. **Tier Lookup**: Find which tier a medication is in
   - "What tier is [medication] in [insurance]?"

2. **Lower-Tier Alternatives**: Find more affordable options
   - "What are lower tier alternatives to [medication]?"
   - "Is there a generic for [medication]?"

3. **Prior Authorization**: Check if PA is required
   - "Does [medication] require prior authorization?"
   - "What are the PA requirements for [medication]?"

4. **Coverage Comparison**: Compare multiple medications
   - "Compare [medication1] and [medication2] coverage"
   - "Which is better covered, [medication1] or [medication2]?"

5. **Formulary Restrictions**: Check for quantity limits or step therapy
   - "Are there any restrictions for [medication]?"
   - "Does [medication] have quantity limits?"

## Troubleshooting

If you encounter any issues:

1. **Check API Keys**: Ensure your OpenAI and Pinecone API keys are correct in the `.env` file
2. **Verify Pinecone Index**: Make sure your "form" index exists in Pinecone with the processed formulary data
3. **Check Network Connection**: Ensure you have internet connectivity for API calls
4. **Restart Application**: Sometimes simply restarting the application can resolve issues

For persistent problems, check the terminal output where you started the application for error messages.
