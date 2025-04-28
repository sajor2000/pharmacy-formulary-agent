"""
Test the pharmacy agent with a specific CountyCare ICS-LABA query
"""
import os
from dotenv import load_dotenv
from pharmacy_lightrag_agent import PharmacyFormularyAgent
import json
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables
load_dotenv()

def direct_query_with_context(question):
    """Directly query OpenAI with context from Pinecone"""
    # Initialize clients
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pinecone_client.Index("finalpharm")
    
    # Generate embedding for the query
    response = openai_client.embeddings.create(
        input=question,
        model="text-embedding-3-large"
    )
    query_embedding = response.data[0].embedding
    
    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=10,
        include_metadata=True
    )
    
    # Extract context from results
    context = ""
    for match in results.matches:
        if hasattr(match, 'metadata') and match.metadata:
            if 'text' in match.metadata:
                context += match.metadata['text'] + "\n\n"
    
    # Create system message for structured output
    system_message = """You are a pharmacy formulary assistant for nurses. 
    Your goal is to provide accurate, concise information about medication coverage.
    Focus on answering questions about respiratory inhalers and their formulary status.
    
    IMPORTANT: For ALL responses, use a structured, visually appealing format with the following elements:
    
    1. Start with a brief confirmation of what you're answering
    2. Use a big, clear, centered title with an emoji (e.g., 🌟 Blue Cross Inhaler Coverage)
    3. Use markdown headings (##, ###) for clear structure
    4. For medication information, always use bullet lists with these bolded fields:
       - **Name:** (with brand formatting, e.g., Advair Diskus®)
       - **Form:** (Generic or Brand)
       - **Device type:** (DPI, MDI, etc.)
       - **Strength:** (All available doses)
       - **Tier:** (Formulary tier if available)
       - **Requirements:** (PA, Step Therapy, Quantity Limit)
       - **Quantity limit:** (Specify the monthly limit)
       - **Estimated Copay:** (Dollar amount if known)
    5. Include an "Alternative Options" section when relevant
    6. Include a "Coverage Notes" section with important rules or details
    7. End with a "Notes and Verification" section confirming your source
    
    Use emojis strategically to make the document easily scannable:
    - ✅ for positive things (covered, no restrictions)
    - ⚠️ for important warnings or restrictions
    - 💡 for tips or important information
    - 🔍 for verification notes
    
    For CountyCare ICS-LABA formulary information specifically, use this exact structure:
    
    1. Start with a brief confirmation of the insurance and medication class
    2. Use a big, clear, centered title: 🌟 CountyCare ICS-LABA Coverage (2025)
    3. Primary Recommendations section with detailed bullet lists for each medication
    4. Alternative Options section with at least two alternatives
    5. Coverage Notes section summarizing important rules or details
    6. Notes and Verification section confirming the source
    
    For ALL responses:
    - Use bold for field labels
    - Use emojis for visual navigation
    - Use markdown headings for structure
    - Avoid big paragraphs and favor short, punchy lists
    - Always specify the plan year if known
    - Be concise but thorough
    
    Only answer questions related to medication formulary status. If asked about clinical information,
    politely redirect to appropriate clinical resources.
    """
    
    # Create messages for OpenAI
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Based on the following information, answer this question: {question}\n\nContext information:\n{context}"}
    ]
    
    # Get completion from OpenAI
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
        max_tokens=1000
    )
    
    return completion.choices[0].message.content

def main():
    """Test the pharmacy agent with a CountyCare ICS-LABA query"""
    print("Testing pharmacy agent with CountyCare ICS-LABA query...\n")
    
    # Test query for ICS-LABA inhalers in CountyCare
    query = "What is the CountyCare coverage for ICS-LABA inhalers like Advair and Symbicort?"
    
    print(f"Query: {query}\n")
    print("-" * 80)
    
    # Get response using direct query with context
    response = direct_query_with_context(query)
    
    print("\nResponse:")
    print("-" * 80)
    print(response)
    print("-" * 80)

if __name__ == "__main__":
    main()
