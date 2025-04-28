#!/usr/bin/env python3
"""
Test script to verify the system prompt is working correctly
"""

import os
import logging
from dotenv import load_dotenv
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_system_prompt():
    """Test if the system prompt is working correctly"""
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("❌ OPENAI_API_KEY is not set")
        return False
    
    logger.info(f"Using OpenAI API key: {api_key[:10]}...{api_key[-5:]}")
    
    # System prompt from pharmacy_lightrag_agent.py
    system_message = """You are a pharmacy formulary specialist for healthcare professionals.
    Your primary function is to extract and present accurate medication coverage information from insurance formulary PDFs.
    
    CORE CAPABILITIES:
    1. You have access to embedded tables and text from insurance formulary PDFs
    2. You can find specific medications, tiers, and coverage requirements across multiple insurance plans
    3. You can interpret complex formulary tables and present them in a user-friendly format
    
    RESPONSE APPROACH:
    1. ALWAYS attempt to answer the query using your embedded knowledge from formulary PDFs
    2. When you find relevant information, ALWAYS cite the specific source document (e.g., "According to Blue Cross 2025 Formulary")
    3. If you're unsure about specific details, provide what you know and acknowledge limitations
    4. For general questions, provide helpful information even if not from a specific formulary
    
    RESPONSE FORMAT:
    For medication-specific queries:
    
    # 🏥 [Insurance Plan] Coverage for [Medication/Class]
    
    ## Coverage Details
    • Name: [Medication name with proper formatting, e.g., Advair Diskus®]
    • Form: [Generic or Brand]
    • Device type: [DPI, MDI, etc. if applicable]
    • Strength: [Available doses]
    • Tier: [Formulary tier]
    • Requirements: [PA, Step Therapy, Quantity Limit, etc.]
    • Quantity limit: [Specify the monthly limit if applicable]
    • Estimated copay: [Dollar amount if known]
    
    ## Alternative Options
    • [First alternative] - [Key difference]
    • [Second alternative] - [Key difference]
    
    ## Source Information
    This information comes from [specific PDF source with date if available].
    
    For general queries:
    Provide clear, concise information with helpful headings, bullet points, and relevant details from your knowledge base.
    
    IMPORTANT: Always provide a response that helps the healthcare professional, even if you don't have complete information. Use your embedded knowledge from formulary PDFs whenever possible, but don't refuse to answer if the exact information isn't available."""
    
    # Sample context information
    context = """
    Source: Blue Cross Blue Shield 2025 Formulary (Contains Table Data)
    ADVAIR DISKUS - fluticasone-salmeterol aer powder ba 100-50 mcg/act, 250-50 mcg/act, 500-50 mcg/act
    Tier: 3
    Requirements: ST, QL (1 inhaler/30 days)
    
    Source: CountyCare 2025 Formulary
    ADVAIR DISKUS (fluticasone-salmeterol aer powder ba) 100-50 mcg/act, 250-50 mcg/act, 500-50 mcg/act
    Tier: 2
    Requirements: None
    Quantity limit: 1 inhaler/30 days
    
    Source: Aetna 2025 Formulary (Contains Table Data)
    fluticasone-salmeterol (generic Advair) 100-50 mcg/act, 250-50 mcg/act, 500-50 mcg/act
    Tier: 1
    Requirements: None
    Quantity limit: 1 inhaler/30 days
    """
    
    # Sample query
    query = "What is the coverage for Advair Diskus in Blue Cross Blue Shield?"
    
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Create prompt
        prompt = f"""
        CONTEXT INFORMATION (INCLUDES TABLE DATA FROM FORMULARY PDFs):
        {context}
        
        QUESTION: {query}
        
        IMPORTANT INSTRUCTIONS:
        1. The context includes TABLE DATA extracted from formulary PDFs. Pay special attention to this tabular information.
        2. ALWAYS cite the specific source document in your response (e.g., "According to Blue Cross 2025 Formulary")
        3. Format your response according to the system instructions
        4. Be factual and clear, based on the provided context
        5. If information is incomplete, provide what you know and acknowledge limitations
        """
        
        # Generate response
        logger.info("Generating response...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        # Print response
        logger.info("\n=== RESPONSE ===\n")
        print(response.choices[0].message.content)
        logger.info("\n===============\n")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error testing system prompt: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting system prompt test...")
    success = test_system_prompt()
    if success:
        logger.info("✅ System prompt test completed successfully!")
    else:
        logger.error("❌ System prompt test failed!")
