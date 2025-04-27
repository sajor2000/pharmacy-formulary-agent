#!/usr/bin/env python3
"""
Pharmacy Formulary LightRAG Agent
---------------------------------
A lightweight RAG agent for pharmacy formulary data using LightRAG framework.
"""

import os
import sys
from dotenv import load_dotenv
import openai
from pinecone import Pinecone
import numpy as np
from sklearn.decomposition import PCA

# Import LightRAG components
try:
    # Try importing from the HKUDS/LightRAG repository structure
    from LightRAG.src.utils.openai_utils import get_embedding as openai_embed
    from LightRAG.src.utils.openai_utils import get_completion as gpt_4o_complete
    LIGHTRAG_AVAILABLE = True
except ImportError:
    LIGHTRAG_AVAILABLE = False
    print("LightRAG not available, using direct API calls instead.")

# Load environment variables
load_dotenv()

# Initialize logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pharmacy_agent")

class PharmacyFormularyAgent:
    """Pharmacy Formulary Agent using LightRAG framework"""
    
    def __init__(self):
        """Initialize the pharmacy formulary agent"""
        # Get API keys from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
        
        if not all([self.openai_api_key, self.pinecone_api_key, self.pinecone_environment]):
            raise ValueError("Missing required API keys in .env file")
        
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        
        # Initialize Pinecone
        self.pinecone_client = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pinecone_client.Index("form")
        
        # Define custom prompt template for pharmacy formulary queries
        self.prompt_template = """
            # ROLE
            You are a pharmacy inhaler formulary specialist who matches respiratory medications to insurance formulary preferences. You maintain comprehensive knowledge of all inhaler classes, formulations, and common coverage patterns.
            
            # CONTEXT INFORMATION
            {context}
            
            # COMPLETE MEDICATION REFERENCE
            Short-Acting Rescue Inhalers (SABA):
            - Albuterol Sulfate HFA (Generic) - 90 mcg
            - ProAir® Digihaler™ - 117 mcg
            - ProAir® RespiClick - 117 mcg
            - Proventil® HFA - 120 mcg
            - Ventolin® HFA - 90 mcg
            - Xopenex HFA® (levalbuterol) - 50 mcg
            - Nebulizer solutions: Albuterol (0.63mg/3ml, 1.25mg/3ml, 2.5mg/3ml), Levalbuterol (0.31mg/3ml, 0.63mg/3ml, 1.25mg/3ml)
            
            Inhaled Corticosteroids (ICS):
            - Arnuity® Ellipta® (fluticasone furoate) - 100/200 mcg
            - QVAR® RediHaler™ (beclomethasone) - 40/80 mcg
            - Pulmicort® Flexhaler® (budesonide) - 90/180 mcg
            - Alvesco® (ciclesonide) - 80/160 mcg
            - Asmanex® HFA/Twisthaler® (mometasone) - 100/200 mcg
            - ArmonAir® RespiClick® (fluticasone) - 55/113/232 mcg
            
            ICS-LABA Combinations:
            - Fluticasone/Salmeterol options: Wixela™ Inhub™ (generic) - 100/50, 250/50, 500/50 mcg; Advair® Diskus® - 100/50, 250/50, 500/50 mcg; Advair® HFA - 45/21, 115/21, 230/21 mcg; AirDuo® RespiClick® - 55/14, 113/14, 232/14 mcg
            - Breo® Ellipta® (fluticasone/vilanterol) - 100/25, 200/25 mcg
            - Symbicort® (budesonide/formoterol) - 80/4.5, 160/4.5 mcg
            - Dulera® (mometasone/formoterol) - 50/5, 100/5, 200/5 mcg
            
            LAMA Medications:
            - Spiriva® HandiHaler®/Respimat® (tiotropium) - 18 mcg/1.25 mcg
            - Incruse® Ellipta® (umeclidinium) - 62.5 mcg
            - Tudorza™ Pressair™ (aclidinium) - 400 mcg
            - Yupelri® (revefenacin) solution - 175 mcg/3ml
            
            LAMA-LABA Combinations:
            - Anoro® Ellipta® (umeclidinium/vilanterol) - 62.5/25 mcg
            - Stiolto® Respimat® (tiotropium/olodaterol) - 2.5/2.5 mcg
            - Bevespi Aerosphere® (glycopyrrolate/formoterol) - 9/4.8 mcg
            - Duaklir® Pressair® (aclidinium/formoterol) - 400/12 mcg
            
            Triple Therapy (ICS-LABA-LAMA):
            - Trelegy Ellipta (fluticasone/vilanterol/umeclidinium) - 100/62.5/25, 200/62.5/25 mcg
            - Breztri Aerosphere® (budesonide/glycopyrrolate/formoterol) - 160/9/4.8 mcg
            
            # STEPS TO FOLLOW
            1. Confirm insurance details and preferences
            2. Identify medication class options
            3. Check formulary status and restrictions
            4. Present recommendations in order of preference
            
            # QUESTION FROM HEALTHCARE PROVIDER
            {question}
            
            # RESPONSE FORMAT
            Present your recommendations in this format:
            
            Primary Recommendation:
            Medication: [Name]
            - Form: [Generic/Brand]
            - Device type: [MDI/DPI/Respimat/etc.]
            - Strength: [Available doses]
            - Tier: [Formulary tier]
            - Requirements: [PA/Step therapy/None]
            - Quantity limit: [Per 30 days]
            - Estimated copay: [If available]
            
            Alternative Options:
            1. First Alternative:
               - Name: [Medication]
               - Key difference: [Cost/Device/Coverage]
               - Requirements: [Key restrictions]
            
            2. Second Alternative:
               - Name: [Medication]
               - Key difference: [Cost/Device/Coverage]
               - Requirements: [Key restrictions]
            
            Coverage Notes:
            - Prior authorization criteria
            - Step therapy requirements
            - Preferred pharmacy requirements
            - Quantity restrictions
            - Special instructions
            
            # IMPORTANT VERIFICATION NOTE
            Check the provided documents first. If they do not contain information specific to the insurance plan/location, search an official and current source before answering. Always specify whether the answer is from uploaded documents or an external source.
            
            # ANSWER
            """
        
        # We'll implement the RAG pipeline manually instead of using LightRAG's built-in pipeline
    
    def query(self, question):
        """Query the pharmacy formulary agent"""
        try:
            # Get embedding for the query
            logger.info(f"Getting embedding for query: {question[:50]}...")
            query_embedding = self.get_embedding(question)
            
            if query_embedding is None:
                logger.error("Failed to get embedding for query")
                return "Error: Failed to get embedding for query"
                
            logger.info(f"Embedding generated successfully, dimension: {len(query_embedding)}")
            
            # Search Pinecone for relevant documents
            logger.info("Searching Pinecone for relevant documents")
            results = self.search_pinecone(query_embedding)
            
            if results is None:
                logger.error("Failed to search Pinecone")
                return "Error: Failed to search Pinecone for relevant documents"
            
            # Format the context from search results
            logger.info("Formatting context from search results")
            context = self.format_context(results)
            logger.info(f"Context length: {len(context)}")
            
            # Generate response using the prompt template and GPT-4o directly
            logger.info("Generating response using GPT-4o")
            try:
                formatted_prompt = self.prompt_template.format(context=context, question=question)
            except Exception as e:
                logger.error(f"Error formatting prompt template: {e}")
                # Try a simpler approach with the correct format
                formatted_prompt = f"""You are a pharmacy inhaler formulary specialist who matches respiratory medications to insurance formulary preferences. 

Here is some context information from formulary documents:
{context}

Question: {question}

You MUST present your recommendations in this EXACT format, even if information is limited:

Primary Recommendation:
Medication: [Name or 'No specific recommendation due to limited information']
- Form: [Generic/Brand or 'Unknown']
- Device type: [MDI/DPI/Respimat/etc. or 'Unknown']
- Strength: [Available doses or 'Unknown']
- Tier: [Formulary tier or 'Unknown']
- Requirements: [PA/Step therapy/None or 'Unknown']
- Quantity limit: [Per 30 days or 'Unknown']
- Estimated copay: [If available or 'Unknown']

Alternative Options:
1. First Alternative:
   - Name: [Medication or 'No alternatives identified']
   - Key difference: [Cost/Device/Coverage or 'Unknown']
   - Requirements: [Key restrictions or 'Unknown']

2. Second Alternative:
   - Name: [Medication or 'No additional alternatives identified']
   - Key difference: [Cost/Device/Coverage or 'Unknown']
   - Requirements: [Key restrictions or 'Unknown']

Coverage Notes:
[Include any relevant notes about coverage, prior authorization, step therapy, etc. If no information is available, state 'Limited coverage information available from the provided documents.']

Verification Note:
[Clearly state whether this answer is from the uploaded documents or external knowledge. If information is incomplete, acknowledge this.]"""
            
            try:
                # Use a more effective prompt structure
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a pharmacy inhaler formulary specialist who matches respiratory medications to insurance formulary preferences."},
                        {"role": "user", "content": formatted_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1000
                )
                
                # Extract the response text
                response_text = response.choices[0].message.content
                logger.info("Response generated successfully")
                
                return response_text
            except Exception as e:
                logger.error(f"Error generating response with GPT-4o: {e}")
                return f"Error generating response: {str(e)}"
        except Exception as e:
            logger.error(f"Error querying pharmacy formulary agent: {e}")
            return f"Error: {str(e)}"
            
    def get_embedding(self, text):
        """Get embedding for text using OpenAI's embedding model directly"""
        try:
            # Use OpenAI's embedding model directly instead of LightRAG's async function
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            
            # Get the embedding vector
            embedding = response.data[0].embedding
            
            # Resize embedding to match Pinecone index dimension (1024)
            resized_embedding = self.resize_embedding(embedding, target_dim=1024)
            
            return resized_embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
            
    def resize_embedding(self, embedding, target_dim=1024):
        """Resize embedding to target dimension using a simple approach"""
        try:
            # Convert to numpy array
            embedding_array = np.array(embedding)
            
            # Get the current dimension
            current_dim = len(embedding_array)
            
            # If current dimension is larger than target, truncate
            if current_dim > target_dim:
                return embedding_array[:target_dim].tolist()
            
            # If current dimension is smaller than target, pad with zeros
            elif current_dim < target_dim:
                padding = np.zeros(target_dim - current_dim)
                return np.concatenate([embedding_array, padding]).tolist()
            
            # If dimensions match, return as is
            else:
                return embedding_array.tolist()
        except Exception as e:
            logger.error(f"Error resizing embedding: {e}")
            return None
    
    def search_pinecone(self, query_embedding, top_k=10):
        """Search Pinecone for relevant documents"""
        try:
            results = self.index.query(
                namespace="formulary",
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            return results
        except Exception as e:
            logger.error(f"Error searching Pinecone: {e}")
            return None
    
    def format_context(self, results):
        """Format search results into context for the LLM"""
        if not results or not hasattr(results, 'matches') or not results.matches:
            return "No relevant information found in the formulary database."
        
        context_parts = []
        for i, match in enumerate(results.matches, 1):
            if hasattr(match, 'metadata') and match.metadata:
                source = match.metadata.get('source', 'Unknown source')
                insurance = match.metadata.get('insurance', 'Unknown insurance')
                content_type = match.metadata.get('type', 'Unknown type')
                
                context_parts.append(f"[Document {i}] From {insurance} formulary ({source}):")
                if hasattr(match, 'metadata') and 'content' in match.metadata:
                    context_parts.append(match.metadata['content'])
                else:
                    context_parts.append("Content not available")
                context_parts.append("")  # Empty line for separation
        
        return "\n".join(context_parts)
    
    def get_medication_tier(self, medication_name, insurance_provider=None):
        """Get the tier for a specific medication"""
        query = f"What tier is {medication_name}"
        if insurance_provider:
            query += f" in {insurance_provider} insurance"
        
        return self.query(query)
    
    def find_lower_tier_alternatives(self, medication_name, insurance_provider=None):
        """Find lower tier alternatives for a medication"""
        query = f"What are lower tier alternatives to {medication_name}"
        if insurance_provider:
            query += f" in {insurance_provider} insurance"
        
        return self.query(query)
    
    def check_prior_authorization(self, medication_name, insurance_provider=None):
        """Check if a medication requires prior authorization"""
        query = f"Does {medication_name} require prior authorization"
        if insurance_provider:
            query += f" in {insurance_provider} insurance"
        
        return self.query(query)
    
    def compare_coverage(self, medication_names, insurance_provider=None):
        """Compare coverage for multiple medications"""
        medications_list = ", ".join(medication_names)
        query = f"Compare the coverage for these medications: {medications_list}"
        if insurance_provider:
            query += f" in {insurance_provider} insurance"
        
        return self.query(query)

# Example usage
if __name__ == "__main__":
    print("=== Pharmacy Formulary LightRAG Agent ===")
    print("Initializing agent...")
    
    try:
        agent = PharmacyFormularyAgent()
        print("Agent initialized successfully!")
        
        # Interactive query loop
        print("\nEnter your questions about medication coverage (or 'exit' to quit):")
        while True:
            question = input("\nQuestion: ")
            if question.lower() in ['exit', 'quit', 'q']:
                break
            
            print("\nProcessing query...")
            response = agent.query(question)
            print(f"\nAnswer: {response}")
    
    except Exception as e:
        print(f"Error initializing pharmacy formulary agent: {e}")
