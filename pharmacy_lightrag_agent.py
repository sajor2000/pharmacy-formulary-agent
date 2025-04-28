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
import numpy as np
from sklearn.decomposition import PCA

# Handle Pinecone import for both old and new package structures
try:
    # New package structure
    from pinecone import Pinecone
except ImportError:
    try:
        # Old package structure
        import pinecone
        Pinecone = pinecone.Pinecone if hasattr(pinecone, 'Pinecone') else pinecone
    except ImportError:
        raise ImportError("Could not import Pinecone. Please install with 'pip install pinecone'.")

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
        try:
            # New package structure
            self.pinecone_client = Pinecone(api_key=self.pinecone_api_key)
            self.index = self.pinecone_client.Index("finalpharm")
        except Exception as e:
            # Old package structure fallback
            try:
                import pinecone
                pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_environment)
                self.index = pinecone.Index("finalpharm")
            except Exception as nested_e:
                raise Exception(f"Failed to initialize Pinecone: {e}. Nested error: {nested_e}")
        
        # Define system message for structured formatting
        self.system_message = """You are a pharmacy formulary assistant for nurses. 
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
        
        For ICS-LABA formulary information specifically, use this exact structure:
        
        1. Start with a brief confirmation of the insurance and medication class
        2. Use a big, clear, centered title: 🌟 [Insurance Provider] ICS-LABA Coverage (2025)
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
        
        # Define custom prompt template for pharmacy formulary queries
        self.base_prompt_template = """
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
    
    def query(self, question, insurance=None, medication_class=None):
        """Query the pharmacy formulary agent"""
        try:
            # Log the query
            logger.info(f"Query: {question}")
            if insurance:
                logger.info(f"Insurance filter: {insurance}")
            if medication_class:
                logger.info(f"Medication class filter: {medication_class}")
            
            # Augment the query with insurance and medication class if provided
            augmented_query = question
            if insurance:
                augmented_query += f" for {insurance} insurance"
            if medication_class:
                augmented_query += f" in the {medication_class} class"
            
            # Generate embedding for the query
            query_embedding = self._generate_embedding(augmented_query)
            
            if not query_embedding:
                return "I'm sorry, I couldn't process your query. Please try again."
            
            # No need to resize embedding as we're using text-embedding-3-large which outputs 3072 dimensions
            # which matches our Pinecone index
            
            # Search Pinecone for relevant documents
            # Increase top_k for table-heavy content to get more context
            top_k = 15 if 'table' in question.lower() or 'tier' in question.lower() else 10
            search_results = self._search_pinecone(query_embedding, top_k=top_k)
            
            if not search_results or not search_results.get('matches'):
                return "I couldn't find any relevant information in the formulary database. Please try a different query."
            
            # Extract context from search results with table awareness
            context = self._extract_context(search_results, insurance)
            
            # Check if we have table data in the context
            has_table_data = any('TABLE' in match.get('metadata', {}).get('text', '') 
                               for match in search_results.get('matches', []))
            
            if has_table_data:
                logger.info("Query contains table data - using enhanced table handling")
            
            if not context:
                return "I found some information, but it doesn't appear to be relevant to your query. Please try being more specific."
            
            # Generate response using LightRAG or direct API
            if LIGHTRAG_AVAILABLE:
                response = self._generate_lightrag_response(augmented_query, context, has_table_data)
            else:
                response = self._generate_direct_response(augmented_query, context, has_table_data)
            
            return response
        except Exception as e:
            logger.error(f"Error querying pharmacy formulary agent: {e}")
            return f"Error: {str(e)}"
            
    def _generate_lightrag_response(self, query, context, has_table_data=False):
        """Generate response using LightRAG"""
        try:
            # Prepare the prompt template with enhanced table handling if needed
            if has_table_data:
                prompt_template = """
                You are a pharmacy formulary assistant for nurses. Your task is to provide accurate information about medication coverage based on insurance formularies.
                
                CONTEXT INFORMATION (INCLUDES TABLE DATA):
                {context}
                
                QUESTION: {query}
                
                IMPORTANT: The context includes TABLE DATA from formulary documents. Pay special attention to the tabular information when answering.
                
                Please provide a structured response with the following sections:
                1. Primary Recommendation: The main medication option with form, tier, requirements, and quantity limits
                2. Alternative Options: 1-2 alternative medications with key differences
                3. Coverage Notes: Any special requirements, prior authorization criteria, or restrictions
                
                Your response should be factual, clear, and based ONLY on the provided context. If you don't know something, say so clearly.
                """
            else:
                prompt_template = """
                You are a pharmacy formulary assistant for nurses. Your task is to provide accurate information about medication coverage based on insurance formularies.
                
                CONTEXT INFORMATION:
                {context}
                
                QUESTION: {query}
                
                Please provide a structured response with the following sections:
                1. Primary Recommendation: The main medication option with form, tier, requirements, and quantity limits
                2. Alternative Options: 1-2 alternative medications with key differences
                3. Coverage Notes: Any special requirements, prior authorization criteria, or restrictions
                
                Your response should be factual, clear, and based ONLY on the provided context. If you don't know something, say so clearly.
                """
            
            # Format the prompt with query and context
            prompt = prompt_template.format(query=query, context=context)
            
            # Generate completion using LightRAG
            response = gpt_4o_complete(prompt)
            
            return response
        except Exception as e:
            logger.error(f"Error generating LightRAG response: {e}")
            # Fall back to direct API
            return self._generate_direct_response(query, context, has_table_data)
            
    def _generate_embedding(self, text):
        """Generate embedding for text using OpenAI"""
        try:
            # Preprocess text to handle table formatting
            text = self._preprocess_text_for_embedding(text)
            
            if LIGHTRAG_AVAILABLE:
                # Use LightRAG's embedding function
                embedding = openai_embed(text)
                return embedding
            else:
                # Direct API call
                response = self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                embedding = response.data[0].embedding
                return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
            
    def _preprocess_text_for_embedding(self, text):
        """Preprocess text to better handle table data for embeddings"""
        # Check if text contains markdown table markers
        if '|' in text and '-|-' in text:
            # This is likely a markdown table, preserve it
            return text
            
        # Check for TABLE markers from our LlamaParse processor
        if 'TABLE' in text and text.count('\n') > 5:
            # This is likely a table from our processor
            return text
            
        # For regular text, we can do some light preprocessing
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        return text
            
    def _resize_embedding(self, embedding, target_dim=3072):
        """Resize embedding to target dimension"""
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
    
    def _search_pinecone(self, query_embedding, top_k=10):
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
    
    def _extract_context(self, search_results, insurance=None):
        """Extract context from search results with table awareness"""
        context = ""
        
        # Filter by insurance if provided
        filtered_matches = []
        for match in search_results.get('matches', []):
            metadata = match.get('metadata', {})
            filename = metadata.get('filename', '')
            
            # If insurance filter is provided, only include matches from that insurance
            if insurance and insurance.lower() not in filename.lower():
                continue
                
            filtered_matches.append(match)
        
        # If no matches after filtering, return empty context
        if not filtered_matches:
            return ""
        
        # Prioritize table data for certain query types
        table_matches = [m for m in filtered_matches if m.get('metadata', {}).get('has_table', False)]
        non_table_matches = [m for m in filtered_matches if not m.get('metadata', {}).get('has_table', False)]
        
        # If we have both table and non-table matches, prioritize tables but include some non-table context
        if table_matches and non_table_matches:
            # Use all table matches and some non-table matches
            prioritized_matches = table_matches + non_table_matches[:3]
        else:
            # Use all matches
            prioritized_matches = filtered_matches
        
        # Extract text from filtered matches
        for match in prioritized_matches:
            metadata = match.get('metadata', {})
            text = metadata.get('text', '')
            filename = metadata.get('filename', '')
            has_table = metadata.get('has_table', False)
            
                # Add source information with table indicator if relevant
            table_indicator = " (Contains Table Data)" if has_table else ""
            context += f"\nSource: {filename}{table_indicator}\n{text}\n"
        
        return context
    
    def _generate_direct_response(self, query, context, has_table_data=False):
        """Generate response using direct OpenAI API call"""
        try:
            # Prepare the prompt template with enhanced table handling if needed
            if has_table_data:
                prompt_template = """
                You are a pharmacy formulary assistant for nurses. Your task is to provide accurate information about medication coverage based on insurance formularies.
                
                CONTEXT INFORMATION (INCLUDES TABLE DATA):
                {context}
                
                QUESTION: {query}
                
                IMPORTANT: The context includes TABLE DATA from formulary documents. Pay special attention to the tabular information when answering.
                
                Please provide a structured response with the following sections:
                1. Primary Recommendation: The main medication option with form, tier, requirements, and quantity limits
                2. Alternative Options: 1-2 alternative medications with key differences
                3. Coverage Notes: Any special requirements, prior authorization criteria, or restrictions
                
                Your response should be factual, clear, and based ONLY on the provided context. If you don't know something, say so clearly.
                """
            else:
                prompt_template = """
                You are a pharmacy formulary assistant for nurses. Your task is to provide accurate information about medication coverage based on insurance formularies.
                
                CONTEXT INFORMATION:
                {context}
                
                QUESTION: {query}
                
                Please provide a structured response with the following sections:
                1. Primary Recommendation: The main medication option with form, tier, requirements, and quantity limits
                2. Alternative Options: 1-2 alternative medications with key differences
                3. Coverage Notes: Any special requirements, prior authorization criteria, or restrictions
                
                Your response should be factual, clear, and based ONLY on the provided context. If you don't know something, say so clearly.
                """
            
            # Format the prompt with query and context
            prompt = prompt_template.format(query=query, context=context)
            
            # Generate completion using OpenAI
            if is_countycare_ics_laba:
                # Use a specialized prompt for CountyCare ICS-LABA queries
                prompt = f"""Please provide detailed information about CountyCare coverage for ICS-LABA inhalers.
                Follow the exact structured format for CountyCare ICS-LABA information as specified in your instructions.
                Make sure to include all the required sections: confirmation, title, primary recommendations, alternative options,
                coverage notes, and verification.
                
                Use the following context from the formulary documents:
                {context}
                
                Question: {query}
                """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a pharmacy formulary assistant for nurses. Provide accurate, structured information about medication coverage based on insurance formularies."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating direct response: {e}")
            return "I'm sorry, I encountered an error while generating a response. Please try again."
    
    def direct_query(self, query):
        """Query the agent directly with OpenAI without using RAG"""
        try:
            # Check if this is a CountyCare ICS-LABA query
            is_countycare_ics_laba = False
            if "countycare" in query.lower() and ("ics-laba" in query.lower() or 
                                               "ics laba" in query.lower() or
                                               "inhaled corticosteroid" in query.lower()):
                is_countycare_ics_laba = True
                print("Detected CountyCare ICS-LABA query, using structured format")
            
            # Use a simple prompt template
            if is_countycare_ics_laba:
                # Use a specialized prompt for CountyCare ICS-LABA queries
                prompt = f"""Please provide detailed information about CountyCare coverage for ICS-LABA inhalers.
                Follow the exact structured format for CountyCare ICS-LABA information as specified in your instructions.
                Make sure to include all the required sections: confirmation, title, primary recommendations, alternative options,
                coverage notes, and verification.
                
                Question: {query}
                """
            else:
                prompt = f"""
                Answer the following question about pharmacy formularies:
                {query}
                """
            
            # Generate completion using OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error querying agent: {e}")
            return f"I'm sorry, I encountered an error: {e}"
    
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
