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

# Import Pinecone with the new API structure
from pinecone import Pinecone

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
        
        # Initialize Pinecone with the new API structure - no fallback
        logger.info(f"Initializing Pinecone with API key: {self.pinecone_api_key[:5]}...{self.pinecone_api_key[-5:]}")
        self.pinecone_client = Pinecone(api_key=self.pinecone_api_key)
        
        # Connect to the finalpharm index
        logger.info("Connecting to finalpharm index...")
        self.index = self.pinecone_client.Index("finalpharm")
        
        # Verify connection by getting stats
        stats = self.index.describe_index_stats()
        logger.info(f"Successfully connected to Pinecone index 'finalpharm' with {stats.get('total_vector_count', 0)} vectors")
        
        # Define system message for structured formatting
        self.system_message = """You are a pharmacy formulary specialist for healthcare professionals.
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
        
        IMPORTANT: Always provide a response that helps the healthcare professional, even if you don't have complete information. Use your embedded knowledge from formulary PDFs whenever possible, but don't refuse to answer if the exact information isn't available.
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
            
            # Always require Pinecone connection - no fallback mode
            
            # Generate embedding for the query
            try:
                logger.info(f"Generating embedding for query: {augmented_query[:100]}...")
                query_embedding = self._generate_embedding(augmented_query)
                
                if not query_embedding:
                    error_msg = "Failed to generate embedding for query"
                    logger.error(error_msg)
                    return f"Error: {error_msg}"
            except Exception as e:
                error_msg = f"Error generating embedding: {e}"
                logger.error(error_msg)
                return f"Error: {error_msg}"
            
            # No need to resize embedding as we're using text-embedding-3-large which outputs 3072 dimensions
            # which matches our Pinecone index
            
            # Search Pinecone for relevant documents
            # Increase top_k for table-heavy content to get more context
            top_k = 15 if 'table' in question.lower() or 'tier' in question.lower() else 10
            try:
                search_results = self._search_pinecone(query_embedding, top_k=top_k)
                
                if not search_results or not search_results.get('matches'):
                    # Generate a helpful response even if no exact matches are found
                    logger.info("No exact matches found, generating general response")
                    return self._generate_general_response(enhanced_query)
                
                # Extract context from search results with table awareness
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Pinecone search failed: {error_msg}")
                return f"Error: {error_msg}"
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
            
            if not text or len(text.strip()) == 0:
                raise ValueError("Empty text provided for embedding generation")
                
            logger.info(f"Generating embedding with model: text-embedding-3-large")
            
            # Direct API call - not using LightRAG to ensure reliability
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            
            if not response or not response.data or len(response.data) == 0:
                raise ValueError("Empty response from OpenAI embedding API")
                
            embedding = response.data[0].embedding
            logger.info(f"Successfully generated embedding with {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise Exception(f"Failed to generate embedding: {str(e)}")
            
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
        # Check if Pinecone is available
        if self.index is None:
            error_msg = "Pinecone index is not available. Cannot perform search."
            logger.warning(error_msg)
            raise Exception(error_msg)
            
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            return results
        except Exception as e:
            error_msg = f"Error searching Pinecone: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
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
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating direct response: {e}")
            return "I'm sorry, I encountered an error while generating a response. Please try again."
    
    def _generate_fallback_response(self, query):
        """Generate a fallback response when Pinecone is unavailable"""
        try:
            # Use OpenAI to generate a response without RAG context
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": f"I need information about {query}. Please provide a general response about pharmacy formularies and how they work, since our database connection is currently unavailable. Make sure to format your response according to the structured format guidelines with emojis and clear sections."}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating fallback response: {e}")
            return f"I'm sorry, I couldn't process your query due to a technical issue: {e}"
            
    def _generate_general_response(self, query):
        """Generate a helpful general response when no exact matches are found"""
        try:
            # Use OpenAI to generate a helpful response based on the query
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": f"I need information about {query}. Even though we don't have exact matches in our database, please provide a helpful, informative response about this topic related to pharmacy formularies, insurance coverage, or medication access. Use your general knowledge to give useful information to a nurse. Make sure to format your response according to the structured format guidelines with emojis and clear sections."}
                ],
                temperature=0.5,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating general response: {e}")
            return f"I'm sorry, I couldn't process your query due to a technical issue: {e}"

    def direct_query(self, query):
        """Query the agent directly with OpenAI without using RAG"""
        try:
            logger.info(f"Starting direct query for: {query[:100]}...")
            
            # Prepare a general prompt that works for any query type
            prompt = f"""
            Please provide information about the following pharmacy formulary question:
            
            QUESTION: {query}
            
            IMPORTANT INSTRUCTIONS:
            1. Use your general knowledge about pharmacy formularies and medication coverage
            2. Format your response according to the system instructions
            3. If you don't have specific information, provide general guidance
            4. Be helpful and informative even with limited information
            """
            
            logger.info("Generating direct response using OpenAI...")
            # Generate completion using OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1500
            )
            
            logger.info("Direct query response generated successfully")
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in direct_query: {e}")
            raise Exception(f"Direct query failed: {str(e)}")
    
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
