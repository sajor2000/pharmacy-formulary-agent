#!/usr/bin/env python3
"""
Simple Pharmacy Formulary Agent
-------------------------------
A simplified version of the pharmacy formulary agent that uses Pinecone and OpenAI directly.
"""

import os
import time
from dotenv import load_dotenv
import openai
import pinecone
import numpy as np

# Load environment variables
load_dotenv()

class SimpleFormularyAgent:
    """Simple Pharmacy Formulary Agent"""
    
    def __init__(self):
        """Initialize the pharmacy formulary agent"""
        # Get API keys from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        
        if not all([self.openai_api_key, self.pinecone_api_key]):
            raise ValueError("Missing required API keys in .env file")
        
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        
        # Initialize Pinecone
        pinecone.init(api_key=self.pinecone_api_key)
        
        # Connect to the 'form' index
        self.index = pinecone.Index("form")
    
    def get_embedding(self, text):
        """Get embedding for text"""
        try:
            # Use OpenAI's embedding model
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            
            # Get the embedding vector
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def search_pinecone(self, query_embedding, top_k=5):
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
            print(f"Error searching Pinecone: {e}")
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
    
    def query(self, question):
        """Query the pharmacy formulary agent"""
        try:
            # Get embedding for the query
            query_embedding = self.get_embedding(question)
            if not query_embedding:
                return "Error: Could not generate embedding for the query."
            
            # Search Pinecone for relevant documents
            results = self.search_pinecone(query_embedding)
            if not results:
                return "Error: Could not search the formulary database."
            
            # Format the context from search results
            context = self.format_context(results)
            
            # Prepare the prompt with the specialized instructions
            prompt = f"""
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
            
            # Generate response using OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a pharmacy formulary specialist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error querying pharmacy formulary agent: {e}"

# Example usage
if __name__ == "__main__":
    print("=== Simple Pharmacy Formulary Agent ===")
    
    try:
        agent = SimpleFormularyAgent()
        
        # Example query
        question = "What tier is Advair in BCBS insurance?"
        print(f"\nQuestion: {question}")
        
        response = agent.query(question)
        print(f"\nResponse: {response}")
    
    except Exception as e:
        print(f"Error: {e}")
