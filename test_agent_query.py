"""
Test the pharmacy formulary agent with a query about CountyCare ICS-LABA coverage
"""
import os
from dotenv import load_dotenv
from pharmacy_lightrag_agent import PharmacyFormularyAgent

# Load environment variables
load_dotenv()

def main():
    """Test the pharmacy agent with a query"""
    print("Testing pharmacy formulary agent...")
    
    # Initialize the agent
    agent = PharmacyFormularyAgent()
    
    # Test query about CountyCare ICS-LABA coverage
    query = "What is the CountyCare coverage for ICS-LABA inhalers like Advair, Symbicort, and Breo Ellipta?"
    
    print(f"\nQuery: {query}\n")
    print("-" * 80)
    
    # Get response from the agent
    response = agent.query(query)
    
    print("\nResponse:")
    print("-" * 80)
    print(response)
    print("-" * 80)

if __name__ == "__main__":
    main()
