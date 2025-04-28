"""
Test the pharmacy agent with specific medication queries
"""
import os
from dotenv import load_dotenv
from pharmacy_lightrag_agent import PharmacyFormularyAgent

# Load environment variables
load_dotenv()

def main():
    """Test the pharmacy agent with specific medication queries"""
    print("Testing pharmacy agent with specific medication queries...")
    
    # Initialize the agent
    agent = PharmacyFormularyAgent()
    
    # Test query for Advair in CountyCare
    query = "Is Advair covered by CountyCare? What tier is it in?"
    
    print(f"\nQuery: {query}\n")
    print("-" * 80)
    
    # Get response from the agent
    response = agent.query(query)
    
    print("\nResponse:")
    print("-" * 80)
    print(response)
    print("-" * 80)
    
    # Test query for Symbicort in CountyCare
    query = "Is Symbicort covered by CountyCare? What are the requirements?"
    
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
