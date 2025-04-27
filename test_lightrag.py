#!/usr/bin/env python3
"""
Test LightRAG Answer Quality
---------------------------
A script to test the quality of LightRAG answers for pharmacy formulary queries.
"""

import os
import sys
import json
from dotenv import load_dotenv
from pharmacy_lightrag_agent import PharmacyFormularyAgent

# Load environment variables
load_dotenv()

# Test queries for different scenarios
TEST_QUERIES = [
    {
        "name": "Basic Tier Query",
        "query": "What tier is Advair in BCBS insurance?",
        "expected_elements": ["tier", "Advair", "BCBS"]
    },
    {
        "name": "Prior Authorization Query",
        "query": "Does Trelegy require prior authorization for Cigna?",
        "expected_elements": ["prior authorization", "Trelegy", "Cigna"]
    },
    {
        "name": "Alternative Medication Query",
        "query": "What are lower cost alternatives to Symbicort for asthma?",
        "expected_elements": ["alternative", "Symbicort", "cost", "asthma"]
    },
    {
        "name": "Medication Class Query",
        "query": "What ICS-LABA combinations are covered by Aetna?",
        "expected_elements": ["ICS-LABA", "Aetna", "covered"]
    },
    {
        "name": "Quantity Limit Query",
        "query": "What is the quantity limit for Ventolin HFA on UnitedHealthcare?",
        "expected_elements": ["quantity limit", "Ventolin", "UnitedHealthcare"]
    }
]

def evaluate_response(query_info, response):
    """Evaluate the quality of a response"""
    score = 0
    max_score = len(query_info["expected_elements"])
    
    # Check if response contains expected elements
    for element in query_info["expected_elements"]:
        if element.lower() in response.lower():
            score += 1
    
    # Calculate percentage score
    percentage = (score / max_score) * 100
    
    return {
        "score": score,
        "max_score": max_score,
        "percentage": percentage,
        "contains_all_elements": score == max_score
    }

def main():
    """Main function to test LightRAG answer quality"""
    print("=== Testing LightRAG Answer Quality ===\n")
    
    try:
        # Initialize the pharmacy formulary agent
        print("Initializing agent...")
        agent = PharmacyFormularyAgent()
        
        results = []
        
        # Test each query
        for i, query_info in enumerate(TEST_QUERIES, 1):
            print(f"\n{i}. Testing: {query_info['name']}")
            print(f"Query: {query_info['query']}")
            
            # Get response from agent
            response = agent.query(query_info["query"])
            
            # Evaluate response
            evaluation = evaluate_response(query_info, response)
            
            # Print results
            print(f"Score: {evaluation['score']}/{evaluation['max_score']} ({evaluation['percentage']:.1f}%)")
            print(f"Contains all expected elements: {'Yes' if evaluation['contains_all_elements'] else 'No'}")
            print("\nResponse Preview:")
            print("-" * 80)
            print(response[:500] + "..." if len(response) > 500 else response)
            print("-" * 80)
            
            # Store results
            results.append({
                "query_info": query_info,
                "response": response,
                "evaluation": evaluation
            })
        
        # Calculate overall score
        total_score = sum(r["evaluation"]["score"] for r in results)
        total_max = sum(r["evaluation"]["max_score"] for r in results)
        overall_percentage = (total_score / total_max) * 100
        
        print("\n=== Overall Results ===")
        print(f"Total Score: {total_score}/{total_max} ({overall_percentage:.1f}%)")
        
        # Save results to file
        with open("lightrag_test_results.json", "w") as f:
            json.dump({
                "overall": {
                    "total_score": total_score,
                    "total_max": total_max,
                    "overall_percentage": overall_percentage
                },
                "query_results": results
            }, f, indent=2)
        
        print("\nResults saved to lightrag_test_results.json")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
