#!/usr/bin/env python3
"""
Evaluate LightRAG Response Format
--------------------------------
A script to evaluate if LightRAG responses follow the expected pharmacy formulary format.
"""

import os
import sys
import json
from dotenv import load_dotenv
from pharmacy_lightrag_agent import PharmacyFormularyAgent

# Load environment variables
load_dotenv()

# Define test queries with expected format elements
TEST_QUERIES = [
    {
        "name": "Comprehensive Tier Query",
        "query": "What tier is Advair in BCBS insurance and what are the alternatives?",
        "expected_format_elements": [
            "Primary Recommendation",
            "Alternative Options",
            "Coverage Notes"
        ]
    },
    {
        "name": "Medication Comparison Query",
        "query": "Compare Symbicort and Advair for asthma treatment coverage on Cigna",
        "expected_format_elements": [
            "Primary Recommendation",
            "Alternative Options",
            "Coverage Notes"
        ]
    },
    {
        "name": "Prior Authorization Requirements",
        "query": "What are the prior authorization requirements for Trelegy on Aetna?",
        "expected_format_elements": [
            "Requirements",
            "Prior authorization",
            "Coverage Notes"
        ]
    }
]

def evaluate_format(response):
    """Evaluate if the response follows the expected format"""
    # Format elements to check
    format_elements = [
        "Primary Recommendation",
        "Alternative Options",
        "Coverage Notes",
        "Verification Note"
    ]
    
    # Check which format elements are present
    present_elements = []
    for element in format_elements:
        if element in response:
            present_elements.append(element)
    
    # Calculate format score
    format_score = len(present_elements) / len(format_elements) * 100
    
    return {
        "present_elements": present_elements,
        "missing_elements": [e for e in format_elements if e not in present_elements],
        "format_score": format_score,
        "follows_format": format_score >= 75  # At least 75% of format elements present
    }

def main():
    """Main function to evaluate LightRAG response format"""
    print("=== Evaluating LightRAG Response Format ===\n")
    
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
            
            # Evaluate format
            format_evaluation = evaluate_format(response)
            
            # Print results
            print(f"\nFormat Score: {format_evaluation['format_score']:.1f}%")
            print(f"Follows Expected Format: {'Yes' if format_evaluation['follows_format'] else 'No'}")
            print("\nPresent Format Elements:")
            for element in format_evaluation["present_elements"]:
                print(f"- {element}")
            
            if format_evaluation["missing_elements"]:
                print("\nMissing Format Elements:")
                for element in format_evaluation["missing_elements"]:
                    print(f"- {element}")
            
            print("\nResponse:")
            print("-" * 80)
            print(response)
            print("-" * 80)
            
            # Store results
            results.append({
                "query_info": query_info,
                "response": response,
                "format_evaluation": format_evaluation
            })
        
        # Calculate overall format score
        overall_format_score = sum(r["format_evaluation"]["format_score"] for r in results) / len(results)
        
        print("\n=== Overall Format Evaluation ===")
        print(f"Overall Format Score: {overall_format_score:.1f}%")
        print(f"All Responses Follow Expected Format: {'Yes' if all(r['format_evaluation']['follows_format'] for r in results) else 'No'}")
        
        # Save results to file
        with open("lightrag_format_evaluation.json", "w") as f:
            json.dump({
                "overall": {
                    "overall_format_score": overall_format_score,
                    "all_follow_format": all(r["format_evaluation"]["follows_format"] for r in results)
                },
                "query_results": results
            }, f, indent=2)
        
        print("\nResults saved to lightrag_format_evaluation.json")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
