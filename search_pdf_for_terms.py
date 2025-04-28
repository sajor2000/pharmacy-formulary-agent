"""
Search the CountyCare PDF for specific medication terms
"""
import os
import fitz  # PyMuPDF
import re

def search_pdf_for_terms(pdf_path, search_terms):
    """Search a PDF for specific terms and return context around matches"""
    print(f"Searching {os.path.basename(pdf_path)} for: {', '.join(search_terms)}")
    
    # Open the PDF
    doc = fitz.open(pdf_path)
    results = {}
    
    # Search each page for each term
    for term in search_terms:
        term_results = []
        for page_num, page in enumerate(doc):
            # Search for the term (case insensitive)
            text = page.get_text()
            pattern = re.compile(r'(?i)' + re.escape(term))
            matches = pattern.finditer(text)
            
            for match in matches:
                # Get context around the match (200 chars before and after)
                start = max(0, match.start() - 200)
                end = min(len(text), match.end() + 200)
                context = text[start:end]
                
                # Add to results
                term_results.append({
                    'page': page_num + 1,
                    'context': context
                })
        
        results[term] = term_results
    
    doc.close()
    return results

def main():
    """Main function to search PDFs for medication terms"""
    # Path to the CountyCare PDF
    pdf_path = os.path.join('data', 'CountyCare.0209.1Q2024.pdf')
    
    # Terms to search for
    search_terms = [
        'Advair', 
        'Symbicort', 
        'Breo', 
        'fluticasone/salmeterol',
        'budesonide/formoterol',
        'ICS-LABA'
    ]
    
    # Search the PDF
    results = search_pdf_for_terms(pdf_path, search_terms)
    
    # Print results
    for term, matches in results.items():
        print(f"\n{'-' * 80}")
        print(f"Results for '{term}': {len(matches)} matches")
        print(f"{'-' * 80}")
        
        for i, match in enumerate(matches):
            print(f"\nMatch {i+1} (Page {match['page']}):")
            print(f"{match['context']}")
            print(f"{'-' * 40}")

if __name__ == "__main__":
    main()
