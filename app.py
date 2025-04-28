#!/usr/bin/env python3
"""
Pharmacy Formulary RAG Web Application
---------------------------------
A web interface for the Pharmacy Formulary RAG Agent with table-aware extraction
and structured, nurse-friendly responses.
"""

import os
import logging
import time
from flask import Flask, render_template, request, jsonify
from pharmacy_lightrag_agent import PharmacyFormularyAgent
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize the pharmacy formulary agent
agent = None

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    """Process a query from the user"""
    global agent
    
    # Initialize agent if not already done
    if agent is None:
        try:
            agent = PharmacyFormularyAgent()
        except Exception as e:
            return jsonify({'error': str(e)})
    
    # Get the query from the request
    data = request.json
    question = data.get('question', '')
    insurance = data.get('insurance', None)
    medication_class = data.get('medication_class', None)
    
    if not question:
        return jsonify({'error': 'No question provided'})
    
    # Enhance the query with structured information if provided
    enhanced_query = question
    if insurance:
        enhanced_query += f"\nInsurance/PBM name: {insurance}"
    if medication_class:
        enhanced_query += f"\nMedication class needed: {medication_class}"
    
    # Process the query
    try:
        logger.info(f"Processing query: '{enhanced_query}' with insurance: {insurance}, medication_class: {medication_class}")
        
        # Add detailed logging
        logger.info("Step 1: Starting query processing")
        
        # Try a direct query first to see if that works
        try:
            logger.info("Attempting direct query without RAG...")
            direct_response = agent.direct_query(enhanced_query)
            logger.info("Direct query successful")
            return jsonify({'response': direct_response, 'method': 'direct'})
        except Exception as direct_error:
            logger.warning(f"Direct query failed: {direct_error}. Falling back to RAG query.")
        
        # If direct query fails, try the regular RAG query
        logger.info("Step 2: Attempting regular RAG query")
        response = agent.query(enhanced_query, insurance, medication_class)
        
        logger.info("Query processed successfully")
        return jsonify({'response': response, 'method': 'rag'})
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing query: {error_msg}")
        
        # Try one last fallback with a simple completion
        try:
            logger.info("Attempting emergency fallback response...")
            if agent and agent.openai_client:
                fallback = agent.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful pharmacy formulary assistant."},
                        {"role": "user", "content": f"Please provide a general response about: {enhanced_query}"}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                fallback_response = fallback.choices[0].message.content
                logger.info("Emergency fallback successful")
                return jsonify({'response': fallback_response, 'method': 'emergency_fallback'})
        except Exception as fallback_error:
            logger.error(f"Emergency fallback also failed: {fallback_error}")
        
        return jsonify({'error': error_msg})

@app.route('/medication_tier', methods=['POST'])
def medication_tier():
    """Get the tier for a specific medication"""
    global agent
    
    # Initialize agent if not already done
    if agent is None:
        try:
            agent = PharmacyFormularyAgent()
        except Exception as e:
            return jsonify({'error': str(e)})
    
    # Get the parameters from the request
    data = request.json
    medication_name = data.get('medication_name', '')
    insurance_provider = data.get('insurance_provider', None)
    
    if not medication_name:
        return jsonify({'error': 'No medication name provided'})
    
    # Process the query
    try:
        response = agent.get_medication_tier(medication_name, insurance_provider)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/lower_tier_alternatives', methods=['POST'])
def lower_tier_alternatives():
    """Find lower tier alternatives for a medication"""
    global agent
    
    # Initialize agent if not already done
    if agent is None:
        try:
            agent = PharmacyFormularyAgent()
        except Exception as e:
            return jsonify({'error': str(e)})
    
    # Get the parameters from the request
    data = request.json
    medication_name = data.get('medication_name', '')
    insurance_provider = data.get('insurance_provider', None)
    
    if not medication_name:
        return jsonify({'error': 'No medication name provided'})
    
    # Process the query
    try:
        response = agent.find_lower_tier_alternatives(medication_name, insurance_provider)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/prior_authorization', methods=['POST'])
def prior_authorization():
    """Check if a medication requires prior authorization"""
    global agent
    
    # Initialize agent if not already done
    if agent is None:
        try:
            agent = PharmacyFormularyAgent()
        except Exception as e:
            return jsonify({'error': str(e)})
    
    # Get the parameters from the request
    data = request.json
    medication_name = data.get('medication_name', '')
    insurance_provider = data.get('insurance_provider', None)
    
    if not medication_name:
        return jsonify({'error': 'No medication name provided'})
    
    # Process the query
    try:
        response = agent.check_prior_authorization(medication_name, insurance_provider)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/compare_coverage', methods=['POST'])
def compare_coverage():
    """Compare coverage for multiple medications"""
    global agent
    
    # Initialize agent if not already done
    if agent is None:
        try:
            agent = PharmacyFormularyAgent()
        except Exception as e:
            return jsonify({'error': str(e)})
    
    # Get the parameters from the request
    data = request.json
    medication_names = data.get('medication_names', [])
    insurance_provider = data.get('insurance_provider', None)
    
    if not medication_names:
        return jsonify({'error': 'No medication names provided'})
    
    # Process the query
    try:
        response = agent.compare_coverage(medication_names, insurance_provider)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'agent_initialized': agent is not None
    })

@app.route('/status')
def status():
    """Status page with information about the application"""
    global agent
    
    # Initialize agent if not already done
    if agent is None:
        try:
            agent = PharmacyFormularyAgent()
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'agent_initialized': False
            })
    
    # Get environment information
    env_info = {
        'openai_api_key': 'configured' if os.getenv('OPENAI_API_KEY') else 'missing',
        'pinecone_api_key': 'configured' if os.getenv('PINECONE_API_KEY') else 'missing',
        'pinecone_environment': os.getenv('PINECONE_ENVIRONMENT', 'not configured'),
        'llamaparse_api_key': 'configured' if os.getenv('LLAMAPARSE_API_KEY') else 'not configured (optional)'
    }
    
    return jsonify({
        'status': 'ok',
        'agent_initialized': True,
        'environment': env_info,
        'version': '1.0.0'
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Get port from environment variable for Render compatibility
    # Default to 5006 for local development
    port = int(os.environ.get("PORT", 5006))
    
    logger.info(f"Starting Pharmacy Formulary RAG Agent on port {port}")
    
    # Run the app with host='0.0.0.0' to make it publicly accessible
    app.run(debug=False, port=port, host='0.0.0.0')
