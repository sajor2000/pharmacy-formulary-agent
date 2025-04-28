#!/usr/bin/env python3
"""
Deployment Readiness Check Script
--------------------------------
Verifies that all required components are properly configured for deployment.
"""

import os
import sys
import requests
from dotenv import load_dotenv
import importlib.util
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_env_variables():
    """Check that all required environment variables are set"""
    load_dotenv()
    
    required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY', 'PINECONE_ENVIRONMENT']
    optional_vars = ['LLAMAPARSE_API_KEY', 'PORT', 'FLASK_ENV']
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    logger.info("✅ All required environment variables are set")
    
    # Check optional variables
    for var in optional_vars:
        if os.getenv(var):
            logger.info(f"✅ Optional variable {var} is set")
        else:
            logger.info(f"ℹ️ Optional variable {var} is not set")
    
    return True

def check_dependencies():
    """Check that all required dependencies are installed"""
    required_modules = [
        'openai', 'pinecone', 'flask', 'gunicorn', 'python-dotenv',
        'pymupdf', 'camelot-py', 'pandas', 'numpy', 'tabulate'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            # Handle special case for python-dotenv
            if module == 'python-dotenv':
                module = 'dotenv'
            importlib.util.find_spec(module)
            logger.info(f"✅ Module {module} is installed")
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"❌ Missing required modules: {', '.join(missing_modules)}")
        return False
    
    logger.info("✅ All required dependencies are installed")
    return True

def check_pinecone_connection():
    """Check that Pinecone connection is working"""
    try:
        from pinecone import Pinecone
        
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            logger.error("❌ PINECONE_API_KEY is not set")
            return False
        
        # Initialize with the new API structure only
        pc = Pinecone(api_key=api_key)
        
        # Try to list indexes
        indexes = pc.list_indexes()
        logger.info(f"✅ Successfully connected to Pinecone. Available indexes: {indexes}")
        
        # Check if finalpharm index exists
        if any(index.name == "finalpharm" for index in indexes):
            logger.info("✅ finalpharm index exists")
            
            # Try to connect to the index
            index = pc.Index("finalpharm")
            stats = index.describe_index_stats()
            logger.info(f"✅ Connected to finalpharm index. Vector count: {stats.get('total_vector_count', 0)}")
            return True
        else:
            logger.error("❌ finalpharm index does not exist")
            return False

    
    except Exception as e:
        logger.error(f"❌ Error connecting to Pinecone: {str(e)}")
        return False

def check_openai_connection():
    """Check that OpenAI connection is working"""
    try:
        from openai import OpenAI
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("❌ OPENAI_API_KEY is not set")
            return False
        
        client = OpenAI(api_key=api_key)
        
        # Try a simple completion
        response = client.embeddings.create(
            input="Test connection",
            model="text-embedding-3-large"
        )
        
        if response and hasattr(response, 'data') and len(response.data) > 0:
            logger.info("✅ Successfully connected to OpenAI API")
            return True
        else:
            logger.error("❌ Failed to get a valid response from OpenAI API")
            return False
    
    except Exception as e:
        logger.error(f"❌ Error connecting to OpenAI: {str(e)}")
        return False

def check_flask_app():
    """Check that Flask app is properly configured"""
    try:
        # Check if app.py exists
        if not os.path.exists('app.py'):
            logger.error("❌ app.py does not exist")
            return False
        
        # Check if templates directory exists
        if not os.path.exists('templates'):
            logger.error("❌ templates directory does not exist")
            return False
        
        # Check if index.html exists
        if not os.path.exists('templates/index.html'):
            logger.error("❌ templates/index.html does not exist")
            return False
        
        # Check if static directory exists
        if not os.path.exists('static'):
            logger.error("❌ static directory does not exist")
            return False
        
        logger.info("✅ Flask app is properly configured")
        return True
    
    except Exception as e:
        logger.error(f"❌ Error checking Flask app: {str(e)}")
        return False

def check_render_config():
    """Check that Render configuration is properly set up"""
    try:
        # Check if render.yaml exists
        if not os.path.exists('render.yaml'):
            logger.error("❌ render.yaml does not exist")
            return False
        
        # Check if Procfile exists
        if not os.path.exists('Procfile'):
            logger.error("❌ Procfile does not exist")
            return False
        
        logger.info("✅ Render configuration is properly set up")
        return True
    
    except Exception as e:
        logger.error(f"❌ Error checking Render configuration: {str(e)}")
        return False

def main():
    """Run all checks"""
    logger.info("Starting deployment readiness check...")
    
    checks = [
        ("Environment Variables", check_env_variables),
        ("Dependencies", check_dependencies),
        ("Pinecone Connection", check_pinecone_connection),
        ("OpenAI Connection", check_openai_connection),
        ("Flask App", check_flask_app),
        ("Render Configuration", check_render_config)
    ]
    
    results = []
    for name, check_func in checks:
        logger.info(f"\n--- Checking {name} ---")
        result = check_func()
        results.append((name, result))
    
    # Print summary
    logger.info("\n=== SUMMARY ===")
    all_passed = True
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        if not result:
            all_passed = False
        logger.info(f"{name}: {status}")
    
    if all_passed:
        logger.info("\n✅ All checks passed! The application is ready for deployment.")
        return 0
    else:
        logger.error("\n❌ Some checks failed. Please fix the issues before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
