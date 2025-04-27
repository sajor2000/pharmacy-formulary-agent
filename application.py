#!/usr/bin/env python3
"""
Pharmacy Formulary Agent - AWS Elastic Beanstalk Entry Point
"""
from app import app as application

# AWS Elastic Beanstalk expects a variable named 'application'
if __name__ == "__main__":
    application.run(host='0.0.0.0', port=5000)
