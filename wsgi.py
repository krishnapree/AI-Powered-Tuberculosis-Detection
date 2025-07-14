#!/usr/bin/env python3
"""
WSGI entry point for Render deployment
"""
import sys
import os
import logging

# Add backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Configure basic logging for WSGI
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

try:
    # Import the Flask app
    from app import create_app

    # Create the application
    app = create_app(os.environ.get('FLASK_ENV', 'production'))

    print("WSGI application created successfully")

except Exception as e:
    print(f"Error creating WSGI application: {e}")
    import traceback
    traceback.print_exc()
    raise

if __name__ == "__main__":
    app.run()
