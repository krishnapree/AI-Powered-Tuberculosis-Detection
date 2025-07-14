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
    print("Starting WSGI application...")

    # Import the Flask app
    from app import create_app

    # Create the application
    app = create_app(os.environ.get('FLASK_ENV', 'production'))

    print("WSGI application created successfully")

except Exception as e:
    print(f"Error creating WSGI application: {e}")
    import traceback
    traceback.print_exc()

    # Try to create a minimal Flask app as fallback
    try:
        from flask import Flask
        app = Flask(__name__)

        @app.route('/')
        def error_page():
            return f"Application startup error: {str(e)}", 500

        print("Created fallback Flask app")
    except Exception as e2:
        print(f"Failed to create fallback app: {e2}")
        raise e

if __name__ == "__main__":
    app.run()
