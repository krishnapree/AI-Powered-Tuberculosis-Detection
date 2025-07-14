#!/usr/bin/env python3
"""
Tuberculosis Detection Platform - Backend API Server
===================================================

An advanced AI-powered tuberculosis detection platform with two core services:
1. AI-Powered Tuberculosis Detection (99.84% accuracy)
2. AI Health Assistant (Google Gemini powered)

This is the backend server designed for frontend/backend separation with user authentication.
Memory optimized for deployment on Render.

Author: TB Detection Platform Team
Version: 3.0.1
License: MIT
"""

import os
import sys
import logging
import gc
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS

# Import configuration
from config import config

# Import service blueprints
# Try TensorFlow service first (production), then fallback to mock (local testing)
TB_SERVICE_AVAILABLE = False
TB_SERVICE_TYPE = "none"

try:
    from services.tb_detection.tensorflow_tb_service import tb_bp
    TB_SERVICE_AVAILABLE = True
    TB_SERVICE_TYPE = "tensorflow"
    print("[SUCCESS] TB Detection service loaded with TensorFlow (Production)")
except ImportError as e:
    print(f"TensorFlow not available: {e}")
    try:
        from services.tb_detection.mock_tb_service import tb_bp
        TB_SERVICE_AVAILABLE = True
        TB_SERVICE_TYPE = "mock"
        print("[WARNING] TB Detection service loaded with Mock (Local Testing Only)")
        print("   Note: Mock service provides random predictions for testing UI/UX")
        print("   Production deployment will use real TensorFlow model")
    except ImportError as e2:
        print(f"[ERROR] TB Detection service not available: {e2}")
        TB_SERVICE_AVAILABLE = False

# Heart Rate Monitoring service removed
HR_SERVICE_AVAILABLE = False

try:
    from services.ai_assistant.ai_health_assistant import ai_health_bp
    AI_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"AI Assistant service not available: {e}")
    AI_SERVICE_AVAILABLE = False

try:
    from services.auth.auth_service import auth_bp
    AUTH_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"Auth service not available: {e}")
    AUTH_SERVICE_AVAILABLE = False

# Configure logging
import os
os.makedirs('logs', exist_ok=True)  # Create logs directory if it doesn't exist

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def create_app(config_name='default'):
    """Application factory pattern"""
    # Set up Flask app with frontend template and static directories
    app = Flask(__name__,
                template_folder='../frontend/templates',
                static_folder='../frontend/static')
    app.config.from_object(config[config_name])

    # Enable CORS for frontend communication (allow all origins for development)
    CORS(app, resources={
        r"/*": {
            "origins": ["*"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })

    # Ensure required directories exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Track service availability
    services_status = {
        'tb_detection': TB_SERVICE_AVAILABLE,
        'ai_assistant': AI_SERVICE_AVAILABLE,
        'authentication': AUTH_SERVICE_AVAILABLE
    }

    # Register service blueprints
    if TB_SERVICE_AVAILABLE:
        app.register_blueprint(tb_bp, url_prefix='/api/tb-detection')
        logger.info("[SUCCESS] TB Detection service registered")

    # Heart Rate Monitoring service removed

    if AI_SERVICE_AVAILABLE:
        app.register_blueprint(ai_health_bp, url_prefix='/api/ai-assistant')
        logger.info("[SUCCESS] AI Assistant service registered")

    if AUTH_SERVICE_AVAILABLE:
        app.register_blueprint(auth_bp, url_prefix='/api/auth')
        logger.info("[SUCCESS] Authentication service registered")

    # Frontend routes
    @app.route('/')
    def landing():
        """Landing page with authentication"""
        return render_template('landing.html')

    @app.route('/dashboard')
    def dashboard():
        """Main TB detection platform dashboard"""
        return send_from_directory('../frontend/public', 'index.html')

    @app.route('/tb-detection')
    def tb_detection():
        """TB Detection page"""
        return render_template('tb_detection.html')

    @app.route('/ai-assistant')
    def ai_assistant():
        """AI Assistant page"""
        return render_template('ai_assistant.html')

    @app.route('/tb-detection')
    def tb_detection_page():
        """TB Detection service page"""
        return render_template('tb_detection.html')

    # Heart Rate Monitoring page removed

    @app.route('/ai-assistant')
    def ai_assistant_page():
        """AI Assistant service page"""
        return render_template('ai_assistant.html')

    # Register API routes
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'services': services_status,
            'tb_service_type': TB_SERVICE_TYPE,
            'timestamp': str(datetime.now())
        })

    # Simple test endpoints for now
    @app.route('/api/tb-detection/test', methods=['GET'])
    def tb_test():
        return jsonify({'service': 'TB Detection', 'status': 'available'})

    # Heart Rate Monitoring test endpoint removed

    @app.route('/api/ai-assistant/test', methods=['GET'])
    def ai_test():
        return jsonify({'service': 'AI Assistant', 'status': 'available'})

    @app.route('/api/memory-status', methods=['GET'])
    def memory_status():
        """Memory monitoring endpoint for debugging"""
        import psutil
        import sys

        process = psutil.Process()
        memory_info = process.memory_info()

        return jsonify({
            'memory_usage_mb': round(memory_info.rss / 1024 / 1024, 2),
            'memory_percent': round(process.memory_percent(), 2),
            'python_version': sys.version,
            'gc_count': len(gc.get_objects())
        })

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({'error': 'Bad request'}), 400

    return app

if __name__ == '__main__':
    # Create the Flask app
    app = create_app(os.environ.get('FLASK_ENV', 'default'))

    # Run the application
    logger.info("Healthcare Portal Backend API Server starting...")
    logger.info("API endpoints available at http://localhost:5000/api/")
    logger.info("Health check: http://localhost:5000/api/health")

    # Get port from environment (for Render deployment)
    port = int(os.environ.get('PORT', 5001))
    host = os.environ.get('HOST', '0.0.0.0')  # Changed to 0.0.0.0 for deployment

    # Memory optimization
    gc.collect()

    app.run(
        host=host,
        port=port,
        debug=app.config.get('DEBUG', False),
        threaded=True  # Enable threading for better performance
    )

# For Gunicorn deployment (memory optimized)
gc.collect()  # Force garbage collection
app = create_app(os.environ.get('FLASK_ENV', 'default'))
