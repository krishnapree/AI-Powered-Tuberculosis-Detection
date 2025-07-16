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

# CRITICAL MEMORY FIX: Don't import TensorFlow at startup - only when needed
# This prevents 200MB+ memory usage at startup
try:
    # Set TensorFlow environment variables BEFORE any import
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    # Import TB service without loading TensorFlow yet
    from services.tb_detection.tensorflow_tb_service import tb_bp
    TB_SERVICE_AVAILABLE = True
    TB_SERVICE_TYPE = "tensorflow"
    print("[SUCCESS] TB Detection service registered (TensorFlow lazy-loaded)")
except ImportError as e:
    print(f"TensorFlow service not available: {e}")
    try:
        from services.tb_detection.mock_tb_service import tb_bp
        TB_SERVICE_AVAILABLE = True
        TB_SERVICE_TYPE = "mock"
        print("[WARNING] TB Detection service loaded with Mock")
    except ImportError as e2:
        print(f"[ERROR] TB Detection service not available: {e2}")
        TB_SERVICE_AVAILABLE = False
except Exception as e:
    print(f"Error loading TB service: {e}")
    try:
        from services.tb_detection.mock_tb_service import tb_bp
        TB_SERVICE_AVAILABLE = True
        TB_SERVICE_TYPE = "mock"
        print("[WARNING] Fallback to Mock service due to error")
    except Exception as e3:
        print(f"[ERROR] All TB services failed: {e3}")
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

# Configure logging (with error handling for deployment)
try:
    os.makedirs('logs', exist_ok=True)
    log_handlers = [
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
except (PermissionError, OSError):
    # If we can't write to logs directory, just use stdout
    log_handlers = [logging.StreamHandler(sys.stdout)]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)
logger = logging.getLogger(__name__)

def create_app(config_name='default'):
    """Application factory pattern"""
    # Set up Flask app with frontend template and static directories
    # Handle both local development and deployment paths

    # Creating Flask app for deployment

    # Try multiple possible paths for templates and static files
    possible_template_paths = [
        '../frontend/templates',  # Local development from backend/
        'frontend/templates',     # Deployment from root
        './frontend/templates',   # Alternative deployment path
    ]

    possible_static_paths = [
        '../frontend/static',     # Local development from backend/
        'frontend/static',        # Deployment from root
        './frontend/static',      # Alternative deployment path
        '../frontend/public',     # Try public folder as static
        'frontend/public',        # Public folder from root
        './frontend/public',      # Alternative public path
    ]

    template_folder = None
    static_folder = None

    # Find the correct template folder
    for path in possible_template_paths:
        if os.path.exists(path):
            template_folder = os.path.abspath(path)  # Use absolute path
            break

    # Find the correct static folder
    for path in possible_static_paths:
        if os.path.exists(path):
            static_folder = os.path.abspath(path)  # Use absolute path
            break

    if not template_folder:
        print("Warning: No template folder found, using default")
        template_folder = 'templates'

    if not static_folder:
        print("Warning: No static folder found, using default")
        static_folder = 'static'

    app = Flask(__name__,
                template_folder=template_folder,
                static_folder=static_folder)
    app.config.from_object(config[config_name])

    # Enable CORS for frontend communication (allow all origins for development)
    CORS(app, resources={
        r"/*": {
            "origins": ["*"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })

    # Ensure required directories exist (with error handling for deployment)
    try:
        os.makedirs('logs', exist_ok=True)
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('data', exist_ok=True)
    except PermissionError:
        # On some deployment platforms, we might not have write permissions
        # Log the issue but continue
        print("Warning: Could not create directories (permission denied)")
    except Exception as e:
        print(f"Warning: Directory creation failed: {e}")

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

    # Frontend routes - ensure proper routing for Render deployment
    @app.route('/')
    @app.route('/index.html')
    @app.route('/home')
    def landing():
        """Landing page with authentication - handles multiple routes"""
        return render_template('landing.html')

    @app.route('/dashboard')
    def dashboard():
        """Main TB detection platform dashboard"""
        # Handle both local development and deployment paths
        possible_public_paths = [
            '../frontend/public',  # Local development from backend/
            'frontend/public',     # Deployment from root
            './frontend/public',   # Alternative deployment path
        ]

        public_folder = None
        for path in possible_public_paths:
            if os.path.exists(path):
                public_folder = os.path.abspath(path)
                break

        if public_folder and os.path.exists(os.path.join(public_folder, 'index.html')):
            return send_from_directory(public_folder, 'index.html')
        else:
            # Fallback: render a simple dashboard template if public folder not found
            return render_template('tb_detection.html')

    @app.route('/tb-detection')
    def tb_detection():
        """TB Detection page"""
        return render_template('tb_detection.html')

    @app.route('/ai-assistant')
    def ai_assistant():
        """AI Assistant page"""
        return render_template('ai_assistant.html')

    # Add favicon route with healthcare-themed icon
    @app.route('/favicon.ico')
    def favicon():
        """Serve healthcare-themed favicon"""
        # Create a simple healthcare-themed favicon using SVG
        favicon_svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32">
            <circle cx="16" cy="16" r="15" fill="#2563eb" stroke="#1e40af" stroke-width="2"/>
            <path d="M12 8h8v4h4v8h-4v4h-8v-4H8v-8h4V8z" fill="white"/>
            <circle cx="16" cy="16" r="3" fill="#2563eb"/>
        </svg>'''

        response = app.response_class(
            favicon_svg,
            mimetype='image/svg+xml'
        )
        response.headers['Cache-Control'] = 'public, max-age=86400'  # Cache for 1 day
        return response

    # Add health check route for deployment monitoring
    @app.route('/health')
    @app.route('/healthcheck')
    def simple_health_check():
        """Simple health check endpoint for deployment monitoring"""
        return jsonify({
            'status': 'healthy',
            'service': 'TB Detection Platform',
            'timestamp': datetime.now().isoformat()
        })

    # Add static file serving for production
    @app.route('/static/<path:filename>')
    def static_files(filename):
        """Serve static files"""
        # Handle both local development and deployment paths
        possible_static_paths = [
            '../frontend/static',  # Local development
            'frontend/static',     # Deployment
            './frontend/static',   # Alternative
            '../frontend/public',  # Public assets
            'frontend/public',     # Public assets deployment
        ]

        for static_path in possible_static_paths:
            if os.path.exists(static_path):
                try:
                    return send_from_directory(static_path, filename)
                except:
                    continue

        # If file not found in any static directory, return 404
        return '', 404

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
        """Memory monitoring endpoint with automatic cleanup"""
        import psutil
        import sys

        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = round(memory_info.rss / 1024 / 1024, 2)

        # Automatic memory cleanup if usage is high
        if memory_mb > 300:  # If memory usage > 300MB
            logger.warning(f"High memory usage detected: {memory_mb}MB - triggering cleanup")
            try:
                # Force aggressive garbage collection
                for _ in range(3):
                    gc.collect()

                # Try to clean up TB detector if available
                try:
                    from services.tb_detection.tensorflow_tb_service import force_memory_cleanup
                    force_memory_cleanup()
                except:
                    pass

                # Get updated memory info
                memory_info = process.memory_info()
                memory_mb = round(memory_info.rss / 1024 / 1024, 2)
                logger.info(f"Memory after cleanup: {memory_mb}MB")
            except Exception as e:
                logger.error(f"Error during automatic memory cleanup: {e}")

        return jsonify({
            'memory_usage_mb': memory_mb,
            'memory_percent': round(process.memory_percent(), 2),
            'memory_status': 'critical' if memory_mb > 400 else 'warning' if memory_mb > 250 else 'normal',
            'render_limit_mb': 512,
            'available_mb': round(512 - memory_mb, 2),
            'python_version': sys.version.split()[0],
            'gc_count': len(gc.get_objects())
        })

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({'error': 'Internal server error'}), 500

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({'error': 'Bad request'}), 400

    @app.errorhandler(502)
    def bad_gateway(error):
        logger.error(f"Bad gateway error: {error}")
        return jsonify({
            'error': 'Service temporarily unavailable. Please try again in a moment.',
            'suggestion': 'The AI model may be loading. Please wait a moment and try again.'
        }), 502

    @app.errorhandler(503)
    def service_unavailable(error):
        logger.error(f"Service unavailable: {error}")
        return jsonify({
            'error': 'Service temporarily unavailable. Please try again in a moment.',
            'suggestion': 'The AI model may be loading. Please wait a moment and try again.'
        }), 503

    @app.errorhandler(504)
    def gateway_timeout(error):
        logger.error(f"Gateway timeout: {error}")
        return jsonify({
            'error': 'Request timed out. Please try again.',
            'suggestion': 'The analysis is taking longer than expected. Please try again.'
        }), 504

    return app

# Only create app once to prevent memory issues and URL redirect problems
if __name__ == '__main__':
    # Local development mode
    app = create_app(os.environ.get('FLASK_ENV', 'development'))
    logger.info("Healthcare Portal Backend API Server starting...")
    logger.info("API endpoints available at http://localhost:5000/api/")
    logger.info("Health check: http://localhost:5000/api/health")

    port = int(os.environ.get('PORT', 5001))
    host = os.environ.get('HOST', '0.0.0.0')

    app.run(
        host=host,
        port=port,
        debug=app.config.get('DEBUG', False),
        threaded=True
    )
else:
    # WSGI deployment mode (Render, Gunicorn, etc.)
    gc.collect()  # Force garbage collection before creating app
    app = create_app(os.environ.get('FLASK_ENV', 'production'))
