web: gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 300 --max-requests 10 --max-requests-jitter 2 --worker-class sync --worker-connections 100 --preload --log-level error wsgi:app
