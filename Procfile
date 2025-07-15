web: gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 300 --max-requests 50 --max-requests-jitter 5 --worker-class sync --worker-tmp-dir /dev/shm --preload --log-level info wsgi:app
