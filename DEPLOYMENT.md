# Render Deployment Guide

## Files Required for Deployment

✅ **requirements.txt** - Python dependencies (CPU-optimized for Render)
✅ **runtime.txt** - Python version specification (3.11.5)
✅ **Procfile** - Process configuration for Render
✅ **wsgi.py** - WSGI entry point
✅ **render.yaml** - Render service configuration (optional)

## Deployment Steps

### 1. Push to GitHub
```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### 2. Create Render Service
1. Go to [render.com](https://render.com)
2. Connect your GitHub repository
3. Create a new **Web Service**
4. Select your repository: `AI-Powered-Tuberculosis-Detection`

### 3. Configure Build Settings
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn --bind 0.0.0.0:$PORT wsgi:app`
- **Python Version**: `3.11.5` (set via PYTHON_VERSION environment variable)

### 4. Environment Variables
Set these in Render dashboard:
- `PYTHON_VERSION`: `3.11.5`
- `FLASK_ENV`: `production`
- `PORT`: `10000` (Render will set this automatically)

### 5. Deploy
Click **Deploy** and wait for build to complete.

## Troubleshooting

### Build Fails with "requirements.txt not found"
- Ensure requirements.txt is in the root directory
- Check file permissions and encoding
- Verify the file is committed to git

### Memory Issues
- The requirements.txt uses CPU-only TensorFlow to reduce memory usage
- Model size is optimized for Render's free tier

### Unicode Errors
- All emoji characters have been replaced with text labels
- Logging should work correctly on Render's Linux environment

## File Structure
```
/
├── requirements.txt     # Python dependencies
├── runtime.txt         # Python version
├── Procfile           # Process configuration
├── wsgi.py           # WSGI entry point
├── render.yaml       # Render configuration (optional)
├── backend/          # Backend application
├── frontend/         # Frontend templates
└── README.md         # Project documentation
```
