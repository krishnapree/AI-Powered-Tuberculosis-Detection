#!/bin/bash

# Build script for Render deployment
echo "Starting build process..."

# Upgrade pip first
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p backend/uploads
mkdir -p backend/logs
mkdir -p backend/models

echo "Build completed successfully!"
