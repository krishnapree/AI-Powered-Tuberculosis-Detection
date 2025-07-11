#!/bin/bash
echo "=== Build Debug Information ==="
echo "Current directory: $(pwd)"
echo "Files in current directory:"
ls -la
echo "Looking for requirements.txt:"
find . -name "requirements.txt" -type f
echo "Contents of requirements.txt (if found):"
if [ -f "requirements.txt" ]; then
    cat requirements.txt
else
    echo "requirements.txt not found in current directory"
fi
echo "=== Installing dependencies ==="
pip install -r requirements.txt
