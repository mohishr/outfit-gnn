#!/bin/bash
# Run the NGNN Outfit Generation Web App

cd "$(dirname "$0")"
cd ..

echo "========================================"
echo "  NGNN Outfit Intelligence"
echo "========================================"
echo ""
echo "Starting Flask server..."
echo "Open http://localhost:5000 in your browser"
echo ""

python app/app.py
