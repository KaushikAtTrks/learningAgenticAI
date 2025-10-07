#!/bin/bash

echo "=========================================="
echo "Research Assistant Setup & Run"
echo "=========================================="

# Navigate to project root
cd "/home/dreamworld/Documents/Trakshym/Learning AgenticAI"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Navigate to Research Assistant folder
cd "Research Assistant"

# Check if research_db exists and has data
if [ ! -d "research_db" ] || [ ! -f "research_db/chroma.sqlite3" ]; then
    echo ""
    echo "=========================================="
    echo "Step 1: Creating embeddings and populating database..."
    echo "=========================================="
    python create_embedding.py
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create embeddings"
        exit 1
    fi
else
    echo ""
    echo "Database already exists. Skipping embedding creation."
    echo "To recreate, delete the 'research_db' folder first."
fi

# Run the intelligent RAG system
echo ""
echo "=========================================="
echo "Step 2: Starting Intelligent RAG System..."
echo "=========================================="
python intelligent_RAG.py

echo ""
echo "=========================================="
echo "Session ended. Deactivating virtual environment..."
echo "=========================================="
deactivate
