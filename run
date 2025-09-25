#!/bin/bash

# Auto-grader API run script
# Usage: ./run [install|URL_FILE|test]

set -e

if [ $# -eq 0 ]; then
    echo "Usage: ./run [install|URL_FILE|test]" >&2
    exit 1
fi

ARG="$1"

case "$ARG" in
    "install")
        echo "Installing dependencies..."
        
        pip install requests
        echo "Dependencies installed successfully."
        exit 0
        ;;
    "test")
        echo "Running test suite..."
        
        if [ -f "testUrls.txt" ]; then
            echo "Testing with sample URLs..."
            python -m src.main testUrls.txt
        else
            echo "Error: testUrls.txt not found"
            exit 1
        fi
        
        exit 0
        ;;
    *)
        URL_FILE="$ARG"
        echo "Processing URL file: $URL_FILE"

        if [ ! -f "$URL_FILE" ]; then
            echo "Error: File '$URL_FILE' not found"
            exit 1
        fi

        python -m src.main "$URL_FILE"
        exit 0;
        ;;
esac
