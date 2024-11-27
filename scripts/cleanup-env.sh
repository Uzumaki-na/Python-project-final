#!/bin/bash

# List of files to remove
files_to_remove=(
    "backend/.env"
    ".env.production"
)

# Remove old env files
for file in "${files_to_remove[@]}"; do
    if [ -f "$file" ]; then
        echo "Removing $file..."
        rm "$file"
    fi
done

# Create new .env from example if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating new .env from .env.example..."
    cp .env.example .env
fi

echo "Environment cleanup complete!"
