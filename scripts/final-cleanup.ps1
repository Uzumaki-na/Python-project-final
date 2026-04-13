# Move database to backend/data
New-Item -Path "backend/data" -ItemType Directory -Force
if (Test-Path "health_assessment.db") {
    Move-Item -Path "health_assessment.db" -Destination "backend/data/" -Force
    Write-Host "Moved database to backend/data/"
}

# Remove empty directories
$emptyDirs = @(
    "datasets",
    "models",
    "ml",
    ".bolt"
)

foreach ($dir in $emptyDirs) {
    if (Test-Path $dir) {
        Remove-Item -Recurse -Force $dir
        Write-Host "Removed empty directory: $dir"
    }
}

# Update .gitignore
$gitignoreContent = @"
# Dependencies
node_modules/
venv/
__pycache__/
*.pyc

# Environment
.env
.env.local
.env.*.local

# Build
dist/
build/
*.egg-info/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Database
*.db
*.sqlite3

# ML Models
*.pth
*.h5
*.onnx

# Cache
.pytest_cache/
.coverage
htmlcov/

# System
.DS_Store
Thumbs.db
"@

Set-Content -Path ".gitignore" -Value $gitignoreContent
Write-Host "Updated .gitignore"

Write-Host "Final cleanup complete!"
