# Create new directory structure
$dirs = @(
    "frontend/public",
    "frontend/src",
    "deployment/nginx",
    "deployment/docker",
    "backend/ml/models",
    "backend/ml/datasets",
    "backend/ml/training"
)

foreach ($dir in $dirs) {
    New-Item -Path $dir -ItemType Directory -Force
    Write-Host "Created directory: $dir"
}

# Move frontend files
$frontendFiles = @(
    "src",
    "index.html",
    "package.json",
    "package-lock.json",
    "tsconfig.json",
    "tsconfig.app.json",
    "tsconfig.node.json",
    "vite.config.ts",
    "eslint.config.js",
    "postcss.config.js",
    "tailwind.config.js"
)

foreach ($file in $frontendFiles) {
    if (Test-Path $file) {
        if (Test-Path "frontend/$file") {
            Remove-Item -Recurse -Force "frontend/$file"
        }
        Move-Item -Path $file -Destination "frontend/" -Force
        Write-Host "Moved to frontend: $file"
    }
}

# Move node_modules if it exists
if (Test-Path "node_modules") {
    if (Test-Path "frontend/node_modules") {
        Remove-Item -Recurse -Force "frontend/node_modules"
    }
    Move-Item -Path "node_modules" -Destination "frontend/" -Force
    Write-Host "Moved node_modules to frontend/"
}

# Move ML-related files
$mlFiles = @(
    @{Source="*.pth"; Destination="backend/ml/models/"},
    @{Source="datasets/*"; Destination="backend/ml/datasets/"},
    @{Source="train_models_torch.py"; Destination="backend/ml/training/"}
)

foreach ($file in $mlFiles) {
    if (Test-Path $file.Source) {
        Move-Item -Path $file.Source -Destination $file.Destination -Force
        Write-Host "Moved to $($file.Destination): $($file.Source)"
    }
}

# Move deployment files
$deploymentFiles = @(
    @{Source="nginx.conf"; Destination="deployment/nginx/"},
    @{Source="Dockerfile"; Destination="deployment/docker/"},
    @{Source="docker-compose.yml"; Destination="deployment/docker/"}
)

foreach ($file in $deploymentFiles) {
    if (Test-Path $file.Source) {
        Move-Item -Path $file.Source -Destination $file.Destination -Force
        Write-Host "Moved to $($file.Destination): $($file.Source)"
    }
}

# Move backend files
if (Test-Path "auth.py") {
    Move-Item -Path "auth.py" -Destination "backend/app/auth/" -Force
    Write-Host "Moved auth.py to backend/app/auth/"
}

if (Test-Path "main.py") {
    Move-Item -Path "main.py" -Destination "backend/" -Force
    Write-Host "Moved main.py to backend/"
}

# Clean up unnecessary files
$cleanupFiles = @(
    "copy_of_untitled11.py",
    "venv310",
    "__pycache__"
)

foreach ($file in $cleanupFiles) {
    if (Test-Path $file) {
        Remove-Item -Recurse -Force $file
        Write-Host "Removed: $file"
    }
}

Write-Host "Project restructuring complete!"
