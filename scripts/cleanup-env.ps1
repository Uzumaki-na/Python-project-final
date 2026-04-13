# List of files to remove
$filesToRemove = @(
    "backend\.env",
    ".env.production"
)

# Remove old env files
foreach ($file in $filesToRemove) {
    if (Test-Path $file) {
        Write-Host "Removing $file..."
        Remove-Item $file
    }
}

# Create new .env from example if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "Creating new .env from .env.example..."
    Copy-Item ".env.example" ".env"
}

Write-Host "Environment cleanup complete!"
