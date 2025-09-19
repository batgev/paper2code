# Run backend API and frontend dev servers
# Usage: Right click > Run with PowerShell (or: pwsh -File .\run_all.ps1)

$ErrorActionPreference = "Stop"

# Start API (serves frontend/dist if exists)
Write-Host "Starting API on http://127.0.0.1:8000 ..."
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "-m","paper2code.api.server"

Start-Sleep -Seconds 2

# Start Vite dev server (front-end)
Write-Host "Starting frontend dev on http://127.0.0.1:5173 ..."
Push-Location frontend
npm run dev
Pop-Location
