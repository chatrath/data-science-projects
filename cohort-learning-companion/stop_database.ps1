$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    & (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') -m src.local_database stop
    if ($LASTEXITCODE -ne 0) { throw 'Could not stop the local database.' }
} finally { Pop-Location }
