param([int]$Port = 8000, [switch]$Watch, [switch]$Check)
$ErrorActionPreference = 'Stop'
$runtimeRoot = Join-Path $PSScriptRoot '.local-db\python'
$uiPython = Join-Path $PSScriptRoot '.venv-ui\Scripts\python.exe'

function Find-ProjectPython {
    if (Test-Path -LiteralPath $runtimeRoot) {
        $candidates = Get-ChildItem -LiteralPath $runtimeRoot -Directory -Filter 'cpython-3.13.*-windows-x86_64-none' |
            Sort-Object Name -Descending
        foreach ($candidate in $candidates) {
            $executable = Join-Path $candidate.FullName 'python.exe'
            if (Test-Path -LiteralPath $executable) { return $executable }
        }
    }
    return $null
}

Push-Location -LiteralPath $PSScriptRoot
try {
    $projectPython = Find-ProjectPython
    if (-not $projectPython) {
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            throw 'uv is required to restore the local UI Python runtime. Install uv and run this launcher again.'
        }
        Write-Host 'Installing project-local Python 3.13...'
        & uv python install 3.13 --install-dir $runtimeRoot --no-bin
        $projectPython = Find-ProjectPython
        if (-not $projectPython) { throw 'Could not install Python 3.13. Check the download error above.' }
    }

    $environmentReady = $false
    if (Test-Path -LiteralPath $uiPython) {
        try {
            $basePrefix = 'import sys; print(sys.base_prefix)' | & $uiPython - 2>$null
            $environmentReady = ($LASTEXITCODE -eq 0 -and
                [string]::Equals([string]$basePrefix, (Split-Path $projectPython), [System.StringComparison]::OrdinalIgnoreCase))
        } catch { $environmentReady = $false }
    }
    if (-not $environmentReady) {
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'uv is required to repair the UI virtual environment.' }
        Write-Host 'Repairing the UI environment with project-local Python...'
        & uv venv --allow-existing --python $projectPython .venv-ui
        if ($LASTEXITCODE -ne 0) { throw 'Could not repair the UI environment. Stop any running Chainlit process and retry.' }
        & uv pip install --python $uiPython -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw 'Could not install UI dependencies.' }
    }

    'import chainlit, langchain_ollama; import sys; print("UI Python:", sys.version.split()[0])' | & $uiPython -
    if ($LASTEXITCODE -ne 0) { throw 'UI dependencies are unavailable. Run uv pip install --python .venv-ui/Scripts/python.exe -r requirements.txt.' }
    if ($Check) { return }

    $launchArguments = @('-m', 'chainlit', 'run', 'app.py', '--host', '127.0.0.1', '--port', "$Port")
    if ($Watch) { $launchArguments += '--watch' }
    & $uiPython @launchArguments
    if ($LASTEXITCODE -ne 0) { throw 'Chainlit stopped with an error. Review the message above.' }
} finally { Pop-Location }
