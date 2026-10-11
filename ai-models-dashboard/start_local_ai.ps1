$ErrorActionPreference = 'Stop'
$runtimeDirectory = Join-Path $PSScriptRoot '.local-ai'
$ollamaExecutable = Join-Path $runtimeDirectory 'ollama.exe'
if (-not (Test-Path -LiteralPath $ollamaExecutable)) { throw 'Ollama runtime is missing.' }
$env:OLLAMA_MODELS = Join-Path $runtimeDirectory 'models'
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$localModelName = if ($env:OLLAMA_MODEL) { $env:OLLAMA_MODEL } else { 'qwen3:1.7b' }
$localServerReady = $false
try { Invoke-RestMethod 'http://127.0.0.1:11434/api/version' -TimeoutSec 2 | Out-Null; $localServerReady = $true } catch {}
if (-not $localServerReady) {
    Start-Process -FilePath $ollamaExecutable -ArgumentList 'serve' -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtimeDirectory 'server.out.log') -RedirectStandardError (Join-Path $runtimeDirectory 'server.err.log') | Out-Null
    for ($attemptNumber = 0; $attemptNumber -lt 20; $attemptNumber++) {
        Start-Sleep -Milliseconds 500
        try { Invoke-RestMethod 'http://127.0.0.1:11434/api/version' -TimeoutSec 2 | Out-Null; $localServerReady = $true; break } catch {}
    }
}
if (-not $localServerReady) { throw 'Ollama did not start. Check .local-ai/server.err.log.' }
$installedModels = Invoke-RestMethod 'http://127.0.0.1:11434/api/tags'
if ($localModelName -notin $installedModels.models.name) {
    & $ollamaExecutable pull $localModelName
    if ($LASTEXITCODE -ne 0) { throw 'The model download failed.' }
}
Write-Output ('Local AI is ready: ' + $localModelName)
