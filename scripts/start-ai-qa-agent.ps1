$ErrorActionPreference = 'Stop'

function Show-LauncherError([string]$Message) {
    [Console]::Error.WriteLine($Message)
    try {
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.MessageBox]::Show($Message, 'AI QA Agent', 'OK', 'Error') | Out-Null
    } catch { }
    exit 1
}

$startMutex = [System.Threading.Mutex]::new($false, 'Local\AIQAAgentLauncherStart')
$mutexAcquired = $false
try {
    $mutexAcquired = $startMutex.WaitOne(30000)
    if (-not $mutexAcquired) {
        Show-LauncherError 'Another AI QA Agent launcher is still checking or starting the server. Try again in a few seconds.'
    }
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
    $pythonCandidates = @(
        (Join-Path $repoRoot '.venv\Scripts\python.exe'),
        (Join-Path $repoRoot '.venv-original\Scripts\python.exe')
    )
    $python = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    if (-not $python) {
        Show-LauncherError "No project Python environment was found. Create .venv or restore .venv-original, then run scripts\start-ai-qa-agent.ps1 again."
    }
    if (-not (Test-Path -LiteralPath (Join-Path $repoRoot 'qa_agent\web.py') -PathType Leaf)) {
        Show-LauncherError 'The application files are missing from the repository folder.'
    }

    $url = 'http://127.0.0.1:8000'
    $healthUrl = "$url/health"
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 2
        if ($health.status -eq 'ok' -and $health.service -eq 'ai-qa-agent') {
            Start-Process $url
            exit 0
        }
    } catch { }

    $runtime = Join-Path $repoRoot '.runtime'
    $pidPath = Join-Path $runtime 'ai-qa-agent.pid.json'
    if (Test-Path -LiteralPath $pidPath -PathType Leaf) {
        $record = $null
        try { $record = Get-Content -LiteralPath $pidPath -Raw | ConvertFrom-Json } catch { }
        if ($record -and $record.process_id -and $record.python_path -and $record.launcher_token) {
            try {
                $recordedProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$record.process_id)" -ErrorAction Stop
            } catch {
                Show-LauncherError 'The existing launcher process could not be verified safely. No additional server was started.'
            }
            if ($recordedProcess) {
                $expectedPython = [System.IO.Path]::GetFullPath([string]$record.python_path)
                $actualPython = [System.IO.Path]::GetFullPath([string]$recordedProcess.ExecutablePath)
                $tokenPattern = '--launcher-instance\s+' + [Regex]::Escape([string]$record.launcher_token)
                $owned = $actualPython.Equals($expectedPython, [StringComparison]::OrdinalIgnoreCase) `
                    -and ([string]$recordedProcess.CommandLine).Contains('-m qa_agent.web') `
                    -and ([string]$recordedProcess.CommandLine).Contains('--port 8000') `
                    -and [Regex]::IsMatch([string]$recordedProcess.CommandLine, $tokenPattern)
                if ($owned) {
                    for ($attempt = 0; $attempt -lt 40; $attempt++) {
                        Start-Sleep -Milliseconds 500
                        try {
                            $health = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 2
                            if ($health.status -eq 'ok' -and $health.service -eq 'ai-qa-agent') {
                                Start-Process $url
                                exit 0
                            }
                        } catch { }
                        try {
                            $recordedProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$record.process_id)" -ErrorAction Stop
                        } catch {
                            Show-LauncherError 'The existing launcher process could not be verified safely. No additional server was started.'
                        }
                        if (-not $recordedProcess) { break }
                    }
                    if ($recordedProcess) {
                        Show-LauncherError 'The launcher-owned server is still starting but has not passed its health check. No duplicate server was started; check .runtime server logs.'
                    }
                }
            }
        }
        # The recorded PID is gone, malformed, or belongs to a different process.
        # Clear only our stale ownership record; never stop the process.
        Remove-Item -LiteralPath $pidPath -Force
    }

    $probe = [System.Net.Sockets.TcpClient]::new()
    try {
        $pending = $probe.BeginConnect('127.0.0.1', 8000, $null, $null)
        if ($pending.AsyncWaitHandle.WaitOne(500) -and $probe.Connected) {
            Show-LauncherError 'Port 8000 is in use by another process. AI QA Agent was not started and no process was stopped.'
        }
    } finally {
        $probe.Dispose()
    }

    New-Item -ItemType Directory -Path $runtime -Force | Out-Null
    $database = Join-Path $repoRoot 'qa_agent.db'
    $evidence = Join-Path $repoRoot '.evidence'
    New-Item -ItemType Directory -Path $evidence -Force | Out-Null
    $token = [Guid]::NewGuid().ToString('N')
    $arguments = '-m qa_agent.web --database "{0}" --evidence-directory "{1}" --host 127.0.0.1 --port 8000 --launcher-instance {2}' -f $database, $evidence, $token
    $stdout = Join-Path $runtime 'server.out.log'
    $stderr = Join-Path $runtime 'server.err.log'
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot `
        -PassThru -WindowStyle Minimized -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $pidRecord = @{
        process_id = $process.Id
        python_path = [System.IO.Path]::GetFullPath($python)
        launcher_token = $token
        started_at_utc = [DateTime]::UtcNow.ToString('o')
    } | ConvertTo-Json -Compress
    $temporaryPidPath = "$pidPath.tmp"
    Set-Content -LiteralPath $temporaryPidPath -Value $pidRecord -Encoding UTF8
    Move-Item -LiteralPath $temporaryPidPath -Destination $pidPath -Force

    $ready = $false
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        Start-Sleep -Milliseconds 500
        if ($process.HasExited) { break }
        try {
            $health = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 2
            if ($health.status -eq 'ok' -and $health.service -eq 'ai-qa-agent') {
                $ready = $true
                break
            }
        } catch { }
    }
    if (-not $ready) {
        $outPath = Join-Path $runtime 'server.out.log'
        $errPath = Join-Path $runtime 'server.err.log'
        Show-LauncherError "AI QA Agent did not become healthy on 127.0.0.1:8000. Check server logs at $outPath and $errPath. If the process is still running, use Stop AI QA Agent."
    }
    Start-Process $url
} catch {
    Show-LauncherError "AI QA Agent could not be started: $($_.Exception.Message)"
} finally {
    if ($mutexAcquired) { $startMutex.ReleaseMutex() }
    $startMutex.Dispose()
}
