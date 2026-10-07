$ErrorActionPreference = 'Stop'

function Show-StopMessage([string]$Message, [string]$Kind = 'Information') {
    Write-Output $Message
    try {
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.MessageBox]::Show($Message, 'AI QA Agent', 'OK', $Kind) | Out-Null
    } catch { }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pidPath = Join-Path $repoRoot '.runtime\ai-qa-agent.pid.json'
if (-not (Test-Path -LiteralPath $pidPath -PathType Leaf)) {
    Show-StopMessage 'No launcher-owned AI QA Agent process is recorded.'
    exit 0
}

$record = $null
try {
    $record = Get-Content -LiteralPath $pidPath -Raw | ConvertFrom-Json
} catch {
    Remove-Item -LiteralPath $pidPath -Force
    Show-StopMessage 'The malformed stale PID file was removed. No process was stopped.' 'Warning'
    exit 0
}

try {
    $processId = [int]$record.process_id
    if ($processId -lt 1 -or -not $record.python_path -or -not $record.launcher_token) {
        Remove-Item -LiteralPath $pidPath -Force
        Show-StopMessage 'The stale AI QA Agent PID file was removed. No process was stopped.'
        exit 0
    }
    try {
        $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction Stop
    } catch {
        Show-StopMessage 'The process identity could not be checked safely. The PID file was kept and no process was stopped.' 'Warning'
        exit 1
    }
    if (-not $processInfo) {
        Remove-Item -LiteralPath $pidPath -Force
        Show-StopMessage 'The AI QA Agent process has already exited. Its stale PID file was removed.'
        exit 0
    }
    $expectedPython = [System.IO.Path]::GetFullPath([string]$record.python_path)
    $actualPython = [System.IO.Path]::GetFullPath([string]$processInfo.ExecutablePath)
    $command = [string]$processInfo.CommandLine
    $tokenPattern = '--launcher-instance\s+' + [Regex]::Escape([string]$record.launcher_token)
    $owned = $actualPython.Equals($expectedPython, [StringComparison]::OrdinalIgnoreCase) `
        -and $command.Contains('-m qa_agent.web') `
        -and $command.Contains('--port 8000') `
        -and [Regex]::IsMatch($command, $tokenPattern)
    if (-not $owned) {
        Remove-Item -LiteralPath $pidPath -Force
        Show-StopMessage 'The PID no longer identifies the AI QA Agent instance started by this launcher. The PID file was cleared; the process was left running.' 'Warning'
        exit 0
    }
    Stop-Process -Id $processId -Force
    Remove-Item -LiteralPath $pidPath -Force
    Show-StopMessage 'The launcher-owned AI QA Agent process was stopped.'
} catch {
    Show-StopMessage "AI QA Agent could not be stopped safely: $($_.Exception.Message). No other process was targeted." 'Warning'
    exit 1
}
