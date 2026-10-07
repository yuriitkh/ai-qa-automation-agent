$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$desktop = [Environment]::GetFolderPath('Desktop')
$powershell = Join-Path $PSHOME 'powershell.exe'
$shell = New-Object -ComObject WScript.Shell

$shortcuts = @(
    @{ Name = 'AI QA Agent'; Script = Join-Path $PSScriptRoot 'start-ai-qa-agent.ps1' },
    @{ Name = 'Stop AI QA Agent'; Script = Join-Path $PSScriptRoot 'stop-ai-qa-agent.ps1' }
)
foreach ($item in $shortcuts) {
    $shortcutPath = Join-Path $desktop ($item.Name + '.lnk')
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $powershell
    $shortcut.Arguments = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $item.Script + '"'
    $shortcut.WorkingDirectory = $repoRoot
    $shortcut.Description = $item.Name + ' local launcher'
    $shortcut.Save()
}
Write-Output "Created AI QA Agent shortcuts on the Desktop: $desktop"
