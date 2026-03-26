Write-Host "Disabling WSL features..." -ForegroundColor Yellow

dism.exe /online /disable-feature /featurename:Microsoft-Windows-Subsystem-Linux /norestart
dism.exe /online /disable-feature /featurename:VirtualMachinePlatform /norestart

Write-Host "Done. Please restart your PC." -ForegroundColor Green
Pause
