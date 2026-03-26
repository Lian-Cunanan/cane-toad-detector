Write-Host "Removing Linux icon namespace entries..." -ForegroundColor Yellow

reg delete "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\MyComputer\NameSpace\DelegateFolders\{b155bdf8-02f0-451e-9a26-ae317cfd7779}" /f
reg delete "HKLM\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Explorer\MyComputer\NameSpace\DelegateFolders\{b155bdf8-02f0-451e-9a26-ae317cfd7779}" /f
reg add "HKLM\SOFTWARE\Classes\CLSID\{B155BDF8-02F0-451E-9A26-AE317CFD7779}" /v "System.IsPinnedToNameSpaceTree" /t REG_DWORD /d 0 /f

Write-Host "Done. Restart Explorer or reboot PC." -ForegroundColor Green
Pause
