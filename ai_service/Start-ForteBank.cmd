@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-ForteBank.ps1" %*
set "FORTE_EXIT_CODE=%ERRORLEVEL%"
if not "%FORTE_EXIT_CODE%"=="0" (
    echo.
    echo ForteBank could not start. Read the error above.
    pause
)
exit /b %FORTE_EXIT_CODE%
