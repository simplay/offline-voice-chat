@echo off
setlocal DisableDelayedExpansion
rem Bypass the execution policy for this PowerShell process only; the profile
rem loads so conda is available.
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -ExecutionPolicy Bypass -File "%~dp0run_offline_voice_chat.ps1" %*
exit /b %errorlevel%
