@echo off
REM run-validation.bat — Quick launcher for PowerShell validation script
REM Windows batch file wrapper for validate-submission.ps1
REM Double-click this file to run the hackathon validation

title Content Moderation OpenEnv - Validation
cls

echo.
echo ================================================
echo Content Moderation OpenEnv
echo Hackathon Validation Script
echo ================================================
echo.

REM Run PowerShell script with proper execution policy
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0validate-submission.ps1"

REM Keep window open so user can see results
pause
