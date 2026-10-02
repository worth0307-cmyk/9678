@echo off
setlocal
title moomoo strategy tool
rem Double-click to run. Put it on the desktop or anywhere you like.
rem The code always goes to %USERPROFILE%\9678-claude-btcusdt-trading-system-hl6119
rem First run downloads the code and installs the Python packages.
rem
rem This file is ASCII on purpose: cmd misreads UTF-8 batch files that contain
rem Chinese (lines get split and run as commands). The Chinese menu lives in
rem run.py instead; exit code 99 from the menu means "update the code".

set "REPO=%USERPROFILE%\9678-claude-btcusdt-trading-system-hl6119"
set "RUN=%REPO%\moomoo_strat\run.py"
set "ZIP=https://github.com/worth0307-cmyk/9678/archive/refs/heads/claude/btcusdt-trading-system-hl6119.zip"

where python >nul 2>nul
if errorlevel 1 (
  echo Python not found. Install it from python.org and tick "Add python.exe to PATH" on the first page.
  pause
  exit /b 1
)
if not exist "%RUN%" goto update
rem Code downloaded before the menu existed: update first.
findstr /c:"def cmd_menu" "%RUN%" >nul 2>nul || goto update

python "%RUN%" menu
if errorlevel 99 goto update
if errorlevel 1 pause
exit /b 0

:update
rem The whole block is read before it runs, so overwriting this very file
rem during the update is safe. When done, the menu reopens in a new window.
(
  echo Downloading the latest code to %REPO% ...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference='SilentlyContinue'; $z=Join-Path $env:TEMP '9678.zip'; Invoke-WebRequest -Uri '%ZIP%' -OutFile $z; Expand-Archive -Path $z -DestinationPath $env:USERPROFILE -Force"
  if errorlevel 1 (
    echo Download failed. Check the network and try again.
    pause
    exit /b 1
  )
  echo Installing Python packages ...
  python -m pip install -q --disable-pip-version-check -r "%REPO%\moomoo_strat\requirements.txt"
  echo Done. Your cache and out folders are kept.
  start "" "%~f0"
  exit /b 0
)
