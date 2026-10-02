@echo off
chcp 65001 >nul
setlocal
title moomoo strategy tool
rem 双击运行。放在桌面或任何地方都行：代码固定下载到 %USERPROFILE%\9678-claude-btcusdt-trading-system-hl6119
rem 第一次运行会自动下载代码、装依赖。之后选 9 可以更新。

set "REPO=%USERPROFILE%\9678-claude-btcusdt-trading-system-hl6119"
set "RUN=%REPO%\moomoo_strat\run.py"
set "U80=%REPO%\moomoo_strat\universe\us_large80.txt"
set "ZIP=https://github.com/worth0307-cmyk/9678/archive/refs/heads/claude/btcusdt-trading-system-hl6119.zip"

where python >nul 2>nul
if errorlevel 1 (
  echo 没找到 python。先从 python.org 安装，安装第一页勾上 Add python.exe to PATH，然后重新双击本文件。
  pause
  exit /b 1
)
if not exist "%RUN%" goto update

:menu
cls
echo.
echo   moomoo 策略工具    combo_overlay + impulse_wave
echo   ----------------------------------------------------
echo   先打开 moomoo OpenD 并登录，窗口保持开着
echo.
echo   1  检查连接：OpenD、额度、自选股分组
echo.
echo   2  回测 7姐妹
echo   3  回测 80 只美股大盘股（样本外）
echo   4  回测其他分组（列出分组，按编号选）
echo.
echo   5  今日筛选 7姐妹
echo   6  今日筛选其他分组（按编号选）
echo   7  今日筛选 7姐妹，并把 S2 信号同步进分组「策略信号」（要先在 App 里建好这个空分组）
echo.
echo   8  打开结果文件夹
echo   9  更新代码
echo   0  退出
echo.
set "c="
set /p "c=输入编号后回车："
if "%c%"=="1" (
  python "%RUN%" check
  goto done
)
if "%c%"=="2" (
  python "%RUN%" backtest --group 7姐妹
  goto done
)
if "%c%"=="3" (
  python "%RUN%" backtest --codes-file "%U80%"
  goto done
)
if "%c%"=="4" (
  python "%RUN%" backtest --group ?
  goto done
)
if "%c%"=="5" (
  python "%RUN%" screen --group 7姐妹
  goto done
)
if "%c%"=="6" (
  python "%RUN%" screen --group ?
  goto done
)
if "%c%"=="7" (
  python "%RUN%" screen --group 7姐妹 --to-group 策略信号 --rule S2
  goto done
)
if "%c%"=="8" (
  if not exist "%REPO%\moomoo_strat\out" mkdir "%REPO%\moomoo_strat\out"
  start "" "%REPO%\moomoo_strat\out"
  goto menu
)
if "%c%"=="9" goto update
if "%c%"=="0" exit /b 0
goto menu

:done
echo.
echo 报告和表格在 %REPO%\moomoo_strat\out
pause
goto menu

:update
rem 整段放在括号里：cmd 会一次读完再执行。更新可能覆盖正在运行的这个 bat，
rem 读完再执行就不会读到改了一半的文件；做完在新窗口重新打开菜单。
(
  echo 正在下载最新代码到 %REPO% ……
  powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference='SilentlyContinue'; $z=Join-Path $env:TEMP '9678.zip'; Invoke-WebRequest -Uri '%ZIP%' -OutFile $z; Expand-Archive -Path $z -DestinationPath $env:USERPROFILE -Force"
  if errorlevel 1 (
    echo 下载失败：检查网络后再试
    pause
    exit /b 1
  )
  echo 安装 / 检查 Python 依赖 ……
  python -m pip install -q --disable-pip-version-check -r "%REPO%\moomoo_strat\requirements.txt"
  echo.
  echo 更新完成。cache、out 文件夹里的数据和报告不受影响。
  pause
  start "" "%~f0"
  exit /b 0
)
