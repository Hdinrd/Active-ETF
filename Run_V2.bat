@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
call C:\Users\User\anaconda3\Scripts\activate.bat C:\Users\User\anaconda3\envs\stockenv

REM  Usage:  Run_V2.bat              daily update (first run backfills automatically)
REM          Run_V2.bat --backfill   re-download full history
REM          Run_V2.bat --notify     also push to Telegram (needs .env)

python -m etf_tracker.run %*
if exist data\reports\latest.md start "" notepad data\reports\latest.md
pause
