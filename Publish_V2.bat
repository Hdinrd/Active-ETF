@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM  Push the v2 pipeline (code + data) to GitHub. Only these paths are added,
REM  your old V*.py files and .env are left alone.

REM  1) install the GitHub Actions workflow (kept in ci\ until you run this)
if exist ".github\workflows\test.tmp" del ".github\workflows\test.tmp"
if not exist ".github\workflows" mkdir ".github\workflows"
if exist "ci\etf-daily.yml" copy /Y "ci\etf-daily.yml" ".github\workflows\etf-daily.yml" >nul

REM  2) commit and push
for %%P in (.gitignore .env.example .github etf_tracker README_V2.md Run_V2.bat Publish_V2.bat data) do (
  if exist "%%P" git add "%%P"
)
git commit -m "v2: pocket.tw API pipeline, flow-adjusted active changes, GitHub Actions schedule"
git pull --rebase --autostash
git push
echo.
echo Done. On GitHub open Actions - "Active ETF daily" - Run workflow to test it now.
pause
