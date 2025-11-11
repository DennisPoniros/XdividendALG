@echo off
REM Script to push Dividend Algorithm to GitHub
REM Repository: https://github.com/DennisPoniros/XdividendALG

echo ==========================================
echo Pushing Dividend Algorithm to GitHub
echo ==========================================
echo.

set REPO_URL=https://github.com/DennisPoniros/XdividendALG.git

REM Check if git is installed
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo X Git is not installed. Please install git first.
    echo Download from: https://git-scm.com/download/win
    pause
    exit /b 1
)

echo Current directory: %CD%
echo.

REM Initialize git if not already initialized
if not exist ".git" (
    echo Initializing git repository...
    git init
    git remote add origin %REPO_URL%
) else (
    echo Git repository already initialized
)

REM Create .gitignore if it doesn't exist
if not exist ".gitignore" (
    echo Creating .gitignore...
    (
        echo # Python
        echo __pycache__/
        echo *.py[cod]
        echo *$py.class
        echo *.so
        echo .Python
        echo build/
        echo dist/
        echo *.egg-info/
        echo.
        echo # Virtual Environment
        echo venv/
        echo ENV/
        echo env/
        echo.
        echo # IDEs
        echo .vscode/
        echo .idea/
        echo *.swp
        echo.
        echo # Data files
        echo *.csv
        echo *.xlsx
        echo.
        echo # Outputs
        echo outputs/*.png
        echo outputs/*.html
        echo outputs/*.csv
        echo !outputs/.gitkeep
        echo.
        echo # API Keys
        echo config_local.py
        echo secrets.py
        echo .env
        echo.
        echo # OS
        echo .DS_Store
        echo Thumbs.db
        echo.
        echo # Jupyter
        echo .ipynb_checkpoints/
        echo *.ipynb
        echo.
        echo # Logs
        echo *.log
    ) > .gitignore
)

REM Create outputs directory placeholder
if not exist "outputs" mkdir outputs
if not exist "outputs\.gitkeep" type nul > outputs\.gitkeep

REM Stage all files
echo Staging files...
git add .

REM Show what will be committed
echo.
echo Files to be committed:
git status --short

echo.
set /p CONTINUE="Continue with commit? (y/n): "
if /i not "%CONTINUE%"=="y" (
    echo Commit cancelled
    pause
    exit /b 1
)

REM Commit
echo Committing files...
git commit -m "Initial commit: Comprehensive dividend capture trading algorithm - Complete backtesting framework - Real data integration - Advanced risk management - Full documentation"

REM Push to GitHub
echo Pushing to GitHub...
echo You may be prompted for GitHub credentials...
echo.

git branch -M main
git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ==========================================
    echo SUCCESS! Code pushed to GitHub
    echo ==========================================
    echo.
    echo View your repository at:
    echo https://github.com/DennisPoniros/XdividendALG
    echo.
) else (
    echo.
    echo Push failed. Common solutions:
    echo.
    echo 1. Use GitHub Personal Access Token instead of password
    echo    Generate at: https://github.com/settings/tokens
    echo.
    echo 2. Create repository first at: https://github.com/new
    echo    Name: XdividendALG
    echo.
)

pause
