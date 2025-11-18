# Windows Setup Guide - X-Dividend ML Strategy

Complete guide to clone, setup, and run the X-Dividend ML strategy on Windows with VS Code.

---

## Prerequisites

### 1. Install Git for Windows

**Download**: https://git-scm.com/download/win

**Installation**:
1. Download the installer
2. Run installer (use default settings)
3. Verify installation:
   ```cmd
   git --version
   ```

### 2. Install Python 3.8+

**Download**: https://www.python.org/downloads/

**Installation**:
1. Download Python 3.11 or 3.12 (recommended)
2. **IMPORTANT**: Check "Add Python to PATH" during installation
3. Verify installation:
   ```cmd
   python --version
   pip --version
   ```

### 3. Install VS Code

**Download**: https://code.visualstudio.com/

**Installation**:
1. Download and run installer
2. Install recommended extensions:
   - Python (by Microsoft)
   - Jupyter (by Microsoft)
   - GitLens (optional, helpful for git)

---

## Step-by-Step Setup

### Step 1: Clone the Repository

**Option A: Using VS Code** (Easiest)

1. Open VS Code
2. Press `Ctrl+Shift+P` to open Command Palette
3. Type: `Git: Clone`
4. Enter repository URL: `https://github.com/DennisPoniros/XdividendALG.git`
5. Choose a folder location (e.g., `C:\Users\YourName\Projects\`)
6. Click "Open" when prompted

**Option B: Using Command Line**

1. Open Command Prompt or PowerShell
2. Navigate to where you want the project:
   ```cmd
   cd C:\Users\YourName\Projects
   ```
3. Clone the repository:
   ```cmd
   git clone https://github.com/DennisPoniros/XdividendALG.git
   cd XdividendALG
   ```
4. Open in VS Code:
   ```cmd
   code .
   ```

**Option C: Using Git GUI**

1. Open Git GUI or GitHub Desktop
2. Clone repository: `https://github.com/DennisPoniros/XdividendALG.git`
3. Right-click → Open in VS Code

---

### Step 2: Checkout the ML Strategy Branch

The new ML strategy is on a specific branch. In VS Code:

1. Click on branch name in bottom-left corner (should say "main" or "master")
2. Select: `claude/fix-algo-strategy-01UAPSmDv24UxUnvkTW9VvYc`

**Or via command line**:
```cmd
git checkout claude/fix-algo-strategy-01UAPSmDv24UxUnvkTW9VvYc
```

**Verify you have the new files**:
- `strategy_xdiv_ml.py` ✓
- `backtester_xdiv_ml.py` ✓
- `run_xdiv_ml_backtest.py` ✓

---

### Step 3: Create Virtual Environment

**In VS Code Terminal** (View → Terminal or `Ctrl+`):

```cmd
# Create virtual environment
python -m venv venv

# Activate it (Windows Command Prompt)
venv\Scripts\activate

# OR if using PowerShell
venv\Scripts\Activate.ps1

# If PowerShell gives error, run this first as Administrator:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**You'll see `(venv)` in your terminal when activated.**

---

### Step 4: Install Dependencies

```cmd
# Make sure virtual environment is activated (you should see "(venv)")

# Upgrade pip first
python -m pip install --upgrade pip

# Install required packages
pip install pandas numpy scipy yfinance matplotlib seaborn statsmodels scikit-learn

# Optional: Install Alpaca API if you want to use live data
pip install alpaca-trade-api

# Optional: Install Streamlit for dashboard
pip install streamlit plotly
```

**Verify installation**:
```cmd
python -c "import pandas, numpy, scipy, yfinance; print('All packages installed successfully!')"
```

---

### Step 5: Configure API Keys (Optional but Recommended)

**Create `.env` file** in project root:

```cmd
# In VS Code, create new file: .env
# Add your credentials:
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
```

**Update `config.py`**:

Find lines 15-19 and replace with:

```python
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

ALPACA_CONFIG = {
    'API_KEY': os.environ.get('ALPACA_API_KEY', 'PK22RS3CKJJQWEC7IVGI...'),
    'SECRET_KEY': os.environ.get('ALPACA_SECRET_KEY', '9AdsbCdr62JnYT...'),
    'BASE_URL': 'https://paper-api.alpaca.markets'
}
```

**Install python-dotenv**:
```cmd
pip install python-dotenv
```

**⚠️ IMPORTANT**: Never commit `.env` file! (Already in `.gitignore`)

---

### Step 6: Run the Backtest

**Option A: Run from VS Code**

1. Open `run_xdiv_ml_backtest.py`
2. Click "Run Python File" (▶️ button in top-right)
3. Or press `F5` to run with debugger

**Option B: Run from Terminal**

```cmd
# Make sure virtual environment is activated
python run_xdiv_ml_backtest.py
```

**Expected Output**:

```
================================================================================
🚀 X-DIVIDEND ML STRATEGY BACKTEST
================================================================================

PHASE 1: TRAINING
================================================================================
🎓 TRAINING X-DIVIDEND ML STRATEGY
================================================================================
Training Period: 2018-01-01 to 2022-12-31
================================================================================

📊 Step 1/4: Collecting historical dividend data...
📊 Step 2/4: Analyzing 2,456 dividend events...
📊 Step 3/4: Optimizing entry/exit parameters...
📊 Step 4/4: Learning stock-specific patterns...

✅ TRAINING COMPLETED
...
```

**Runtime**: 5-15 minutes (depending on internet speed)

---

### Step 7: View Results

After backtest completes, results are saved to:

**Windows Path**: `<project-folder>\outputs\` or `C:\mnt\user-data\outputs\`

**Files Generated**:
- `xdiv_ml_backtest_results.pkl` - Full results (for dashboard)
- `xdiv_ml_trade_log.csv` - All trades
- `xdiv_ml_equity_curve.csv` - Daily portfolio values
- `xdiv_ml_equity_curve.png` - Performance chart
- `xdiv_ml_drawdown.png` - Drawdown chart
- `xdiv_ml_monthly_returns.png` - Monthly heatmap
- `xdiv_ml_report.html` - Interactive report
- `xdiv_ml_training_summary.txt` - Training results

**View in VS Code**:
1. Open `xdiv_ml_report.html` → Right-click → "Open with Live Server"
2. Open `xdiv_ml_training_summary.txt` to see learned parameters
3. View PNGs directly in VS Code

---

### Step 8: Run the Dashboard (Optional)

```cmd
cd dashboard
streamlit run app.py
```

**Browser opens automatically** to: http://localhost:8501

**To load ML results in dashboard**:

Edit `dashboard/data_interface.py`, change line ~30:

```python
# FROM:
results_file = '/mnt/user-data/outputs/backtest_results.pkl'

# TO:
results_file = '../outputs/xdiv_ml_backtest_results.pkl'
```

---

## Common Issues & Solutions

### Issue 1: "python is not recognized"

**Solution**:
1. Reinstall Python, check "Add to PATH"
2. Or manually add to PATH:
   - Search "Environment Variables" in Windows
   - Edit PATH → Add: `C:\Users\YourName\AppData\Local\Programs\Python\Python311\`

### Issue 2: "pip install" fails

**Solution**:
```cmd
# Try with --user flag
pip install --user pandas numpy scipy

# Or upgrade pip first
python -m pip install --upgrade pip
```

### Issue 3: "No module named 'pandas'"

**Solution**:
```cmd
# Make sure virtual environment is activated!
# You should see (venv) in terminal

# If not activated:
venv\Scripts\activate

# Then reinstall:
pip install pandas
```

### Issue 4: PowerShell execution policy error

**Solution**:
Run PowerShell as Administrator:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue 5: "ModuleNotFoundError: No module named 'yfinance'"

**Solution**:
```cmd
pip install yfinance
```

### Issue 6: Slow download / data fetch

**Solution**:
- Be patient, yfinance downloads 5+ years of data for 100+ tickers
- First run takes 10-15 minutes
- Subsequent runs use cache (faster)

### Issue 7: "Cannot connect to Alpaca API"

**Solution**:
- This is OK! The strategy primarily uses yfinance for data
- Alpaca is optional (for live trading later)
- Script will continue with yfinance data

### Issue 8: Path issues (`/mnt/user-data/`)

**Solution**:

The code uses Linux paths. Update in `run_xdiv_ml_backtest.py`:

```python
# Change:
output_dir = '/mnt/user-data/outputs'

# To:
output_dir = 'outputs'
```

This creates `outputs/` folder in project directory.

---

## VS Code Tips

### Recommended Settings

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "venv/Scripts/python.exe",
    "python.terminal.activateEnvironment": true,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
```

### Useful Keyboard Shortcuts

- `Ctrl+` - Toggle terminal
- `F5` - Run with debugger
- `Ctrl+Shift+P` - Command palette
- `Ctrl+K Ctrl+O` - Open folder
- `Ctrl+P` - Quick file open

### Recommended Extensions

1. **Python** - Python language support
2. **Jupyter** - For notebooks
3. **GitLens** - Advanced git features
4. **Pylance** - Fast Python language server
5. **autoDocstring** - Generate docstrings
6. **Better Comments** - Highlight TODOs

---

## Quick Start Checklist

- [ ] Git installed
- [ ] Python 3.8+ installed
- [ ] VS Code installed
- [ ] Repository cloned
- [ ] Branch checked out (`claude/fix-algo-strategy-01UAPSmDv24UxUnvkTW9VvYc`)
- [ ] Virtual environment created (`python -m venv venv`)
- [ ] Virtual environment activated (`venv\Scripts\activate`)
- [ ] Dependencies installed (`pip install pandas numpy scipy yfinance`)
- [ ] (Optional) `.env` file created with API keys
- [ ] Run backtest (`python run_xdiv_ml_backtest.py`)
- [ ] View results in `outputs/` folder
- [ ] Open `xdiv_ml_report.html` in browser

---

## File Structure After Setup

```
XdividendALG/
├── venv/                          # Virtual environment (created by you)
├── outputs/                       # Generated results (created by backtest)
│   ├── xdiv_ml_report.html
│   ├── xdiv_ml_trade_log.csv
│   ├── xdiv_ml_equity_curve.png
│   └── ...
├── strategy_xdiv_ml.py           # New ML strategy
├── backtester_xdiv_ml.py         # New backtester
├── run_xdiv_ml_backtest.py       # Run this file
├── config.py                     # Configuration
├── data_manager.py               # Data fetching
├── risk_manager.py               # Risk management
├── dashboard/                    # Streamlit dashboard
│   ├── app.py
│   └── ...
├── .env                          # Your API keys (create this)
├── .gitignore                    # Git ignore rules
└── README.md                     # Documentation
```

---

## Next Steps

1. **Run the backtest** - See training and test results
2. **Review performance** - Check if Sharpe > 1.5, Return > 10%
3. **Compare strategies** - Old vs new ML strategy
4. **Optimize if needed** - Adjust parameters in `config.py`
5. **Paper trade** - Test with Alpaca paper trading
6. **Go live** - Only after successful paper trading

---

## Getting Help

### Documentation Files

- **XDIV_ML_STRATEGY_README.md** - Strategy overview and usage
- **PROJECT_ANALYSIS_AND_IMPROVEMENTS.md** - Complete analysis
- **WINDOWS_SETUP_GUIDE.md** - This file

### Command to Test Everything Works

```cmd
# Run this to verify setup
python -c "import pandas as pd; import numpy as np; import yfinance as yf; print('✅ Setup complete!')"
```

### Still Having Issues?

1. Check Python version: `python --version` (should be 3.8+)
2. Check pip version: `pip --version`
3. Verify virtual environment is activated: Look for `(venv)` in terminal
4. Try installing packages one by one to identify which fails
5. Check internet connection (needed to download stock data)

---

## Alternative: Using Anaconda (If You Prefer)

**Install Anaconda**: https://www.anaconda.com/download

```cmd
# Create conda environment
conda create -n xdividend python=3.11
conda activate xdividend

# Install packages
conda install pandas numpy scipy matplotlib seaborn scikit-learn
pip install yfinance alpaca-trade-api streamlit plotly

# Run backtest
python run_xdiv_ml_backtest.py
```

---

**Good luck testing the X-Dividend ML Strategy on Windows!** 🚀

If you encounter any issues not covered here, feel free to ask for help.
