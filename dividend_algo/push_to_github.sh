#!/bin/bash
# Script to push Dividend Algorithm to GitHub
# Repository: https://github.com/DennisPoniros/XdividendALG

echo "=========================================="
echo "Pushing Dividend Algorithm to GitHub"
echo "=========================================="
echo ""

# Set repository URL
REPO_URL="https://github.com/DennisPoniros/XdividendALG.git"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install git first."
    exit 1
fi

echo "📁 Current directory: $(pwd)"
echo ""

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    echo "🔧 Initializing git repository..."
    git init
    git remote add origin $REPO_URL
else
    echo "✅ Git repository already initialized"
fi

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo "📝 Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Data files
*.csv
*.xlsx
*.pkl
*.h5

# Outputs (keep structure, not files)
outputs/*.png
outputs/*.html
outputs/*.csv
!outputs/.gitkeep

# API Keys (IMPORTANT!)
config_local.py
secrets.py
.env

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Logs
*.log
EOF
fi

# Create outputs directory placeholder
mkdir -p outputs
touch outputs/.gitkeep

# Stage all files
echo "📦 Staging files..."
git add .

# Show what will be committed
echo ""
echo "📋 Files to be committed:"
git status --short

echo ""
read -p "Continue with commit? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Commit
    echo "💾 Committing files..."
    git commit -m "Initial commit: Comprehensive dividend capture trading algorithm

- Complete backtesting framework with transaction costs
- Real data integration (Alpaca + yfinance)
- Advanced risk management (Kelly sizing, circuit breakers)
- Walk-forward validation
- Comprehensive analytics and reporting
- Parameter optimization
- 4,378 lines of production-ready code
- Full documentation and setup wizard

Features:
- Multi-factor screening system
- Mean reversion signals
- Technical indicators
- Quality scoring
- Position tracking
- Performance analytics
- HTML report generation

Target Performance:
- Annual Return: 8-12%
- Sharpe Ratio: 1.5-2.0
- Win Rate: 55-60%
- Max Drawdown: <15%"

    # Push to GitHub
    echo "🚀 Pushing to GitHub..."
    echo "You may be prompted for GitHub credentials..."
    echo ""
    
    git branch -M main
    git push -u origin main
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✅ SUCCESS! Code pushed to GitHub"
        echo "=========================================="
        echo ""
        echo "View your repository at:"
        echo "https://github.com/DennisPoniros/XdividendALG"
        echo ""
    else
        echo ""
        echo "❌ Push failed. Common solutions:"
        echo ""
        echo "1. Authentication required:"
        echo "   - Use GitHub Personal Access Token instead of password"
        echo "   - Generate token at: https://github.com/settings/tokens"
        echo "   - Use token as password when prompted"
        echo ""
        echo "2. Repository doesn't exist:"
        echo "   - Create it at: https://github.com/new"
        echo "   - Name: XdividendALG"
        echo "   - Make it public or private"
        echo ""
        echo "3. Branch protection:"
        echo "   - Check repository settings"
        echo "   - Disable branch protection temporarily"
        echo ""
    fi
else
    echo "❌ Commit cancelled"
    exit 1
fi
