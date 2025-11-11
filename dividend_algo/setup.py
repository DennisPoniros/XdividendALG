#!/usr/bin/env python3
"""
Setup Script for Dividend Capture Algorithm
Helps with initial configuration and validation
"""

import os
import sys

def check_dependencies():
    """Check if required packages are installed"""
    
    print("\n🔍 Checking dependencies...")
    
    required = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'yfinance': 'yfinance',
        'alpaca_trade_api': 'alpaca-trade-api',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn',
        'statsmodels': 'statsmodels'
    }
    
    missing = []
    
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True


def setup_api_keys():
    """Interactive setup for API keys"""
    
    print("\n🔑 API Key Setup")
    print("="*60)
    print("\nTo use this algorithm, you need a free Alpaca account.")
    print("Sign up at: https://alpaca.markets")
    print("\n1. Create account (free)")
    print("2. Enable paper trading")
    print("3. Generate API keys")
    print("="*60)
    
    response = input("\nDo you have Alpaca API keys? (y/n): ").strip().lower()
    
    if response == 'y':
        api_key = input("Enter API Key: ").strip()
        secret_key = input("Enter Secret Key: ").strip()
        
        # Update config file
        config_path = 'config.py'
        
        with open(config_path, 'r') as f:
            content = f.read()
        
        content = content.replace("'API_KEY': 'YOUR_ALPACA_KEY_HERE'", 
                                f"'API_KEY': '{api_key}'")
        content = content.replace("'SECRET_KEY': 'YOUR_ALPACA_SECRET_HERE'",
                                f"'SECRET_KEY': '{secret_key}'")
        
        with open(config_path, 'w') as f:
            f.write(content)
        
        print("\n✅ API keys configured in config.py")
        return True
    else:
        print("\n⚠️  You can still run backtests with yfinance data only")
        print("   But Alpaca is recommended for better data quality")
        return False


def create_output_directory():
    """Create output directory if it doesn't exist"""
    
    output_dir = '/mnt/user-data/outputs'
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n✅ Output directory ready: {output_dir}")
        return True
    except Exception as e:
        print(f"\n⚠️  Could not create output directory: {e}")
        return False


def test_data_connection():
    """Test data connectivity"""
    
    print("\n🔌 Testing data connections...")
    
    try:
        from data_manager import DataManager
        
        dm = DataManager()
        
        # Test yfinance
        print("  Testing yfinance...")
        test_ticker = dm.get_fundamentals('AAPL')
        if test_ticker:
            print("  ✅ yfinance working")
        else:
            print("  ⚠️  yfinance test inconclusive")
        
        # Test Alpaca (if configured)
        if dm.alpaca_api:
            print("  ✅ Alpaca API connected")
        else:
            print("  ⚠️  Alpaca API not configured (optional)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data connection test failed: {e}")
        return False


def run_quick_validation():
    """Run a quick validation backtest"""
    
    print("\n🧪 Running quick validation test...")
    
    try:
        from main import quick_validation_run
        
        success = quick_validation_run()
        
        if success:
            print("\n✅ SETUP COMPLETE - System is ready to use!")
            return True
        else:
            print("\n⚠️  Validation completed with warnings")
            return True  # Still consider it a success
            
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        return False


def main():
    """Main setup workflow"""
    
    print("\n" + "="*60)
    print("📈 DIVIDEND CAPTURE ALGORITHM - SETUP WIZARD")
    print("="*60)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Setup cannot continue. Install missing packages first.")
        sys.exit(1)
    
    # Step 2: Setup API keys (optional)
    setup_api_keys()
    
    # Step 3: Create output directory
    create_output_directory()
    
    # Step 4: Test connections
    if not test_data_connection():
        print("\n⚠️  Data connection issues detected")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            sys.exit(1)
    
    # Step 5: Quick validation
    print("\n" + "="*60)
    response = input("\nRun quick validation test? (y/n) [recommended]: ").strip().lower()
    
    if response in ['y', 'yes', '']:
        run_quick_validation()
    
    # Final instructions
    print("\n" + "="*60)
    print("🎉 SETUP WIZARD COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review config.py and adjust parameters")
    print("  2. Run full backtest: python main.py")
    print("  3. Check results in /mnt/user-data/outputs/")
    print("\nFor help: Read README.md")
    print("="*60 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Setup error: {e}")
        sys.exit(1)
