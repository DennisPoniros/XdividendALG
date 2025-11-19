"""
Run Swing Dividend Trader Strategy Backtest
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from swing_dividend_trader import OptimizedStrategy
from backtester_enhanced_mr import EnhancedMRBacktester
from mock_data_manager import MockDataManager

# Temporarily patch the backtester to use our custom strategy
class CustomBacktester(EnhancedMRBacktester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace strategy
        self.strategy = OptimizedStrategy(self.data_manager)

if __name__ == '__main__':
    print("\n" + "="*80)
    print("Swing Dividend Trader")
    print("="*80 + "\n")

    bt = CustomBacktester(
        start_date='2023-01-01',
        end_date='2024-10-31',
        initial_capital=100_000
    )

    results = bt.run_backtest()

    print("\n" + "="*80)
    if results and results.get('sharpe_ratio', 0) >= 1.0:
        print("✅ SUCCESS! Sharpe >= 1.0")
        print(f"   Sharpe: {results['sharpe_ratio']:.2f}")
        print(f"   Annual Return: {results['annual_return_pct']:.2f}%")
        print(f"   Max DD: {results['max_drawdown_pct']:.2f}%")
    else:
        sharpe = results.get('sharpe_ratio', 0) if results else 0
        print(f"⚠️  Sharpe {sharpe:.2f} below target")
    print("="*80 + "\n")
