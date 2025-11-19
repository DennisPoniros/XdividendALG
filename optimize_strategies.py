"""
Rapid Strategy Optimization to Achieve Sharpe > 1
Create multiple profitable dividend mean reversion variants
"""

import pandas as pd
import pickle
from pathlib import Path

# Strategy configurations to test
STRATEGY_VARIANTS = [
    {
        'name': 'Aggressive Dividend Capture',
        'description': 'Wide entry window, high frequency, moderate stops',
        'params': {
            'pre_div_entry_window': (-14, -1),  # Earlier entry
            'post_div_entry_window': (0, 5),
            'pre_div_zscore_threshold': -0.8,  # Less restrictive
            'post_div_zscore_threshold': -1.2,
            'profit_target': 0.025,  # +2.5%
            'stop_loss': 0.015,  # -1.5%
            'max_holding_days': 12,
            'max_positions': 25,  # More positions
            'base_position_size': 0.03,  # 3% per position
        }
    },
    {
        'name': 'Quality Dividend Mean Reversion',
        'description': 'Selective entry, higher conviction, tight stops',
        'params': {
            'pre_div_entry_window': (-5, -2),  # Narrow, focused
            'post_div_entry_window': (0, 3),
            'pre_div_zscore_threshold': -1.8,  # More selective
            'post_div_zscore_threshold': -2.2,
            'profit_target': 0.04,  # +4% (let winners run)
            'stop_loss': 0.012,  # -1.2% (tight)
            'max_holding_days': 10,
            'max_positions': 15,
            'base_position_size': 0.04,  # 4% per position
        }
    },
    {
        'name': 'Balanced Alpha Generator',
        'description': 'Balanced approach, consistent returns',
        'params': {
            'pre_div_entry_window': (-8, -2),
            'post_div_entry_window': (0, 4),
            'pre_div_zscore_threshold': -1.2,
            'post_div_zscore_threshold': -1.6,
            'profit_target': 0.03,  # +3%
            'stop_loss': 0.018,  # -1.8%
            'max_holding_days': 11,
            'max_positions': 20,
            'base_position_size': 0.035,  # 3.5%
        }
    },
    {
        'name': 'Swing Dividend Trader',
        'description': 'Longer holds, larger targets, room to breathe',
        'params': {
            'pre_div_entry_window': (-10, -3),
            'post_div_entry_window': (0, 6),
            'pre_div_zscore_threshold': -1.0,
            'post_div_zscore_threshold': -1.5,
            'profit_target': 0.05,  # +5%
            'stop_loss': 0.025,  # -2.5%
            'max_holding_days': 15,
            'max_positions': 18,
            'base_position_size': 0.04,
        }
    },
    {
        'name': 'Scalper Dividend Edge',
        'description': 'Quick in/out, high frequency, small gains',
        'params': {
            'pre_div_entry_window': (-6, -1),
            'post_div_entry_window': (0, 2),
            'pre_div_zscore_threshold': -0.5,  # Very relaxed
            'post_div_zscore_threshold': -1.0,
            'profit_target': 0.015,  # +1.5% (quick)
            'stop_loss': 0.01,  # -1.0% (very tight)
            'max_holding_days': 7,
            'max_positions': 30,  # Many positions
            'base_position_size': 0.025,  # 2.5%
        }
    },
]

def create_optimized_strategies():
    """
    Create optimized strategy files ready for backtesting.
    """

    print("\n" + "="*80)
    print("🔧 CREATING OPTIMIZED DIVIDEND MEAN REVERSION STRATEGIES")
    print("="*80 + "\n")

    output_dir = Path(__file__).parent / 'strategies_optimized'
    output_dir.mkdir(exist_ok=True)

    for variant in STRATEGY_VARIANTS:
        name = variant['name']
        desc = variant['description']
        params = variant['params']

        print(f"Creating: {name}")
        print(f"  {desc}")

        # Create strategy file
        strategy_code = f'''"""
{name}
{'='*len(name)}

{desc}

PARAMETERS:
- Entry Window (pre-div): {params['pre_div_entry_window']}
- Entry Window (post-div): {params['post_div_entry_window']}
- Z-score thresholds: {params['pre_div_zscore_threshold']}, {params['post_div_zscore_threshold']}
- Profit Target: {params['profit_target']*100:.1f}%
- Stop Loss: {params['stop_loss']*100:.1f}%
- Max Holding: {params['max_holding_days']} days
- Max Positions: {params['max_positions']}
- Position Size: {params['base_position_size']*100:.1f}%
"""

from strategy_enhanced_mr import EnhancedDividendMeanReversionStrategy

class OptimizedStrategy(EnhancedDividendMeanReversionStrategy):
    """Optimized variant with custom parameters."""

    def __init__(self, data_manager):
        super().__init__(data_manager)

        # Override with optimized parameters
        self.params.update({{
            'pre_div_entry_window': {params['pre_div_entry_window']},
            'post_div_entry_window': {params['post_div_entry_window']},
            'pre_div_zscore_threshold': {params['pre_div_zscore_threshold']},
            'post_div_zscore_threshold': {params['post_div_zscore_threshold']},
            'profit_target': {params['profit_target']},
            'stop_loss': {params['stop_loss']},
            'max_holding_days': {params['max_holding_days']},
            'max_positions': {params['max_positions']},
            'base_position_size': {params['base_position_size']},
        }})
'''

        # Save strategy file
        filename = name.lower().replace(' ', '_') + '.py'
        filepath = output_dir / filename

        with open(filepath, 'w') as f:
            f.write(strategy_code)

        print(f"  ✅ Saved to: {filepath}")

        # Create run script
        run_script = f'''"""
Run {name} Strategy Backtest
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from {filename[:-3]} import OptimizedStrategy
from backtester_enhanced_mr import EnhancedMRBacktester
from mock_data_manager import MockDataManager

# Temporarily patch the backtester to use our custom strategy
class CustomBacktester(EnhancedMRBacktester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace strategy
        self.strategy = OptimizedStrategy(self.data_manager)

if __name__ == '__main__':
    print("\\n" + "="*80)
    print("{name}")
    print("="*80 + "\\n")

    bt = CustomBacktester(
        start_date='2023-01-01',
        end_date='2024-10-31',
        initial_capital=100_000
    )

    results = bt.run_backtest()

    print("\\n" + "="*80)
    if results and results.get('sharpe_ratio', 0) >= 1.0:
        print("✅ SUCCESS! Sharpe >= 1.0")
        print(f"   Sharpe: {{results['sharpe_ratio']:.2f}}")
        print(f"   Annual Return: {{results['annual_return_pct']:.2f}}%")
        print(f"   Max DD: {{results['max_drawdown_pct']:.2f}}%")
    else:
        sharpe = results.get('sharpe_ratio', 0) if results else 0
        print(f"⚠️  Sharpe {{sharpe:.2f}} below target")
    print("="*80 + "\\n")
'''

        run_filepath = output_dir / f"run_{filename}"
        with open(run_filepath, 'w') as f:
            f.write(run_script)

        print(f"  ✅ Run script: {run_filepath}\n")

    print("="*80)
    print(f"✅ Created {len(STRATEGY_VARIANTS)} optimized strategy variants")
    print(f"📁 Location: {output_dir}")
    print("\nNext steps:")
    print("  1. Run each strategy: cd strategies_optimized && python run_*.py")
    print("  2. Compare results and select best performer")
    print("  3. Validate winner on real data")
    print("="*80 + "\n")

    return output_dir

if __name__ == '__main__':
    create_optimized_strategies()
