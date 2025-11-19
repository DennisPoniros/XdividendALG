"""
Run Enhanced Dividend Mean Reversion Strategy with Mock Data
For testing and development without external data dependencies
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# Use mock data manager instead of real one
from mock_data_manager import MockDataManager
from strategy_enhanced_mr import EnhancedDividendMeanReversionStrategy
from backtester_enhanced_mr import EnhancedMRBacktester


# Monkey patch the backtester to use mock data
class MockEnhancedMRBacktester(EnhancedMRBacktester):
    """Backtester using mock data"""

    def __init__(self, start_date: str, end_date: str, initial_capital: float):
        super().__init__(start_date, end_date, initial_capital)

        # Replace data manager with mock
        self.dm = MockDataManager(seed=42)
        self.strategy.dm = self.dm


def run_mock_backtest(
    start_date='2023-01-01',
    end_date='2024-10-31',
    initial_capital=100_000,
    save_results=True
):
    """
    Run backtest with mock data

    Args:
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        save_results: Whether to save results

    Returns:
        Performance results dictionary
    """

    print("\n" + "="*80)
    print("🚀 ENHANCED DIVIDEND MEAN REVERSION - MOCK BACKTEST")
    print("="*80)
    print("Using synthetic dividend and price data for testing")
    print(f"Period: {start_date} to {end_date}")
    print(f"Capital: ${initial_capital:,.0f}")
    print("="*80 + "\n")

    # Create backtester with mock data
    bt = MockEnhancedMRBacktester(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )

    # Run backtest
    results = bt.run_backtest()

    # Save results
    if save_results and results:
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(output_dir, exist_ok=True)

        # Save comprehensive results
        full_results = {
            'results': results,
            'equity_curve': bt.equity_curve,
            'trade_log': bt.trade_log,
            'strategy_stats': bt.strategy.get_statistics(),
            'parameters': bt.strategy.params,
        }

        # Pickle
        results_file = os.path.join(output_dir, 'enhanced_mr_mock_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(full_results, f)
        print(f"✅ Saved results: {results_file}")

        # Trade log CSV
        if bt.trade_log:
            trade_file = os.path.join(output_dir, 'enhanced_mr_mock_trades.csv')
            pd.DataFrame(bt.trade_log).to_csv(trade_file, index=False)
            print(f"✅ Saved trades: {trade_file}")

        # Equity curve CSV
        if bt.equity_curve:
            equity_file = os.path.join(output_dir, 'enhanced_mr_mock_equity.csv')
            pd.DataFrame(bt.equity_curve).to_csv(equity_file, index=False)
            print(f"✅ Saved equity curve: {equity_file}")

        # Summary report
        summary_file = os.path.join(output_dir, 'enhanced_mr_mock_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ENHANCED DIVIDEND MEAN REVERSION STRATEGY (MOCK DATA)\n")
            f.write("="*80 + "\n\n")

            f.write(f"Backtest Period: {start_date} to {end_date}\n")
            f.write(f"Initial Capital: ${initial_capital:,.0f}\n")
            f.write(f"Data Source: Synthetic (Mock)\n\n")

            f.write("PERFORMANCE METRICS\n")
            f.write("-"*80 + "\n")

            for key, value in sorted(results.items()):
                if isinstance(value, (int, float)):
                    if 'pct' in key:
                        f.write(f"  {key:<30} {value:>12.2f}%\n")
                    elif 'ratio' in key.lower():
                        f.write(f"  {key:<30} {value:>12.2f}\n")
                    elif 'days' in key or 'trades' in key:
                        f.write(f"  {key:<30} {value:>12.0f}\n")
                    else:
                        f.write(f"  {key:<30} {value:>12,.2f}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("STRATEGY PARAMETERS\n")
            f.write("-"*80 + "\n")

            for key, value in bt.strategy.params.items():
                f.write(f"  {key:<35} {value}\n")

            f.write("\n" + "="*80 + "\n")

        print(f"✅ Saved summary: {summary_file}")

    return results


def optimize_parameters():
    """
    Run parameter optimization to find best settings for Sharpe > 1
    """

    print("\n" + "="*80)
    print("🔧 PARAMETER OPTIMIZATION")
    print("="*80)
    print("Testing different parameter combinations")
    print("Goal: Achieve Sharpe Ratio > 1.0")
    print("="*80 + "\n")

    # Parameter grid to test
    param_grid = {
        'pre_div_zscore_threshold': [-2.0, -1.5, -1.0],
        'post_div_zscore_threshold': [-2.5, -2.0, -1.5],
        'profit_target': [0.025, 0.03, 0.035],
        'stop_loss': [0.015, 0.02, 0.025],
    }

    best_sharpe = -999
    best_params = None
    optimization_results = []

    # Simple grid search
    from itertools import product

    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    print(f"Testing {len(param_combinations)} parameter combinations...\n")

    for i, combo in enumerate(param_combinations):
        # Create backtest
        bt = MockEnhancedMRBacktester(
            start_date='2023-01-01',
            end_date='2024-10-31',
            initial_capital=100_000
        )

        # Update strategy parameters
        for param_name, param_value in zip(param_names, combo):
            bt.strategy.params[param_name] = param_value

        print(f"\n[{i+1}/{len(param_combinations)}] Testing: {dict(zip(param_names, combo))}")

        # Run backtest
        try:
            results = bt.run_backtest()

            if results:
                sharpe = results.get('sharpe_ratio', 0)
                annual_return = results.get('annual_return_pct', 0)
                max_dd = results.get('max_drawdown_pct', 0)

                optimization_results.append({
                    **dict(zip(param_names, combo)),
                    'sharpe_ratio': sharpe,
                    'annual_return_pct': annual_return,
                    'max_drawdown_pct': max_dd,
                })

                # Track best
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = dict(zip(param_names, combo))

                print(f"   → Sharpe: {sharpe:.2f}, Return: {annual_return:.2f}%, DD: {max_dd:.2f}%")

                # Early stopping if great result found
                if sharpe >= 1.5:
                    print(f"\n✅ Excellent result found! Sharpe = {sharpe:.2f}")
                    break

        except Exception as e:
            print(f"   ✗ Error: {e}")
            continue

    # Summary
    print("\n" + "="*80)
    print("📊 OPTIMIZATION RESULTS")
    print("="*80)

    if optimization_results:
        results_df = pd.DataFrame(optimization_results)
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)

        print("\nTop 5 Parameter Combinations:")
        print(results_df.head(10).to_string(index=False))

        print(f"\n🏆 BEST PARAMETERS:")
        print(f"   Sharpe Ratio: {best_sharpe:.2f}")
        for param, value in best_params.items():
            print(f"   {param}: {value}")

        if best_sharpe >= 1.0:
            print("\n✅ SUCCESS: Target Sharpe > 1.0 achieved!")
        else:
            print(f"\n⚠️  Best Sharpe {best_sharpe:.2f} below target")

        # Save results
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(output_dir, exist_ok=True)

        opt_file = os.path.join(output_dir, 'parameter_optimization.csv')
        results_df.to_csv(opt_file, index=False)
        print(f"\n✅ Saved optimization results: {opt_file}")

    print("="*80 + "\n")

    return optimization_results, best_params


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'optimize':
        # Run optimization
        optimize_parameters()
    else:
        # Single backtest
        results = run_mock_backtest()

        print("\n💡 To run parameter optimization, use:")
        print("   python run_enhanced_mr_mock.py optimize")
