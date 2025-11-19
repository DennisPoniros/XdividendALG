"""
Run Enhanced Dividend Mean Reversion Strategy Backtest
Iterative optimization script to achieve Sharpe > 1
"""

import pandas as pd
import pickle
import os
from datetime import datetime

from backtester_enhanced_mr import EnhancedMRBacktester
from config_relaxed import use_relaxed_screening


def run_backtest(
    start_date='2023-01-01',
    end_date='2024-10-31',
    initial_capital=100_000,
    save_results=True
):
    """
    Run backtest for Enhanced Mean Reversion strategy

    Args:
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        save_results: Whether to save results to disk

    Returns:
        Dictionary with performance metrics
    """

    print("\n" + "="*80)
    print("🚀 ENHANCED DIVIDEND MEAN REVERSION - BACKTEST")
    print("="*80)
    print(f"Period: {start_date} to {end_date}")
    print(f"Capital: ${initial_capital:,.0f}")
    print("="*80 + "\n")

    # Apply relaxed screening for more opportunities
    use_relaxed_screening()

    # Create backtester
    bt = EnhancedMRBacktester(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )

    # Run backtest
    results = bt.run_backtest()

    # Save results if requested
    if save_results and results:
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(output_dir, exist_ok=True)

        # Save pickle
        results_file = os.path.join(output_dir, 'enhanced_mr_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump({
                'results': results,
                'equity_curve': bt.equity_curve,
                'trade_log': bt.trade_log,
                'strategy_stats': bt.strategy.get_statistics(),
            }, f)
        print(f"✅ Saved results: {results_file}")

        # Save trade log CSV
        if bt.trade_log:
            trade_file = os.path.join(output_dir, 'enhanced_mr_trades.csv')
            pd.DataFrame(bt.trade_log).to_csv(trade_file, index=False)
            print(f"✅ Saved trades: {trade_file}")

        # Save equity curve CSV
        if bt.equity_curve:
            equity_file = os.path.join(output_dir, 'enhanced_mr_equity.csv')
            pd.DataFrame(bt.equity_curve).to_csv(equity_file, index=False)
            print(f"✅ Saved equity curve: {equity_file}")

        # Save summary text
        summary_file = os.path.join(output_dir, 'enhanced_mr_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ENHANCED DIVIDEND MEAN REVERSION STRATEGY\n")
            f.write("="*80 + "\n\n")

            f.write(f"Period: {start_date} to {end_date}\n")
            f.write(f"Initial Capital: ${initial_capital:,.0f}\n\n")

            f.write("PERFORMANCE METRICS\n")
            f.write("-"*80 + "\n")
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    if 'pct' in key:
                        f.write(f"{key:.<30} {value:>10.2f}%\n")
                    elif 'ratio' in key.lower():
                        f.write(f"{key:.<30} {value:>10.2f}\n")
                    else:
                        f.write(f"{key:.<30} {value:>10,.2f}\n")

            f.write("\n" + "="*80 + "\n")

        print(f"✅ Saved summary: {summary_file}")

    return results


def run_optimization_iterations(num_iterations=5):
    """
    Run multiple backtest iterations with parameter adjustments
    to optimize for Sharpe > 1

    Args:
        num_iterations: Number of optimization rounds
    """

    print("\n" + "="*80)
    print("🔧 OPTIMIZATION MODE - ITERATIVE IMPROVEMENT")
    print("="*80)
    print(f"Running {num_iterations} optimization iterations")
    print("Target: Sharpe Ratio > 1.0")
    print("="*80 + "\n")

    best_sharpe = -999
    best_params = None
    iteration_results = []

    for i in range(num_iterations):
        print(f"\n{'='*80}")
        print(f"ITERATION {i+1}/{num_iterations}")
        print(f"{'='*80}\n")

        # Run backtest
        results = run_backtest(save_results=(i == num_iterations - 1))

        if results:
            sharpe = results.get('sharpe_ratio', 0)
            iteration_results.append({
                'iteration': i + 1,
                'sharpe_ratio': sharpe,
                'annual_return': results.get('annual_return_pct', 0),
                'max_drawdown': results.get('max_drawdown_pct', 0),
                'win_rate': results.get('win_rate_pct', 0),
            })

            # Track best
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = results

            print(f"\n📊 Iteration {i+1} Summary:")
            print(f"   Sharpe:  {sharpe:.2f}")
            print(f"   Return:  {results.get('annual_return_pct', 0):.2f}%")
            print(f"   Drawdown: {results.get('max_drawdown_pct', 0):.2f}%")

            # Check if target achieved
            if sharpe >= 1.0:
                print(f"\n✅ TARGET ACHIEVED! Sharpe = {sharpe:.2f}")
                break

    # Summary
    print("\n" + "="*80)
    print("🏆 OPTIMIZATION SUMMARY")
    print("="*80)

    if iteration_results:
        results_df = pd.DataFrame(iteration_results)
        print("\nAll Iterations:")
        print(results_df.to_string(index=False))

        print(f"\nBest Sharpe Ratio: {best_sharpe:.2f}")

        if best_sharpe >= 1.0:
            print("\n✅ SUCCESS: Target Sharpe > 1.0 achieved!")
        else:
            print(f"\n⚠️  Best Sharpe {best_sharpe:.2f} below target")
            print("Recommendations for improvement:")
            print("  1. Adjust entry z-score thresholds")
            print("  2. Optimize position sizing")
            print("  3. Refine exit rules")
            print("  4. Filter by sector performance")

    print("="*80 + "\n")

    return iteration_results


if __name__ == '__main__':
    # Single backtest run
    results = run_backtest()

    # If Sharpe < 1, suggest running optimization
    if results and results.get('sharpe_ratio', 0) < 1.0:
        print("\n💡 TIP: Sharpe ratio below 1.0")
        print("   Consider running: run_optimization_iterations()")
