"""
Simple Dividend Backtest - Fixed Exit Logic

Fixes the critical bug where stop losses exit immediately after dividend drop.

Key Changes:
1. NO stop losses based on P&L (dividend drop is expected!)
2. Exit based on TIME after ex-div (learned optimal hold period)
3. Optional profit target (5%)
4. Emergency exit only (<-10%)
"""

import pandas as pd
import pickle
import os
from datetime import datetime

from backtester_xdiv_ml import XDividendMLBacktester
from analytics import PerformanceAnalytics
from config_relaxed import use_relaxed_screening
from config_simple_exits import apply_simple_exit_config


def run_simple_backtest():
    """
    Run backtest with FIXED exit logic (no stop losses)
    """

    print("\n" + "="*80)
    print("🔧 FIXED X-DIVIDEND STRATEGY BACKTEST")
    print("="*80)
    print("\nCritical Fixes Applied:")
    print("  ✅ Removed stop losses (dividend drop is EXPECTED)")
    print("  ✅ Exit based on TIME, not P&L")
    print("  ✅ Hold through ex-div for mean reversion")
    print("  ✅ Simple, robust exit rules")
    print("="*80 + "\n")

    # Apply configurations
    use_relaxed_screening()
    apply_simple_exit_config()

    print("\n" + "="*80)
    print("🚀 RUNNING BACKTEST")
    print("="*80)

    # Create backtester
    bt = XDividendMLBacktester(
        train_start='2018-01-01',
        train_end='2022-12-31',
        test_start='2023-01-01',
        test_end='2024-10-31',
        initial_capital=100_000
    )

    # Run backtest
    results = bt.run_backtest_with_training()

    # Create output directory
    output_dir = '/mnt/user-data/outputs'
    try:
        os.makedirs(output_dir, exist_ok=True)
    except (PermissionError, OSError):
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        print(f"ℹ️  Using local output directory: {output_dir}")

    # Save results
    print("\n📁 Saving results...")

    # Pickle
    results_file = os.path.join(output_dir, 'simple_backtest_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"✅ Saved pickle: {results_file}")

    # Trade log
    if len(bt.trade_log) > 0:
        trade_log_file = os.path.join(output_dir, 'simple_trade_log.csv')
        trade_df = pd.DataFrame(bt.trade_log)
        trade_df.to_csv(trade_log_file, index=False)
        print(f"✅ Saved trade log: {trade_log_file}")

    # Equity curve
    if len(bt.equity_curve) > 0:
        equity_file = os.path.join(output_dir, 'simple_equity_curve.csv')
        equity_df = pd.DataFrame(bt.equity_curve)
        equity_df.to_csv(equity_file, index=False)
        print(f"✅ Saved equity curve: {equity_file}")

    # Generate visualizations
    if bt.performance_metrics:
        print("\n📊 Generating visualizations...")

        analytics = PerformanceAnalytics(bt.performance_metrics)

        try:
            analytics.plot_equity_curve(save_path=os.path.join(output_dir, 'simple_equity_curve.png'))
            analytics.plot_drawdown(save_path=os.path.join(output_dir, 'simple_drawdown.png'))
            analytics.plot_monthly_returns(save_path=os.path.join(output_dir, 'simple_monthly_returns.png'))
            print("✅ Generated visualizations")
        except Exception as e:
            print(f"⚠️  Visualization error: {e}")

        # HTML report
        try:
            html_file = os.path.join(output_dir, 'simple_report.html')
            analytics.generate_html_report(
                save_path=html_file,
                strategy_name="Simple X-Dividend Strategy (Fixed Exits)"
            )
            print(f"✅ Generated report: {html_file}")
        except Exception as e:
            print(f"⚠️  Report error: {e}")

    # Training summary
    summary_file = os.path.join(output_dir, 'simple_training_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SIMPLE X-DIVIDEND STRATEGY - RESULTS\n")
        f.write("="*80 + "\n\n")

        if results['training']:
            f.write("TRAINING PHASE\n")
            f.write("-"*80 + "\n")
            training = results['training']
            f.write(f"Events Analyzed: {training.get('num_analyzed_events', 0)}\n")
            learned = training.get('learned_parameters', {})
            f.write(f"Learned Capture Rate: {learned.get('avg_dividend_capture_rate', 0)*100:.1f}%\n")
            f.write(f"Optimal Entry Days: {learned.get('optimal_entry_days', [])}\n")
            f.write(f"Optimal Hold Period: {learned.get('optimal_hold_days', 0)} days\n")

        f.write("\n" + "="*80 + "\n")
        f.write("TEST PERIOD RESULTS\n")
        f.write("="*80 + "\n\n")

        if results['testing']:
            testing = results['testing']
            f.write(f"Annual Return: {testing.get('annual_return_pct', 0):.2f}%\n")
            f.write(f"Sharpe Ratio: {testing.get('sharpe_ratio', 0):.2f}\n")
            f.write(f"Max Drawdown: {testing.get('max_drawdown_pct', 0):.2f}%\n")
            f.write(f"Win Rate: {testing.get('win_rate_pct', 0):.2f}%\n")
            f.write(f"Total Trades: {testing.get('total_trades', 0)}\n")
            f.write(f"Profit Factor: {testing.get('profit_factor', 0):.2f}\n")
            f.write(f"Avg Holding Days: {testing.get('avg_holding_days', 0):.1f}\n")
            f.write(f"\nBest Day: {testing.get('best_day_pct', 0):.2f}%\n")
            f.write(f"Worst Day: {testing.get('worst_day_pct', 0):.2f}%\n")

        f.write("\n" + "="*80 + "\n")
        f.write("CRITICAL FIXES APPLIED\n")
        f.write("="*80 + "\n")
        f.write("✅ Removed stop losses (dividend drop is expected)\n")
        f.write("✅ Exit based on time after ex-div, not P&L\n")
        f.write("✅ Hold through dividend drop for mean reversion\n")
        f.write("✅ Simple profit target (5%) and emergency exit (-10%)\n")

    print(f"✅ Saved summary: {summary_file}")

    print("\n" + "="*80)
    print("✅ BACKTEST COMPLETED")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print("="*80 + "\n")

    # Compare if needed
    if results['testing']:
        testing = results['testing']
        print("\n📊 QUICK RESULTS:")
        print(f"  Annual Return: {testing.get('annual_return_pct', 0):.2f}%")
        print(f"  Sharpe Ratio: {testing.get('sharpe_ratio', 0):.2f}")
        print(f"  Win Rate: {testing.get('win_rate_pct', 0):.2f}%")
        print(f"  Total Trades: {testing.get('total_trades', 0)}")
        print(f"  Best Day: {testing.get('best_day_pct', 0):.2f}%")
        print(f"  Profit Factor: {testing.get('profit_factor', 0):.2f}")

        # Assessment
        if testing.get('sharpe_ratio', 0) > 1.0:
            print("\n✅ Strategy looks GOOD!")
        elif testing.get('profit_factor', 0) > 1.0:
            print("\n⚠️  Strategy is profitable but needs optimization")
        else:
            print("\n❌ Strategy still needs work")

    return results


if __name__ == '__main__':
    results = run_simple_backtest()
