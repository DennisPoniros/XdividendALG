"""
Run X-Dividend ML Strategy Backtest and Save Results for Dashboard
"""

import pandas as pd
import pickle
import os
from datetime import datetime

from backtester_xdiv_ml import XDividendMLBacktester
from analytics import PerformanceAnalytics
from config_relaxed import use_relaxed_screening


def run_xdiv_ml_backtest():
    """
    Run the X-Dividend ML strategy with training and testing
    Save results for dashboard viewing

    NOTE: Uses RELAXED screening to allow sufficient trade opportunities
    """

    # Use relaxed screening (fixes "only 2 trades" issue)
    print("\n" + "="*80)
    print("⚙️  APPLYING RELAXED SCREENING CONFIGURATION")
    print("="*80)
    use_relaxed_screening()
    print("="*80 + "\n")

    print("="*80)
    print("🚀 X-DIVIDEND ML STRATEGY BACKTEST")
    print("="*80)
    print("\nThis backtest will:")
    print("  1. Train on historical data (2018-2022)")
    print("  2. Test on out-of-sample data (2023-2024)")
    print("  3. Generate analytics and save for dashboard")
    print("="*80 + "\n")

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

    # Create output directory (works on Windows and Linux)
    # Try /mnt/user-data/outputs first (Linux), fallback to ./outputs (Windows)
    output_dir = '/mnt/user-data/outputs'
    try:
        os.makedirs(output_dir, exist_ok=True)
    except (PermissionError, OSError):
        # Fallback to local outputs directory
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        print(f"ℹ️  Using local output directory: {output_dir}")

    # Save results for dashboard
    print("\n📁 Saving results for dashboard...")

    # 1. Save as pickle (for dashboard data interface)
    results_file = os.path.join(output_dir, 'xdiv_ml_backtest_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"✅ Saved pickle: {results_file}")

    # 2. Save trade log as CSV
    trade_log_file = os.path.join(output_dir, 'xdiv_ml_trade_log.csv')
    if len(bt.trade_log) > 0:
        trade_df = pd.DataFrame(bt.trade_log)
        trade_df.to_csv(trade_log_file, index=False)
        print(f"✅ Saved trade log: {trade_log_file}")

    # 3. Save equity curve as CSV
    equity_file = os.path.join(output_dir, 'xdiv_ml_equity_curve.csv')
    if len(bt.equity_curve) > 0:
        equity_df = pd.DataFrame(bt.equity_curve)
        equity_df.to_csv(equity_file, index=False)
        print(f"✅ Saved equity curve: {equity_file}")

    # 4. Generate visualizations using existing analytics
    if bt.performance_metrics:
        print("\n📊 Generating performance analytics...")

        analytics = PerformanceAnalytics(bt.performance_metrics)

        # Generate all plots
        analytics.plot_equity_curve(save_path=os.path.join(output_dir, 'xdiv_ml_equity_curve.png'))
        analytics.plot_drawdown(save_path=os.path.join(output_dir, 'xdiv_ml_drawdown.png'))
        analytics.plot_monthly_returns(save_path=os.path.join(output_dir, 'xdiv_ml_monthly_returns.png'))
        analytics.plot_rolling_sharpe(save_path=os.path.join(output_dir, 'xdiv_ml_rolling_sharpe.png'))

        print("✅ Generated all visualizations")

        # Generate HTML report
        html_report_file = os.path.join(output_dir, 'xdiv_ml_report.html')
        analytics.generate_html_report(
            save_path=html_report_file,
            strategy_name="X-Dividend ML Strategy"
        )
        print(f"✅ Generated HTML report: {html_report_file}")

    # 5. Save training summary
    training_summary_file = os.path.join(output_dir, 'xdiv_ml_training_summary.txt')
    with open(training_summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("X-DIVIDEND ML STRATEGY - TRAINING SUMMARY\n")
        f.write("="*80 + "\n\n")

        if results['training']:
            training = results['training']
            f.write(f"Training Period: {training.get('training_period', 'N/A')}\n")
            f.write(f"Dividend Events Analyzed: {training.get('num_analyzed_events', 0)}\n\n")

            f.write("Learned Parameters:\n")
            f.write("-"*80 + "\n")
            learned = training.get('learned_parameters', {})
            f.write(f"  Average Dividend Capture Rate: {learned.get('avg_dividend_capture_rate', 0)*100:.1f}%\n")
            f.write(f"  Optimal Entry Days: {learned.get('optimal_entry_days', [])}\n")
            f.write(f"  Optimal Hold Period: {learned.get('optimal_hold_days', 0)} days\n")
            f.write(f"  Z-Score Threshold: {learned.get('z_score_threshold', 0):.2f}\n")
            f.write(f"  RSI Range: {learned.get('rsi_threshold_low', 0):.0f} - {learned.get('rsi_threshold_high', 0):.0f}\n")
            f.write(f"  Min Expected Return: {learned.get('min_expected_return', 0)*100:.2f}%\n")
            f.write(f"  Stock-Specific Rates Learned: {len(learned.get('stock_specific_rates', {}))}\n")
            f.write(f"  Sector Patterns Learned: {len(learned.get('sector_performance', {}))}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("TEST PERIOD RESULTS\n")
        f.write("="*80 + "\n\n")

        if results['testing']:
            testing = results['testing']
            f.write(f"Total Return: {testing.get('total_return_pct', 0):.2f}%\n")
            f.write(f"Annual Return: {testing.get('annual_return_pct', 0):.2f}%\n")
            f.write(f"Sharpe Ratio: {testing.get('sharpe_ratio', 0):.2f}\n")
            f.write(f"Max Drawdown: {testing.get('max_drawdown_pct', 0):.2f}%\n")
            f.write(f"Win Rate: {testing.get('win_rate_pct', 0):.2f}%\n")
            f.write(f"Total Trades: {testing.get('total_trades', 0)}\n")
            f.write(f"Profit Factor: {testing.get('profit_factor', 0):.2f}\n")

    print(f"✅ Saved training summary: {training_summary_file}")

    print("\n" + "="*80)
    print("✅ BACKTEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print("\nTo view results in dashboard:")
    print("  1. Navigate to dashboard directory: cd dashboard/")
    print("  2. Run dashboard: python run_dashboard.py")
    print("  3. Open browser to: http://localhost:8501")
    print("\nNote: You may need to modify data_interface.py to load xdiv_ml results")
    print("="*80 + "\n")

    return results


def compare_strategies():
    """
    Compare old strategy vs new ML strategy (if both results exist)
    """

    # Try both possible output directories
    output_dir = '/mnt/user-data/outputs'
    if not os.path.exists(output_dir):
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')

    # Check if both result files exist
    old_results_file = os.path.join(output_dir, 'backtest_results.pkl')
    new_results_file = os.path.join(output_dir, 'xdiv_ml_backtest_results.pkl')

    if not os.path.exists(new_results_file):
        print("⚠️  New strategy results not found. Run backtest first.")
        return

    # Load new results
    with open(new_results_file, 'rb') as f:
        new_results = pickle.load(f)

    print("\n" + "="*80)
    print("📊 STRATEGY COMPARISON")
    print("="*80)

    if os.path.exists(old_results_file):
        # Load old results
        with open(old_results_file, 'rb') as f:
            old_results = pickle.load(f)

        print("\n                           Old Strategy    |    New ML Strategy")
        print("-"*80)

        # Extract metrics (handle different result structures)
        old_metrics = old_results if isinstance(old_results, dict) else {}
        new_metrics = new_results.get('testing', {})

        print(f"Annual Return:          {old_metrics.get('annual_return_pct', 0):>10.2f}%    |    {new_metrics.get('annual_return_pct', 0):>10.2f}%")
        print(f"Sharpe Ratio:           {old_metrics.get('sharpe_ratio', 0):>10.2f}     |    {new_metrics.get('sharpe_ratio', 0):>10.2f}")
        print(f"Max Drawdown:           {old_metrics.get('max_drawdown_pct', 0):>10.2f}%    |    {new_metrics.get('max_drawdown_pct', 0):>10.2f}%")
        print(f"Win Rate:               {old_metrics.get('win_rate_pct', 0):>10.2f}%    |    {new_metrics.get('win_rate_pct', 0):>10.2f}%")
        print(f"Total Trades:           {old_metrics.get('total_trades', 0):>10}     |    {new_metrics.get('total_trades', 0):>10}")
        print(f"Profit Factor:          {old_metrics.get('profit_factor', 0):>10.2f}     |    {new_metrics.get('profit_factor', 0):>10.2f}")

        print("="*80)

        # Highlight improvements
        if new_metrics.get('sharpe_ratio', 0) > old_metrics.get('sharpe_ratio', 0):
            print("✅ ML Strategy has BETTER Sharpe Ratio")

        if new_metrics.get('annual_return_pct', 0) > old_metrics.get('annual_return_pct', 0):
            print("✅ ML Strategy has BETTER Annual Return")

        if new_metrics.get('win_rate_pct', 0) > old_metrics.get('win_rate_pct', 0):
            print("✅ ML Strategy has BETTER Win Rate")

        print("="*80 + "\n")
    else:
        print("\n⚠️  Old strategy results not found for comparison")
        print("Only showing new ML strategy results:\n")

        new_metrics = new_results.get('testing', {})
        print(f"Annual Return:    {new_metrics.get('annual_return_pct', 0):.2f}%")
        print(f"Sharpe Ratio:     {new_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown:     {new_metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"Win Rate:         {new_metrics.get('win_rate_pct', 0):.2f}%")
        print(f"Total Trades:     {new_metrics.get('total_trades', 0)}")
        print(f"Profit Factor:    {new_metrics.get('profit_factor', 0):.2f}")
        print("="*80 + "\n")


if __name__ == '__main__':
    # Run the backtest
    results = run_xdiv_ml_backtest()

    # Compare with old strategy if available
    compare_strategies()
