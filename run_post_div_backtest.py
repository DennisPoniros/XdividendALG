"""
Run Post-Dividend Dip Strategy Backtest

STRATEGY: Buy AFTER ex-dividend when price drops, sell at mean reversion
"""

import os
import pickle
import pandas as pd
from datetime import datetime

from backtester_post_div import PostDivDipBacktester


def run_post_div_backtest(
    start_date: str = '2023-01-01',
    end_date: str = '2024-10-31',
    initial_capital: float = 100_000
):
    """
    Run post-dividend dip strategy backtest.

    Args:
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital

    Returns:
        Results dictionary
    """

    print("\n" + "="*80)
    print("🔄 POST-DIVIDEND DIP STRATEGY")
    print("="*80)
    print("\n📋 Strategy Overview:")
    print("  1. Monitor stocks going ex-dividend")
    print("  2. Wait for price to drop (0-2 days after ex-div)")
    print("  3. Buy the dip when stock is oversold (RSI < 40)")
    print("  4. Exit when price mean-reverts to pre-dividend level")
    print("  5. Or exit after 10 days max holding")
    print("\n✨ Key Advantages:")
    print("  ✓ No dividend taxation (avoid 15-37% tax)")
    print("  ✓ Buy at discount (lower entry price)")
    print("  ✓ Same mean reversion opportunity")
    print("  ✓ Simpler tax treatment (capital gains)")
    print("="*80 + "\n")

    # Create backtester
    bt = PostDivDipBacktester(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )

    # Run backtest
    results = bt.run_backtest()

    # Create output directory (platform-aware)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # Print absolute path
    abs_output_dir = os.path.abspath(output_dir)
    print(f"\n📁 Output directory: {abs_output_dir}")

    # Save results
    print("\n📁 Saving results...")

    # Pickle
    results_file = os.path.join(output_dir, 'post_div_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump({
            'results': results,
            'equity_curve': bt.equity_curve,
            'trade_log': bt.trade_log,
        }, f)
    print(f"✅ Saved pickle: {results_file}")

    # Trade log CSV
    if len(bt.trade_log) > 0:
        trade_log_file = os.path.join(output_dir, 'post_div_trade_log.csv')
        trade_df = pd.DataFrame(bt.trade_log)
        trade_df.to_csv(trade_log_file, index=False)
        print(f"✅ Saved trade log: {trade_log_file}")

    # Equity curve CSV
    if len(bt.equity_curve) > 0:
        equity_file = os.path.join(output_dir, 'post_div_equity_curve.csv')
        equity_df = pd.DataFrame(bt.equity_curve)
        equity_df.to_csv(equity_file, index=False)
        print(f"✅ Saved equity curve: {equity_file}")

    # Summary text file
    summary_file = os.path.join(output_dir, 'post_div_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("POST-DIVIDEND DIP STRATEGY - BACKTEST RESULTS\n")
        f.write("="*80 + "\n\n")

        f.write(f"Period: {start_date} to {end_date}\n")
        f.write(f"Initial Capital: ${initial_capital:,.0f}\n\n")

        f.write("PERFORMANCE\n")
        f.write("-"*80 + "\n")
        f.write(f"Final Value: ${results['final_equity']:,.0f}\n")
        f.write(f"Total Return: {results['total_return_pct']:.2f}%\n")
        f.write(f"Annual Return: {results['annual_return_pct']:.2f}%\n")
        f.write(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n")
        f.write(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%\n\n")

        f.write("TRADING STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Trades: {results['total_trades']:.0f}\n")
        f.write(f"Win Rate: {results['win_rate_pct']:.1f}%\n")
        f.write(f"Profit Factor: {results['profit_factor']:.2f}\n")
        f.write(f"Avg Holding: {results['avg_hold_days']:.1f} days\n")

    print(f"✅ Saved summary: {summary_file}")

    print("\n" + "="*80)
    print("✅ BACKTEST COMPLETED")
    print("="*80)
    print(f"\nResults saved to: {abs_output_dir}/")
    print("="*80 + "\n")

    return results


if __name__ == '__main__':
    results = run_post_div_backtest()
