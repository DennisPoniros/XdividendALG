"""
Run Adaptive Dividend Mean Reversion Strategy Backtest
Iterative optimization to achieve Sharpe > 1
"""

import pandas as pd
import pickle
import os
from datetime import datetime

from backtester_adaptive import AdaptiveBacktester


def run_backtest(
    start_date='2023-01-01',
    end_date='2024-10-31',
    initial_capital=100_000,
    use_mock_data=True,
    save_results=True
):
    """
    Run backtest for Adaptive strategy.
    """

    print("\n" + "="*80)
    print("🚀 ADAPTIVE DIVIDEND MEAN REVERSION - BACKTEST v1")
    print("="*80)
    print(f"Period: {start_date} to {end_date}")
    print(f"Capital: ${initial_capital:,.0f}")
    print(f"Data: {'MOCK (synthetic)' if use_mock_data else 'REAL (yfinance)'}")
    print("="*80 + "\n")

    # Create backtester
    bt = AdaptiveBacktester(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        use_mock_data=use_mock_data
    )

    # Run backtest
    results = bt.run_backtest()

    # Save results if requested
    if save_results and results:
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(output_dir, exist_ok=True)

        # Save pickle
        results_file = os.path.join(output_dir, 'adaptive_v1_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump({
                'results': results,
                'equity_curve': bt.equity_curve,
                'trade_log': bt.trade_log,
                'strategy_params': bt.strategy.params,
            }, f)
        print(f"✅ Saved results: {results_file}")

        # Save trade log CSV
        if bt.trade_log:
            trade_file = os.path.join(output_dir, 'adaptive_v1_trades.csv')
            pd.DataFrame(bt.trade_log).to_csv(trade_file, index=False)
            print(f"✅ Saved trades: {trade_file}")

        # Save equity curve CSV
        if bt.equity_curve:
            equity_file = os.path.join(output_dir, 'adaptive_v1_equity.csv')
            pd.DataFrame(bt.equity_curve).to_csv(equity_file, index=False)
            print(f"✅ Saved equity curve: {equity_file}")

        # Save summary text
        summary_file = os.path.join(output_dir, 'adaptive_v1_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ADAPTIVE DIVIDEND MEAN REVERSION STRATEGY v1\n")
            f.write("="*80 + "\n\n")

            f.write(f"Period: {start_date} to {end_date}\n")
            f.write(f"Initial Capital: ${initial_capital:,.0f}\n")
            f.write(f"Data Source: {'Mock' if use_mock_data else 'Real'}\n\n")

            f.write("PERFORMANCE METRICS\n")
            f.write("-"*80 + "\n")
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    if 'pct' in key:
                        f.write(f"{key:.<40} {value:>10.2f}%\n")
                    elif 'ratio' in key.lower():
                        f.write(f"{key:.<40} {value:>10.2f}\n")
                    else:
                        f.write(f"{key:.<40} {value:>10,.2f}\n")

            f.write("\n" + "="*80 + "\n")

        print(f"✅ Saved summary: {summary_file}")

    return results


if __name__ == '__main__':
    # Run with mock data for fast iteration
    results = run_backtest(use_mock_data=True)

    # Print recommendation
    print("\n" + "="*80)
    if results['sharpe_ratio'] >= 1.0:
        print("✅ SUCCESS! Strategy meets target. Ready to test on real data.")
        print("\nTo run on real data:")
        print("   python run_adaptive_backtest.py --real")
    else:
        print(f"⚠️  Sharpe {results['sharpe_ratio']:.2f} below target 1.0")
        print("\nIterating to improve strategy...")
    print("="*80 + "\n")
