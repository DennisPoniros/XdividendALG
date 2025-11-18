#!/usr/bin/env python3
"""
Test Monte Carlo and Attribution analysis features.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent))

from dashboard.monte_carlo import MonteCarloSimulator
from dashboard.attribution import AttributionAnalyzer
from dashboard.data_interface import BacktestDataInterface


def create_test_trades():
    """Create test trades with cost information."""
    np.random.seed(42)

    trades = []
    start_date = datetime(2023, 1, 1)

    for i in range(50):
        ticker = f"STOCK{i % 10}"
        shares = np.random.randint(100, 500)
        entry_price = np.random.uniform(50, 150)
        exit_price = entry_price * np.random.uniform(0.95, 1.10)

        gross_pnl = shares * (exit_price - entry_price)

        # Add costs
        slippage_entry = shares * entry_price * 0.0005  # 5 bps
        slippage_exit = shares * exit_price * 0.0005
        fees_entry = shares * entry_price * 0.0000278  # SEC fees
        fees_exit = shares * exit_price * 0.0000278

        total_costs = slippage_entry + slippage_exit + fees_entry + fees_exit
        net_pnl = gross_pnl - total_costs

        entry_date = start_date + timedelta(days=i*7)
        exit_date = entry_date + timedelta(days=np.random.randint(1, 7))

        # Entry trade
        trades.append({
            'date': entry_date,
            'action': 'ENTRY',
            'ticker': ticker,
            'shares': shares,
            'price': entry_price,
        })

        # Exit trade
        exit_reasons = ['Take Profit', 'Stop Loss', 'Time Limit', 'Post-Dividend']
        trades.append({
            'date': exit_date,
            'action': 'EXIT',
            'ticker': ticker,
            'shares': shares,
            'price': exit_price,
            'pnl': net_pnl,
            'exit_reason': np.random.choice(exit_reasons),
            'entry_date': entry_date,
            'exit_date': exit_date,
            'slippage_entry': slippage_entry,
            'slippage_exit': slippage_exit,
            'fees_entry': fees_entry,
            'fees_exit': fees_exit,
            'commission_entry': 0,
            'commission_exit': 0,
        })

    return trades


def test_monte_carlo():
    """Test Monte Carlo simulation."""
    print("\n" + "="*60)
    print("Testing Monte Carlo Simulation...")
    print("="*60)

    trades = create_test_trades()

    mc = MonteCarloSimulator(n_simulations=100, random_seed=42)
    results = mc.run_simulation(trades)

    if 'error' in results:
        print(f"✗ Error: {results['error']}")
        return False

    print(f"\n✓ Simulation completed")
    print(f"  Simulations: {results['n_simulations']}")
    print(f"  Trades analyzed: {results['n_trades']}")
    print(f"\n  Actual Equity: ${results['actual_equity']:,.2f}")
    print(f"  Median Simulated: ${results['equity_percentiles']['50th']:,.2f}")
    print(f"  Actual Percentile: {results['actual_equity_percentile']:.1f}%")
    print(f"\n  Probability of Profit: {results['prob_profit']:.1%}")
    print(f"  Probability Sharpe > 1: {results['prob_sharpe_gt_1']:.1%}")

    # Test robustness assessment
    robustness = mc.assess_robustness(results)
    print(f"\n  Robustness Score: {robustness['robustness_score']}/100")
    print(f"  Assessment: {robustness['assessment']}")

    return True


def test_attribution():
    """Test attribution analysis."""
    print("\n" + "="*60)
    print("Testing Attribution Analysis...")
    print("="*60)

    trades = create_test_trades()

    analyzer = AttributionAnalyzer()

    # Cost attribution
    print("\n--- Cost Attribution ---")
    cost_attr = analyzer.analyze_cost_attribution(trades)

    print(f"✓ Cost attribution completed")
    print(f"  Total Costs: ${cost_attr['total_costs']:,.2f}")
    print(f"  Slippage: ${cost_attr['total_slippage']:,.2f}")
    print(f"  Fees: ${cost_attr['total_fees']:,.2f}")
    print(f"  Commission: ${cost_attr['total_commission']:,.2f}")
    print(f"  Costs as % of Gross: {cost_attr['costs_as_pct_gross']:.2f}%")

    # Win/loss attribution
    print("\n--- Win/Loss Attribution ---")
    winlose_attr = analyzer.analyze_win_lose_attribution(trades)

    print(f"✓ Win/loss attribution completed")
    print(f"  Total Trades: {winlose_attr['total_trades']}")
    print(f"  Win Rate: {winlose_attr['win_rate']:.1f}%")
    print(f"  Total P&L: ${winlose_attr['total_pnl']:,.2f}")
    print(f"  Top 10% Concentration: {winlose_attr['concentration_top10pct']:.1f}%")
    print(f"  Top 20% Concentration: {winlose_attr['concentration_top20pct']:.1f}%")

    print(f"\n  Top 5 Tickers by P&L:")
    for ticker, data in list(winlose_attr['by_ticker'].head(5).iterrows()):
        print(f"    {ticker}: ${data['total_pnl']:,.2f} ({data['win_rate']:.1f}% WR)")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("MONTE CARLO & ATTRIBUTION FEATURE TESTING")
    print("="*60)

    tests = [
        ("Monte Carlo Simulation", test_monte_carlo),
        ("Attribution Analysis", test_attribution),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! New features are ready to use.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
