#!/usr/bin/env python3
"""
Test script for dashboard components.

Tests:
- Metrics calculator
- Visualizations
- Data interface
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from dashboard.metrics import MetricsCalculator
from dashboard.visualizations import DashboardVisualizations
from dashboard.data_interface import BacktestDataInterface


def create_test_data():
    """Create synthetic backtest data for testing."""
    print("Creating test data...")

    # Create date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 1, 1)
    dates = pd.date_range(start_date, end_date, freq='D')

    # Create synthetic equity curve (random walk with drift)
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, len(dates))  # 0.05% daily return, 1% std
    equity_curve = pd.Series(100000 * (1 + returns).cumprod(), index=dates)

    # Create synthetic trades
    trades = []
    trade_dates = pd.date_range(start_date, end_date, freq='W')  # Weekly trades

    for i, date in enumerate(trade_dates[:-1]):
        ticker = f"STOCK{i % 5 + 1}"
        shares = np.random.randint(100, 500)
        entry_price = np.random.uniform(50, 150)
        exit_price = entry_price * np.random.uniform(0.95, 1.10)
        pnl = shares * (exit_price - entry_price)

        # Entry trade
        trades.append({
            'date': date,
            'action': 'ENTRY',
            'ticker': ticker,
            'shares': shares,
            'price': entry_price,
            'pnl': 0,
            'exit_reason': None
        })

        # Exit trade
        exit_date = date + timedelta(days=np.random.randint(1, 7))
        trades.append({
            'date': exit_date,
            'action': 'EXIT',
            'ticker': ticker,
            'shares': shares,
            'price': exit_price,
            'pnl': pnl,
            'exit_reason': 'Take Profit' if pnl > 0 else 'Stop Loss',
            'entry_date': date,
            'exit_date': exit_date,
        })

    # Create position values (sum of open positions)
    position_values = pd.Series(
        np.random.uniform(0.5, 0.8, len(dates)) * equity_curve.values,
        index=dates
    )

    return equity_curve, trades, position_values


def test_metrics_calculator():
    """Test metrics calculator."""
    print("\n" + "="*60)
    print("Testing MetricsCalculator...")
    print("="*60)

    equity_curve, trades, _ = create_test_data()

    calculator = MetricsCalculator(risk_free_rate=0.02)

    # Calculate metrics
    metrics = calculator.calculate_all_metrics(
        equity_curve=equity_curve,
        trades=trades,
        benchmark_returns=None
    )

    # Display key metrics
    print(f"\n✓ Metrics calculated successfully")
    print(f"\n  Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"  CAGR: {metrics['cagr_pct']:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"  Omega Ratio: {metrics['omega_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Win Rate: {metrics['win_rate_pct']:.1f}%")
    print(f"  Total Trades: {metrics['total_trades']}")

    return True


def test_visualizations():
    """Test visualization components."""
    print("\n" + "="*60)
    print("Testing DashboardVisualizations...")
    print("="*60)

    equity_curve, trades, position_values = create_test_data()
    returns = equity_curve.pct_change().dropna()

    viz = DashboardVisualizations(theme='plotly_dark')

    # Test each visualization
    tests = [
        ("Equity Curve", lambda: viz.plot_equity_curve(equity_curve, trades)),
        ("Drawdown", lambda: viz.plot_drawdown(equity_curve)),
        ("P&L Metrics", lambda: viz.plot_pnl_metrics({
            'sharpe_ratio': 1.5,
            'sortino_ratio': 1.8,
            'omega_ratio': 1.3,
            'calmar_ratio': 1.2
        })),
        ("Returns Histogram", lambda: viz.plot_returns_histogram(returns)),
        ("Win Rate Analysis", lambda: viz.plot_win_rate_analysis(trades)),
        ("Trade Heatmap", lambda: viz.plot_trade_heatmap(trades)),
        ("Outlier Trades", lambda: viz.plot_outlier_trades(trades)),
        ("Rolling Metrics", lambda: viz.plot_rolling_metrics(returns)),
        ("Leverage Utilization", lambda: viz.plot_leverage_utilization(equity_curve, position_values)),
    ]

    for name, test_func in tests:
        try:
            fig = test_func()
            print(f"  ✓ {name} plot created")
        except Exception as e:
            print(f"  ✗ {name} plot failed: {e}")
            return False

    return True


def test_data_interface():
    """Test data interface."""
    print("\n" + "="*60)
    print("Testing BacktestDataInterface...")
    print("="*60)

    # Create test data
    equity_curve, trades, position_values = create_test_data()

    # Test replay mode
    interface = BacktestDataInterface(mode='replay')

    # Save test results
    interface.save_results(
        name='test_backtest',
        equity_curve=equity_curve,
        trades=trades,
        positions={},
        position_values=position_values,
        config={'test': True}
    )
    print("  ✓ Results saved")

    # Load results
    results = interface.load_backtest_results('test_backtest')
    print(f"  ✓ Results loaded: {len(results['equity_curve'])} data points")

    # List backtests
    backtests = interface.list_available_backtests()
    print(f"  ✓ Found {len(backtests)} backtest(s)")

    # Get summary stats
    stats = interface.get_summary_stats()
    print(f"  ✓ Summary stats: {stats['total_trades']} trades, ${stats['total_pnl']:,.2f} P&L")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("DASHBOARD COMPONENT TESTING")
    print("="*60)

    tests = [
        ("Metrics Calculator", test_metrics_calculator),
        ("Visualizations", test_visualizations),
        ("Data Interface", test_data_interface),
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
        print("\n🎉 All tests passed! Dashboard is ready to use.")
        print("\nTo launch the dashboard:")
        print("  python dashboard/run_dashboard.py")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
