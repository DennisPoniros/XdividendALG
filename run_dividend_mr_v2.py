"""
Run Dividend Mean Reversion V2 Strategy
More aggressive version for higher Sharpe
"""

import pandas as pd
import os
from mock_data_manager import MockDataManager
from strategy_dividend_mr_v2 import DividendMeanReversionV2
from backtester_enhanced_mr import EnhancedMRBacktester


class V2Backtester(EnhancedMRBacktester):
    """Backtester for V2 strategy with mock data"""

    def __init__(self, start_date: str, end_date: str, initial_capital: float):
        super().__init__(start_date, end_date, initial_capital)

        # Use mock data and V2 strategy
        self.dm = MockDataManager(seed=42)
        self.strategy = DividendMeanReversionV2(self.dm)


def run_v2_backtest():
    """Run V2 backtest"""

    print("\n" + "="*80)
    print("🚀 DIVIDEND MEAN REVERSION V2 - AGGRESSIVE")
    print("="*80)
    print("Targeting high trade frequency for consistent alpha")
    print("="*80 + "\n")

    bt = V2Backtester(
        start_date='2023-01-01',
        end_date='2024-10-31',
        initial_capital=100_000
    )

    results = bt.run_backtest()

    if results:
        # Save
        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)

        # CSV files
        if bt.trade_log:
            pd.DataFrame(bt.trade_log).to_csv(f'{output_dir}/v2_trades.csv', index=False)
            print(f"\n✅ Saved trades: {output_dir}/v2_trades.csv")

        if bt.equity_curve:
            pd.DataFrame(bt.equity_curve).to_csv(f'{output_dir}/v2_equity.csv', index=False)
            print(f"✅ Saved equity: {output_dir}/v2_equity.csv")

        # Assessment
        sharpe = results.get('sharpe_ratio', 0)
        trades = results.get('total_trades', 0)
        annual_return = results.get('annual_return_pct', 0)

        print("\n" + "="*80)
        print("📊 V2 ASSESSMENT")
        print("="*80)
        print(f"Trades:        {trades}")
        print(f"Sharpe Ratio:  {sharpe:.2f}")
        print(f"Annual Return: {annual_return:.2f}%")

        if sharpe >= 1.0 and trades >= 20:
            print("\n✅✅✅ SUCCESS! Target achieved!")
            print(f"   Sharpe {sharpe:.2f} > 1.0")
            print(f"   {trades} trades executed")
        elif sharpe >= 0.7:
            print("\n⚠️  CLOSE - Needs minor tuning")
        elif trades < 10:
            print("\n❌ ISSUE: Not enough trades")
            print("   Need to relax filters further")
        else:
            print("\n❌ ISSUE: Sharpe too low")
            print("   Need better risk/reward")

        print("="*80 + "\n")

    return results


if __name__ == '__main__':
    run_v2_backtest()
