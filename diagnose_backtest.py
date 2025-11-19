"""
Diagnostic Script - Debug Backtest Results

Run this after a backtest to see what went wrong.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle


def analyze_trade_log():
    """Analyze trade log to find issues."""

    print("\n" + "="*80)
    print("🔍 BACKTEST DIAGNOSTIC ANALYSIS")
    print("="*80 + "\n")

    # Find outputs directory
    output_dir = Path('/mnt/user-data/outputs')
    if not output_dir.exists():
        output_dir = Path('outputs')

    if not output_dir.exists():
        print("❌ No outputs directory found")
        return

    # Find trade log
    trade_files = list(output_dir.glob('*trade_log.csv'))
    if not trade_files:
        print("❌ No trade log found")
        return

    latest_trade_file = max(trade_files, key=lambda p: p.stat().st_mtime)
    print(f"📄 Reading: {latest_trade_file.name}\n")

    df = pd.read_csv(latest_trade_file)

    # Filter to exits only
    exits = df[df['action'] == 'EXIT'].copy()

    if len(exits) == 0:
        print("❌ No exit trades found")
        return

    print(f"📊 TOTAL TRADES: {len(exits)}\n")

    # Separate wins and losses
    exits['is_win'] = exits['pnl'] > 0
    wins = exits[exits['is_win']]
    losses = exits[~exits['is_win']]

    print("="*80)
    print("WIN/LOSS BREAKDOWN")
    print("="*80)
    print(f"Wins: {len(wins)} ({len(wins)/len(exits)*100:.1f}%)")
    print(f"Losses: {len(losses)} ({len(losses)/len(exits)*100:.1f}%)\n")

    # Calculate P&L stats
    print("="*80)
    print("P&L STATISTICS")
    print("="*80)
    print(f"Total P&L: ${exits['pnl'].sum():,.2f}")
    print(f"Avg P&L per trade: ${exits['pnl'].mean():,.2f}")
    print(f"\nWinning trades:")
    print(f"  Total P&L: ${wins['pnl'].sum():,.2f}")
    print(f"  Avg P&L: ${wins['pnl'].mean():,.2f}")
    print(f"  Median P&L: ${wins['pnl'].median():,.2f}")
    print(f"\nLosing trades:")
    print(f"  Total P&L: ${losses['pnl'].sum():,.2f}")
    print(f"  Avg P&L: ${losses['pnl'].mean():,.2f}")
    print(f"  Median P&L: ${losses['pnl'].median():,.2f}")

    # Check if entry_price and shares are available
    if 'entry_price' in exits.columns and 'shares' in exits.columns:
        print("\n" + "="*80)
        print("POSITION SIZING ANALYSIS")
        print("="*80)

        exits['position_value'] = exits['entry_price'] * exits['shares']

        print(f"Avg position value: ${exits['position_value'].mean():,.2f}")
        print(f"Min position value: ${exits['position_value'].min():,.2f}")
        print(f"Max position value: ${exits['position_value'].max():,.2f}")
        print(f"Std position value: ${exits['position_value'].std():,.2f}")

        # Check if winners/losers have different sizing
        if len(wins) > 0:
            wins['position_value'] = wins['entry_price'] * wins['shares']
            print(f"\nWinning trades avg position: ${wins['position_value'].mean():,.2f}")

        if len(losses) > 0:
            losses['position_value'] = losses['entry_price'] * losses['shares']
            print(f"Losing trades avg position: ${losses['position_value'].mean():,.2f}")

            # This is the KEY metric
            if len(wins) > 0:
                ratio = losses['position_value'].mean() / wins['position_value'].mean()
                print(f"\n⚠️  POSITION SIZE RATIO (Loss/Win): {ratio:.2f}x")
                if ratio > 1.5:
                    print("   🔴 PROBLEM: Losing trades have LARGER positions!")
                elif ratio < 0.67:
                    print("   🟡 WARNING: Winning trades have much larger positions")
                else:
                    print("   ✅ Position sizing looks balanced")

    # Exit reason analysis
    if 'exit_reason' in exits.columns:
        print("\n" + "="*80)
        print("EXIT REASON ANALYSIS")
        print("="*80)

        exit_reasons = exits.groupby('exit_reason').agg({
            'pnl': ['count', 'sum', 'mean']
        })
        exit_reasons.columns = ['Count', 'Total P&L', 'Avg P&L']
        exit_reasons = exit_reasons.sort_values('Total P&L', ascending=False)

        print(exit_reasons.to_string())

        # Check for stop losses
        stop_loss_trades = exits[exits['exit_reason'].str.contains('stop', case=False, na=False)]
        if len(stop_loss_trades) > 0:
            print(f"\n⚠️  {len(stop_loss_trades)} trades exited via STOP LOSS")
            print(f"   Total P&L from stops: ${stop_loss_trades['pnl'].sum():,.2f}")
            print(f"   🔴 PROBLEM: Stop losses should be DISABLED!")

    # Outlier trades
    print("\n" + "="*80)
    print("OUTLIER TRADES")
    print("="*80)

    print("\n🔴 WORST 5 TRADES:")
    worst = exits.nsmallest(5, 'pnl')[['date', 'ticker', 'pnl', 'exit_reason']]
    print(worst.to_string(index=False))

    print("\n🟢 BEST 5 TRADES:")
    best = exits.nlargest(5, 'pnl')[['date', 'ticker', 'pnl', 'exit_reason']]
    print(best.to_string(index=False))

    # Concentration analysis
    print("\n" + "="*80)
    print("CONCENTRATION ANALYSIS")
    print("="*80)

    # Top 10% of trades
    n_top = max(1, len(exits) // 10)
    top_trades = exits.nlargest(n_top, 'pnl')
    bottom_trades = exits.nsmallest(n_top, 'pnl')

    top_pnl = top_trades['pnl'].sum()
    bottom_pnl = bottom_trades['pnl'].sum()
    total_pnl = exits['pnl'].sum()

    print(f"Top 10% of trades ({n_top} trades):")
    print(f"  P&L: ${top_pnl:,.2f}")
    print(f"  % of total: {top_pnl/total_pnl*100:.1f}%")

    print(f"\nBottom 10% of trades ({n_top} trades):")
    print(f"  P&L: ${bottom_pnl:,.2f}")
    print(f"  % of total: {bottom_pnl/total_pnl*100:.1f}%")

    # Check for single huge loser
    worst_trade_pnl = exits['pnl'].min()
    if worst_trade_pnl < -5000:
        print(f"\n🔴 ALERT: Single trade lost ${abs(worst_trade_pnl):,.2f}!")
        print(f"   This is {abs(worst_trade_pnl)/abs(total_pnl)*100:.1f}% of total losses")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    # Calculate expected vs actual
    avg_win_pct = (wins['pnl'] / (wins['entry_price'] * wins['shares'])).mean() if len(wins) > 0 else 0
    avg_loss_pct = (losses['pnl'] / (losses['entry_price'] * losses['shares'])).mean() if len(losses) > 0 else 0

    print(f"Avg win: {avg_win_pct*100:.2f}%")
    print(f"Avg loss: {avg_loss_pct*100:.2f}%")
    print(f"Win rate: {len(wins)/len(exits)*100:.1f}%")

    if 'position_value' in exits.columns:
        expected_pnl = (
            len(wins) * wins['position_value'].mean() * avg_win_pct +
            len(losses) * losses['position_value'].mean() * avg_loss_pct
        )
        actual_pnl = exits['pnl'].sum()

        print(f"\n💰 EXPECTED P&L (from stats): ${expected_pnl:,.2f}")
        print(f"💰 ACTUAL P&L: ${actual_pnl:,.2f}")
        print(f"❓ DISCREPANCY: ${actual_pnl - expected_pnl:,.2f}")

        if abs(actual_pnl - expected_pnl) > 1000:
            print("\n🔴 LARGE DISCREPANCY DETECTED!")
            print("   Possible causes:")
            print("   - Position sizing issues")
            print("   - Outlier trades skewing averages")
            print("   - PnL calculation errors")

    print("\n" + "="*80)


if __name__ == '__main__':
    analyze_trade_log()
