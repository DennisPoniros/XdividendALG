"""
Run Simple Dividend Alpha Strategy
Fast iteration to achieve Sharpe > 1
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from mock_data_manager import MockDataManager
from strategy_simple_dividend_alpha import SimpleDividendAlphaStrategy

def run_backtest(start_date='2023-01-01', end_date='2024-10-31', initial_capital=100_000):
    """Run simple backtest."""

    print("\n" + "="*80)
    print("🚀 SIMPLE DIVIDEND ALPHA STRATEGY")
    print("="*80)
    print(f"Period: {start_date} to {end_date}")
    print(f"Capital: ${initial_capital:,.0f}")
    print("="*80 + "\n")

    # Initialize
    dm = MockDataManager()
    strategy = SimpleDividendAlphaStrategy(dm)

    capital = initial_capital
    positions = {}
    trade_log = []
    equity_curve = []

    # Generate trading days
    date_range = pd.bdate_range(start=start_date, end=end_date)

    print(f"📊 Simulating {len(date_range)} trading days...\n")

    for idx, current_date in enumerate(date_range):
        current_date_str = current_date.strftime('%Y-%m-%d')

        # Check exits FIRST (including stops)
        exit_signals = strategy.generate_exit_signals(current_date_str, positions)

        for exit_sig in exit_signals:
            ticker = exit_sig['ticker']
            if ticker not in positions:
                continue

            pos = positions[ticker]

            # Get exit price
            try:
                prices = dm.get_stock_prices(ticker,
                                            (current_date - timedelta(days=3)).strftime('%Y-%m-%d'),
                                            current_date_str)
                exit_price = prices['close'].iloc[-1]
            except:
                continue

            # Calculate P&L
            pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price']
            pnl_dollars = pos['shares'] * (exit_price - pos['entry_price'])

            # Log trade
            days_held = (current_date - pd.to_datetime(pos['entry_date'])).days

            trade_log.append({
                'entry_date': pos['entry_date'],
                'exit_date': current_date_str,
                'ticker': ticker,
                'entry_price': pos['entry_price'],
                'exit_price': exit_price,
                'shares': pos['shares'],
                'pnl_pct': pnl_pct * 100,
                'pnl_dollars': pnl_dollars,
                'exit_reason': exit_sig['reason'],
                'days_held': days_held,
            })

            # Return capital
            capital += pos['shares'] * exit_price
            del positions[ticker]

        # Generate entry signals
        current_equity = capital + sum(p['shares'] * p.get('current_price', p['entry_price']) for p in positions.values())

        entry_signals = strategy.generate_entry_signals(current_date_str, current_equity, positions)

        for entry_sig in entry_signals:
            ticker = entry_sig['ticker']
            if ticker in positions:
                continue

            # Calculate shares
            position_value = current_equity * strategy.params['position_size']
            shares = int(position_value / entry_sig['price'])

            if shares <= 0:
                continue

            cost = shares * entry_sig['price']

            if cost > capital:
                continue

            # Enter position
            positions[ticker] = {
                'ticker': ticker,
                'entry_date': current_date_str,
                'entry_price': entry_sig['price'],
                'shares': shares,
                'ex_div_date': entry_sig.get('ex_div_date'),
                'current_price': entry_sig['price'],
            }

            capital -= cost

        # Update position values
        for ticker, pos in positions.items():
            try:
                prices = dm.get_stock_prices(ticker,
                                            (current_date - timedelta(days=3)).strftime('%Y-%m-%d'),
                                            current_date_str)
                pos['current_price'] = prices['close'].iloc[-1]
            except:
                pass

        # Record equity
        position_value = sum(p['shares'] * p.get('current_price', p['entry_price']) for p in positions.values())
        total_equity = capital + position_value

        equity_curve.append({
            'date': current_date_str,
            'equity': total_equity,
            'positions': len(positions),
        })

        # Progress
        if idx % 21 == 0 or idx == len(date_range) - 1:
            pct_return = ((total_equity - initial_capital) / initial_capital) * 100
            print(f"📈 {current_date_str}: Value=${total_equity:,.0f} ({pct_return:+.2f}%), "
                  f"Positions={len(positions)}, Trades={len(trade_log)}")

    # Calculate metrics
    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()

    final_value = equity_df['equity'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital

    # Annualized
    days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    years = days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Volatility
    equity_df['daily_return'] = equity_df['equity'].pct_change()
    daily_returns = equity_df['daily_return'].dropna()
    annual_vol = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0

    # Sharpe
    risk_free = 0.04 / 252
    excess_returns = daily_returns - risk_free
    sharpe = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)) if len(excess_returns) > 1 and excess_returns.std() > 0 else 0

    # Drawdown
    equity_df['cummax'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
    max_dd = abs(equity_df['drawdown'].min()) if len(equity_df) > 0 else 0

    # Trade stats
    if len(trades_df) > 0:
        wins = trades_df[trades_df['pnl_pct'] > 0]
        losses = trades_df[trades_df['pnl_pct'] <= 0]
        win_rate = len(wins) / len(trades_df) * 100
        avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0
        best = trades_df['pnl_pct'].max()
        worst = trades_df['pnl_pct'].min()
    else:
        win_rate = avg_win = avg_loss = best = worst = 0

    # Print results
    print("\n" + "="*80)
    print("📊 RESULTS")
    print("="*80)
    print(f"\n💰 Returns:")
    print(f"   Total: {total_return*100:+.2f}%")
    print(f"   Annual: {annual_return*100:+.2f}%")
    print(f"   Final Value: ${final_value:,.0f}")
    print(f"\n📈 Risk:")
    print(f"   Sharpe Ratio: {sharpe:.2f}")
    print(f"   Max Drawdown: {max_dd*100:.2f}%")
    print(f"   Volatility: {annual_vol*100:.2f}%")
    print(f"\n🎯 Trades:")
    print(f"   Total: {len(trades_df)}")
    print(f"   Win Rate: {win_rate:.1f}%")
    print(f"   Avg Win: {avg_win:+.2f}%")
    print(f"   Avg Loss: {avg_loss:+.2f}%")
    print(f"   Best: {best:+.2f}%")
    print(f"   Worst: {worst:+.2f}%")

    print("\n" + "="*80)
    if sharpe >= 1.0:
        print("✅ SUCCESS! Sharpe >= 1.0")
    else:
        print(f"⚠️  Sharpe {sharpe:.2f} below target")
    print("="*80 + "\n")

    return {'sharpe': sharpe, 'return': annual_return*100, 'trades': len(trades_df), 'win_rate': win_rate}

if __name__ == '__main__':
    results = run_backtest()
