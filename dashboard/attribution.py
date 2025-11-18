"""
Attribution analysis for understanding sources of returns.

Breaks down performance by:
- Costs (slippage, fees, commissions)
- Win/loss patterns (by ticker, time, reason)
- Trade characteristics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict


class AttributionAnalyzer:
    """Analyze sources of returns and costs."""

    def __init__(self):
        """Initialize attribution analyzer."""
        pass

    def analyze_cost_attribution(self, trades: List[Dict]) -> Dict:
        """
        Break down all costs and their impact on returns.

        Args:
            trades: List of trades with cost information

        Returns:
            Dictionary with cost attribution analysis
        """
        closed_trades = [t for t in trades if t.get('action') == 'EXIT']

        if not closed_trades:
            return self._empty_cost_attribution()

        # Extract cost components
        total_pnl = sum(t.get('pnl', 0) for t in closed_trades)

        # Initialize cost tracking
        costs = {
            'slippage_entry': [],
            'slippage_exit': [],
            'fees_entry': [],
            'fees_exit': [],
            'commission_entry': [],
            'commission_exit': [],
            'total_costs': [],
        }

        gross_pnls = []
        net_pnls = []
        tickers = []
        dates = []

        for trade in closed_trades:
            # Try to extract costs (may not be in all trade formats)
            slippage_entry = trade.get('slippage_entry', 0)
            slippage_exit = trade.get('slippage_exit', 0)
            fees_entry = trade.get('fees_entry', 0)
            fees_exit = trade.get('fees_exit', 0)
            commission_entry = trade.get('commission_entry', 0)
            commission_exit = trade.get('commission_exit', 0)

            total_cost = (slippage_entry + slippage_exit +
                         fees_entry + fees_exit +
                         commission_entry + commission_exit)

            costs['slippage_entry'].append(slippage_entry)
            costs['slippage_exit'].append(slippage_exit)
            costs['fees_entry'].append(fees_entry)
            costs['fees_exit'].append(fees_exit)
            costs['commission_entry'].append(commission_entry)
            costs['commission_exit'].append(commission_exit)
            costs['total_costs'].append(total_cost)

            net_pnl = trade.get('pnl', 0)
            gross_pnl = net_pnl + total_cost  # Approximate gross P&L

            gross_pnls.append(gross_pnl)
            net_pnls.append(net_pnl)
            tickers.append(trade.get('ticker', 'UNKNOWN'))
            dates.append(trade.get('date'))

        # Calculate totals
        total_slippage = sum(costs['slippage_entry']) + sum(costs['slippage_exit'])
        total_fees = sum(costs['fees_entry']) + sum(costs['fees_exit'])
        total_commission = sum(costs['commission_entry']) + sum(costs['commission_exit'])
        total_all_costs = sum(costs['total_costs'])

        gross_total = sum(gross_pnls)
        net_total = sum(net_pnls)

        # Calculate percentages
        cost_as_pct_gross = (total_all_costs / gross_total * 100) if gross_total > 0 else 0
        slippage_pct = (total_slippage / total_all_costs * 100) if total_all_costs > 0 else 0
        fees_pct = (total_fees / total_all_costs * 100) if total_all_costs > 0 else 0
        commission_pct = (total_commission / total_all_costs * 100) if total_all_costs > 0 else 0

        # Cost by ticker
        ticker_costs = defaultdict(lambda: {'costs': 0, 'gross_pnl': 0, 'count': 0})
        for ticker, cost, gross_pnl in zip(tickers, costs['total_costs'], gross_pnls):
            ticker_costs[ticker]['costs'] += cost
            ticker_costs[ticker]['gross_pnl'] += gross_pnl
            ticker_costs[ticker]['count'] += 1

        # Calculate cost per trade by ticker
        ticker_cost_analysis = {}
        for ticker, data in ticker_costs.items():
            avg_cost = data['costs'] / data['count'] if data['count'] > 0 else 0
            cost_pct = (data['costs'] / data['gross_pnl'] * 100) if data['gross_pnl'] > 0 else 0
            ticker_cost_analysis[ticker] = {
                'total_costs': data['costs'],
                'gross_pnl': data['gross_pnl'],
                'trade_count': data['count'],
                'avg_cost_per_trade': avg_cost,
                'cost_as_pct_gross': cost_pct,
            }

        # Sort by total costs
        top_cost_tickers = sorted(
            ticker_cost_analysis.items(),
            key=lambda x: x[1]['total_costs'],
            reverse=True
        )[:10]

        # Cost over time analysis
        df = pd.DataFrame({
            'date': dates,
            'cost': costs['total_costs'],
            'gross_pnl': gross_pnls,
        })
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        # Monthly aggregation
        monthly_costs = df.groupby(pd.Grouper(freq='M')).agg({
            'cost': 'sum',
            'gross_pnl': 'sum',
        })
        monthly_costs['cost_pct'] = (monthly_costs['cost'] / monthly_costs['gross_pnl'] * 100).fillna(0)

        return {
            # Total cost breakdown
            'total_costs': total_all_costs,
            'total_slippage': total_slippage,
            'total_fees': total_fees,
            'total_commission': total_commission,

            # As percentage of costs
            'slippage_pct_of_costs': slippage_pct,
            'fees_pct_of_costs': fees_pct,
            'commission_pct_of_costs': commission_pct,

            # Impact on returns
            'gross_pnl': gross_total,
            'net_pnl': net_total,
            'costs_as_pct_gross': cost_as_pct_gross,
            'avg_cost_per_trade': total_all_costs / len(closed_trades),

            # By ticker
            'ticker_analysis': dict(top_cost_tickers),

            # Over time
            'monthly_costs': monthly_costs,

            # Trade-level details
            'trade_costs': pd.DataFrame({
                'ticker': tickers,
                'date': dates,
                'gross_pnl': gross_pnls,
                'net_pnl': net_pnls,
                'total_cost': costs['total_costs'],
                'slippage': [e + x for e, x in zip(costs['slippage_entry'], costs['slippage_exit'])],
                'fees': [e + x for e, x in zip(costs['fees_entry'], costs['fees_exit'])],
                'commission': [e + x for e, x in zip(costs['commission_entry'], costs['commission_exit'])],
            }),
        }

    def analyze_win_lose_attribution(self, trades: List[Dict]) -> Dict:
        """
        Analyze win/loss patterns to understand what drives performance.

        Args:
            trades: List of trades

        Returns:
            Dictionary with win/loss attribution
        """
        closed_trades = [t for t in trades if t.get('action') == 'EXIT']

        if not closed_trades:
            return self._empty_winlose_attribution()

        # Create DataFrame for easier analysis
        df = pd.DataFrame([{
            'ticker': t.get('ticker', 'UNKNOWN'),
            'date': pd.to_datetime(t.get('date')),
            'entry_date': pd.to_datetime(t.get('entry_date')) if 'entry_date' in t else None,
            'exit_date': pd.to_datetime(t.get('exit_date')) if 'exit_date' in t else None,
            'pnl': t.get('pnl', 0),
            'exit_reason': t.get('exit_reason', 'UNKNOWN'),
            'shares': t.get('shares', 0),
            'price': t.get('price', 0),
        } for t in closed_trades])

        # Win/Loss classification
        df['is_winner'] = df['pnl'] > 0
        df['is_loser'] = df['pnl'] < 0

        # Attribution by ticker
        ticker_attribution = df.groupby('ticker').agg({
            'pnl': ['sum', 'count', 'mean'],
            'is_winner': 'sum',
            'is_loser': 'sum',
        }).round(2)
        ticker_attribution.columns = ['total_pnl', 'trade_count', 'avg_pnl', 'wins', 'losses']
        ticker_attribution['win_rate'] = (ticker_attribution['wins'] / ticker_attribution['trade_count'] * 100).round(1)
        ticker_attribution = ticker_attribution.sort_values('total_pnl', ascending=False)

        # Top contributors
        top_winners = ticker_attribution.nlargest(10, 'total_pnl')
        top_losers = ticker_attribution.nsmallest(10, 'total_pnl')

        # Attribution by exit reason
        exit_reason_attribution = df.groupby('exit_reason').agg({
            'pnl': ['sum', 'count', 'mean'],
            'is_winner': 'sum',
        }).round(2)
        exit_reason_attribution.columns = ['total_pnl', 'trade_count', 'avg_pnl', 'wins']
        exit_reason_attribution['win_rate'] = (
            exit_reason_attribution['wins'] / exit_reason_attribution['trade_count'] * 100
        ).round(1)
        exit_reason_attribution = exit_reason_attribution.sort_values('total_pnl', ascending=False)

        # Attribution by time period
        df['year_month'] = df['date'].dt.to_period('M')
        df['quarter'] = df['date'].dt.to_period('Q')
        df['weekday'] = df['date'].dt.day_name()

        monthly_attribution = df.groupby('year_month').agg({
            'pnl': ['sum', 'count'],
            'is_winner': 'sum',
        })
        monthly_attribution.columns = ['total_pnl', 'trade_count', 'wins']
        monthly_attribution['win_rate'] = (
            monthly_attribution['wins'] / monthly_attribution['trade_count'] * 100
        ).round(1)

        weekday_attribution = df.groupby('weekday').agg({
            'pnl': ['sum', 'count', 'mean'],
            'is_winner': 'sum',
        }).round(2)
        weekday_attribution.columns = ['total_pnl', 'trade_count', 'avg_pnl', 'wins']
        weekday_attribution['win_rate'] = (
            weekday_attribution['wins'] / weekday_attribution['trade_count'] * 100
        ).round(1)

        # Hold duration analysis (if entry_date available)
        if 'entry_date' in df.columns and df['entry_date'].notna().any():
            df['hold_days'] = (df['exit_date'] - df['entry_date']).dt.days

            # Group by hold duration buckets
            df['hold_bucket'] = pd.cut(
                df['hold_days'],
                bins=[0, 1, 3, 5, 7, 14, 30, 100],
                labels=['1 day', '2-3 days', '4-5 days', '6-7 days', '1-2 weeks', '2-4 weeks', '4+ weeks']
            )

            hold_attribution = df.groupby('hold_bucket').agg({
                'pnl': ['sum', 'count', 'mean'],
                'is_winner': 'sum',
            }).round(2)
            hold_attribution.columns = ['total_pnl', 'trade_count', 'avg_pnl', 'wins']
            hold_attribution['win_rate'] = (
                hold_attribution['wins'] / hold_attribution['trade_count'] * 100
            ).round(1)
        else:
            hold_attribution = pd.DataFrame()

        # Outlier analysis
        pnl_std = df['pnl'].std()
        pnl_mean = df['pnl'].mean()
        outlier_threshold = pnl_mean + 2 * pnl_std

        outliers_positive = df[df['pnl'] > outlier_threshold].sort_values('pnl', ascending=False)
        outliers_negative = df[df['pnl'] < (pnl_mean - 2 * pnl_std)].sort_values('pnl')

        # Contribution analysis
        total_pnl = df['pnl'].sum()
        df['contribution_pct'] = (df['pnl'] / total_pnl * 100) if total_pnl != 0 else 0

        # What % of total P&L comes from top X% of trades?
        df_sorted = df.sort_values('pnl', ascending=False)
        df_sorted['cumulative_contribution'] = df_sorted['contribution_pct'].cumsum()

        top_10pct_trades = int(len(df) * 0.1)
        top_20pct_trades = int(len(df) * 0.2)

        concentration_top10 = df_sorted.head(top_10pct_trades)['contribution_pct'].sum()
        concentration_top20 = df_sorted.head(top_20pct_trades)['contribution_pct'].sum()

        return {
            # Overall stats
            'total_trades': len(df),
            'winning_trades': df['is_winner'].sum(),
            'losing_trades': df['is_loser'].sum(),
            'total_pnl': total_pnl,
            'win_rate': (df['is_winner'].sum() / len(df) * 100),

            # Attribution tables
            'by_ticker': ticker_attribution,
            'top_winners': top_winners,
            'top_losers': top_losers,
            'by_exit_reason': exit_reason_attribution,
            'by_month': monthly_attribution,
            'by_weekday': weekday_attribution,
            'by_hold_duration': hold_attribution,

            # Outliers
            'outliers_positive': outliers_positive,
            'outliers_negative': outliers_negative,
            'pnl_std': pnl_std,
            'pnl_mean': pnl_mean,

            # Concentration
            'concentration_top10pct': concentration_top10,
            'concentration_top20pct': concentration_top20,
            'top10pct_trade_count': top_10pct_trades,
            'top20pct_trade_count': top_20pct_trades,

            # Full data for further analysis
            'trade_data': df,
        }

    def _empty_cost_attribution(self) -> Dict:
        """Return empty cost attribution."""
        return {
            'total_costs': 0,
            'total_slippage': 0,
            'total_fees': 0,
            'total_commission': 0,
            'gross_pnl': 0,
            'net_pnl': 0,
            'costs_as_pct_gross': 0,
            'ticker_analysis': {},
            'trade_costs': pd.DataFrame(),
        }

    def _empty_winlose_attribution(self) -> Dict:
        """Return empty win/lose attribution."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'by_ticker': pd.DataFrame(),
            'by_exit_reason': pd.DataFrame(),
            'trade_data': pd.DataFrame(),
        }
