"""
Visualization components for backtesting dashboard.

Provides:
- Equity curve with annotations
- Drawdown chart
- Returns distribution and histograms
- Performance heatmaps
- Trade analysis plots
- Benchmark comparisons
- Signal visualizations
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
from datetime import datetime


class DashboardVisualizations:
    """Create all dashboard visualizations using Plotly."""

    def __init__(self, theme: str = 'plotly_dark'):
        """
        Initialize visualizations.

        Args:
            theme: Plotly theme ('plotly', 'plotly_dark', 'plotly_white')
        """
        self.theme = theme
        self.colors = {
            'profit': '#00ff00',
            'loss': '#ff0000',
            'neutral': '#4a9eff',
            'benchmark': '#ffa500',
            'grid': '#333333',
        }

    def plot_equity_curve(
        self,
        equity_curve: pd.Series,
        trades: List[Dict],
        benchmark: Optional[pd.Series] = None,
        title: str = "Portfolio Equity Curve"
    ) -> go.Figure:
        """
        Plot equity curve with entry/exit annotations.

        Args:
            equity_curve: Series with datetime index and equity values
            trades: List of trade dicts with entry/exit info
            benchmark: Optional benchmark equity curve
            title: Chart title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Main equity curve
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Strategy',
            line=dict(color=self.colors['neutral'], width=2),
            hovertemplate='<b>Date:</b> %{x}<br>' +
                          '<b>Equity:</b> $%{y:,.2f}<br>' +
                          '<extra></extra>'
        ))

        # Add benchmark if provided
        if benchmark is not None and len(benchmark) > 0:
            # Normalize benchmark to same starting value
            benchmark_normalized = benchmark * (equity_curve.iloc[0] / benchmark.iloc[0])

            fig.add_trace(go.Scatter(
                x=benchmark_normalized.index,
                y=benchmark_normalized.values,
                mode='lines',
                name='S&P 500',
                line=dict(color=self.colors['benchmark'], width=1, dash='dash'),
                hovertemplate='<b>Date:</b> %{x}<br>' +
                              '<b>Equity:</b> $%{y:,.2f}<br>' +
                              '<extra></extra>'
            ))

        # Add entry/exit markers
        entry_dates = []
        entry_values = []
        entry_text = []

        exit_dates = []
        exit_values = []
        exit_text = []

        for trade in trades:
            if trade.get('action') == 'ENTRY':
                date = pd.to_datetime(trade['date'])
                if date in equity_curve.index:
                    entry_dates.append(date)
                    entry_values.append(equity_curve[date])
                    entry_text.append(
                        f"{trade.get('ticker', 'N/A')}<br>"
                        f"Shares: {trade.get('shares', 0):.0f}<br>"
                        f"Price: ${trade.get('price', 0):.2f}"
                    )

            elif trade.get('action') == 'EXIT':
                date = pd.to_datetime(trade['date'])
                if date in equity_curve.index:
                    exit_dates.append(date)
                    exit_values.append(equity_curve[date])
                    pnl = trade.get('pnl', 0)
                    color = 'green' if pnl > 0 else 'red'
                    exit_text.append(
                        f"{trade.get('ticker', 'N/A')}<br>"
                        f"P&L: ${pnl:,.2f}<br>"
                        f"Reason: {trade.get('exit_reason', 'N/A')}"
                    )

        # Entry markers (green triangles)
        if entry_dates:
            fig.add_trace(go.Scatter(
                x=entry_dates,
                y=entry_values,
                mode='markers',
                name='Entries',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color=self.colors['profit'],
                    line=dict(width=1, color='white')
                ),
                text=entry_text,
                hovertemplate='<b>ENTRY</b><br>%{text}<extra></extra>'
            ))

        # Exit markers (red triangles)
        if exit_dates:
            fig.add_trace(go.Scatter(
                x=exit_dates,
                y=exit_values,
                mode='markers',
                name='Exits',
                marker=dict(
                    symbol='triangle-down',
                    size=10,
                    color=self.colors['loss'],
                    line=dict(width=1, color='white')
                ),
                text=exit_text,
                hovertemplate='<b>EXIT</b><br>%{text}<extra></extra>'
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            template=self.theme,
            hovermode='x unified',
            height=500,
        )

        return fig

    def plot_drawdown(
        self,
        equity_curve: pd.Series,
        title: str = "Drawdown"
    ) -> go.Figure:
        """
        Plot underwater drawdown chart.

        Args:
            equity_curve: Series with datetime index and equity values
            title: Chart title

        Returns:
            Plotly figure
        """
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max * 100

        fig = go.Figure()

        # Fill drawdown area
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill='tozeroy',
            name='Drawdown',
            line=dict(color=self.colors['loss'], width=1),
            fillcolor='rgba(255, 0, 0, 0.3)',
            hovertemplate='<b>Date:</b> %{x}<br>' +
                          '<b>Drawdown:</b> %{y:.2f}%<br>' +
                          '<extra></extra>'
        ))

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template=self.theme,
            hovermode='x unified',
            height=400,
        )

        return fig

    def plot_pnl_metrics(
        self,
        metrics: Dict,
        title: str = "Risk-Adjusted Performance"
    ) -> go.Figure:
        """
        Plot Sharpe, Sortino, Omega ratios as bar chart.

        Args:
            metrics: Dictionary with metric values
            title: Chart title

        Returns:
            Plotly figure
        """
        metric_names = ['Sharpe Ratio', 'Sortino Ratio', 'Omega Ratio', 'Calmar Ratio']
        metric_keys = ['sharpe_ratio', 'sortino_ratio', 'omega_ratio', 'calmar_ratio']

        values = [metrics.get(key, 0) for key in metric_keys]

        # Color code based on quality
        colors = []
        for val in values:
            if val > 2:
                colors.append('#00ff00')  # Excellent
            elif val > 1:
                colors.append('#90ee90')  # Good
            elif val > 0:
                colors.append('#ffa500')  # Okay
            else:
                colors.append('#ff0000')  # Poor

        fig = go.Figure(data=[
            go.Bar(
                x=metric_names,
                y=values,
                marker_color=colors,
                text=[f'{v:.2f}' for v in values],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' +
                              'Value: %{y:.3f}<br>' +
                              '<extra></extra>'
            )
        ])

        # Add reference lines
        fig.add_hline(y=1, line_dash="dash", line_color="white",
                      opacity=0.5, annotation_text="Good (1.0)")
        fig.add_hline(y=2, line_dash="dash", line_color="green",
                      opacity=0.5, annotation_text="Excellent (2.0)")

        fig.update_layout(
            title=title,
            yaxis_title="Ratio Value",
            template=self.theme,
            height=400,
            showlegend=False,
        )

        return fig

    def plot_returns_histogram(
        self,
        returns: pd.Series,
        bins: int = 50,
        title: str = "Returns Distribution"
    ) -> go.Figure:
        """
        Plot histogram of returns with normal curve overlay.

        Args:
            returns: Series of returns
            bins: Number of histogram bins
            title: Chart title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Histogram
        fig.add_trace(go.Histogram(
            x=returns * 100,  # Convert to percentage
            nbinsx=bins,
            name='Returns',
            marker_color=self.colors['neutral'],
            opacity=0.7,
            hovertemplate='<b>Return Range:</b> %{x:.2f}%<br>' +
                          '<b>Frequency:</b> %{y}<br>' +
                          '<extra></extra>'
        ))

        # Add normal distribution overlay
        mu = returns.mean() * 100
        sigma = returns.std() * 100

        x_range = np.linspace(returns.min() * 100, returns.max() * 100, 100)
        normal_dist = (1 / (sigma * np.sqrt(2 * np.pi))) * \
                      np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)

        # Scale to match histogram
        normal_dist = normal_dist * len(returns) * (returns.max() - returns.min()) * 100 / bins

        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_dist,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', width=2),
            hovertemplate='<b>Normal Distribution</b><br>' +
                          'Return: %{x:.2f}%<br>' +
                          '<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            template=self.theme,
            height=400,
            bargap=0.1,
        )

        return fig

    def plot_win_rate_analysis(
        self,
        trades: List[Dict],
        title: str = "Win Rate Analysis"
    ) -> go.Figure:
        """
        Plot win/loss analysis by trade size buckets.

        Args:
            trades: List of trade dicts
            title: Chart title

        Returns:
            Plotly figure
        """
        closed_trades = [t for t in trades if t.get('action') == 'EXIT']

        if not closed_trades:
            return self._empty_chart("No closed trades yet")

        # Group by P&L buckets
        pnls = [t.get('pnl', 0) for t in closed_trades]

        # Create buckets
        min_pnl = min(pnls)
        max_pnl = max(pnls)
        bucket_size = (max_pnl - min_pnl) / 10

        buckets = {}
        for pnl in pnls:
            bucket_idx = int((pnl - min_pnl) / bucket_size) if bucket_size > 0 else 0
            bucket_idx = min(bucket_idx, 9)  # Cap at 9
            buckets[bucket_idx] = buckets.get(bucket_idx, 0) + 1

        # Create bar chart
        x_labels = [f"${min_pnl + i * bucket_size:.0f}" for i in range(10)]
        y_values = [buckets.get(i, 0) for i in range(10)]

        colors_buckets = [self.colors['profit'] if (min_pnl + i * bucket_size) > 0
                         else self.colors['loss'] for i in range(10)]

        fig = go.Figure(data=[
            go.Bar(
                x=x_labels,
                y=y_values,
                marker_color=colors_buckets,
                hovertemplate='<b>P&L Range:</b> %{x}<br>' +
                              '<b>Count:</b> %{y}<br>' +
                              '<extra></extra>'
            )
        ])

        fig.update_layout(
            title=title,
            xaxis_title="P&L Range",
            yaxis_title="Number of Trades",
            template=self.theme,
            height=400,
            showlegend=False,
        )

        return fig

    def plot_trade_heatmap(
        self,
        trades: List[Dict],
        title: str = "Trade Performance Heatmap"
    ) -> go.Figure:
        """
        Plot heatmap of trade performance by day/month.

        Args:
            trades: List of trade dicts
            title: Chart title

        Returns:
            Plotly figure
        """
        closed_trades = [t for t in trades if t.get('action') == 'EXIT']

        if not closed_trades:
            return self._empty_chart("No closed trades yet")

        # Create DataFrame
        df = pd.DataFrame([
            {
                'date': pd.to_datetime(t['date']),
                'pnl': t.get('pnl', 0)
            }
            for t in closed_trades
        ])

        df['year_month'] = df['date'].dt.to_period('M').astype(str)
        df['day'] = df['date'].dt.day

        # Pivot table
        pivot = df.pivot_table(
            values='pnl',
            index='day',
            columns='year_month',
            aggfunc='sum',
            fill_value=0
        )

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn',
            zmid=0,
            hovertemplate='<b>Month:</b> %{x}<br>' +
                          '<b>Day:</b> %{y}<br>' +
                          '<b>P&L:</b> $%{z:,.2f}<br>' +
                          '<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Month",
            yaxis_title="Day of Month",
            template=self.theme,
            height=500,
        )

        return fig

    def plot_outlier_trades(
        self,
        trades: List[Dict],
        n_outliers: int = 10,
        title: str = "Top Outlier Trades"
    ) -> go.Figure:
        """
        Identify and plot outlier trades (best and worst).

        Args:
            trades: List of trade dicts
            n_outliers: Number of outliers to show
            title: Chart title

        Returns:
            Plotly figure
        """
        closed_trades = [t for t in trades if t.get('action') == 'EXIT']

        if not closed_trades:
            return self._empty_chart("No closed trades yet")

        # Sort by P&L
        sorted_trades = sorted(closed_trades, key=lambda x: x.get('pnl', 0))

        # Get worst and best
        worst = sorted_trades[:n_outliers]
        best = sorted_trades[-n_outliers:][::-1]

        # Combine
        outliers = worst + best

        tickers = [t.get('ticker', 'N/A') for t in outliers]
        pnls = [t.get('pnl', 0) for t in outliers]
        dates = [t.get('date', 'N/A') for t in outliers]

        colors_outliers = [self.colors['loss'] if p < 0 else self.colors['profit'] for p in pnls]

        fig = go.Figure(data=[
            go.Bar(
                x=pnls,
                y=[f"{ticker} ({date})" for ticker, date in zip(tickers, dates)],
                orientation='h',
                marker_color=colors_outliers,
                hovertemplate='<b>%{y}</b><br>' +
                              'P&L: $%{x:,.2f}<br>' +
                              '<extra></extra>'
            )
        ])

        fig.update_layout(
            title=title,
            xaxis_title="P&L ($)",
            yaxis_title="Trade",
            template=self.theme,
            height=600,
            showlegend=False,
        )

        return fig

    def plot_rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 30,
        title: str = "Rolling Performance Metrics"
    ) -> go.Figure:
        """
        Plot rolling Sharpe and volatility.

        Args:
            returns: Series of returns
            window: Rolling window size
            title: Chart title

        Returns:
            Plotly figure
        """
        rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
        rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Rolling Sharpe Ratio', 'Rolling Volatility'),
            vertical_spacing=0.15
        )

        # Sharpe
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode='lines',
                name='Sharpe',
                line=dict(color=self.colors['neutral'], width=2),
                hovertemplate='<b>Date:</b> %{x}<br>' +
                              '<b>Sharpe:</b> %{y:.2f}<br>' +
                              '<extra></extra>'
            ),
            row=1, col=1
        )

        # Volatility
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode='lines',
                name='Volatility',
                line=dict(color=self.colors['loss'], width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.2)',
                hovertemplate='<b>Date:</b> %{x}<br>' +
                              '<b>Volatility:</b> %{y:.2f}%<br>' +
                              '<extra></extra>'
            ),
            row=2, col=1
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)

        fig.update_layout(
            title=title,
            template=self.theme,
            height=600,
            showlegend=False,
            hovermode='x unified'
        )

        return fig

    def plot_leverage_utilization(
        self,
        equity_curve: pd.Series,
        position_values: pd.Series,
        title: str = "Leverage & Capital Utilization"
    ) -> go.Figure:
        """
        Plot leverage ratio and capital utilization over time.

        Args:
            equity_curve: Series with equity values
            position_values: Series with total position values
            title: Chart title

        Returns:
            Plotly figure
        """
        leverage = position_values / equity_curve
        utilization = (position_values / equity_curve * 100).clip(0, 200)  # Cap at 200%

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Leverage Ratio', 'Capital Utilization'),
            vertical_spacing=0.15
        )

        # Leverage
        fig.add_trace(
            go.Scatter(
                x=leverage.index,
                y=leverage.values,
                mode='lines',
                name='Leverage',
                line=dict(color=self.colors['neutral'], width=2),
                fill='tozeroy',
                fillcolor='rgba(74, 158, 255, 0.2)',
                hovertemplate='<b>Date:</b> %{x}<br>' +
                              '<b>Leverage:</b> %{y:.2f}x<br>' +
                              '<extra></extra>'
            ),
            row=1, col=1
        )

        # Utilization
        fig.add_trace(
            go.Scatter(
                x=utilization.index,
                y=utilization.values,
                mode='lines',
                name='Utilization',
                line=dict(color=self.colors['profit'], width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.2)',
                hovertemplate='<b>Date:</b> %{x}<br>' +
                              '<b>Utilization:</b> %{y:.1f}%<br>' +
                              '<extra></extra>'
            ),
            row=2, col=1
        )

        # Add 100% reference line
        fig.add_hline(y=100, line_dash="dash", line_color="white",
                      opacity=0.5, row=2, col=1)

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Leverage (x)", row=1, col=1)
        fig.update_yaxes(title_text="Utilization (%)", row=2, col=1)

        fig.update_layout(
            title=title,
            template=self.theme,
            height=600,
            showlegend=False,
            hovermode='x unified'
        )

        return fig

    def _empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message."""
        fig = go.Figure()

        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray")
        )

        fig.update_layout(
            template=self.theme,
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )

        return fig
