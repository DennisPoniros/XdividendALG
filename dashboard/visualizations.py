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

    def plot_monte_carlo_distribution(
        self,
        simulation_results: Dict,
        metric: str = 'equity',
        title: str = "Monte Carlo Simulation Results"
    ) -> go.Figure:
        """
        Plot Monte Carlo simulation distribution.

        Args:
            simulation_results: Results from Monte Carlo simulation
            metric: 'equity', 'drawdown', or 'sharpe'
            title: Chart title

        Returns:
            Plotly figure
        """
        if metric == 'equity':
            data = simulation_results.get('final_equities', [])
            actual_value = simulation_results.get('actual_equity', 0)
            xlabel = "Final Equity ($)"
        elif metric == 'drawdown':
            data = simulation_results.get('max_drawdowns', []) * 100  # Convert to %
            actual_value = simulation_results.get('actual_max_dd', 0) * 100
            xlabel = "Max Drawdown (%)"
        elif metric == 'sharpe':
            data = simulation_results.get('sharpe_ratios', [])
            actual_value = simulation_results.get('actual_sharpe', 0)
            xlabel = "Sharpe Ratio"
        else:
            return self._empty_chart(f"Unknown metric: {metric}")

        if len(data) == 0:
            return self._empty_chart("No simulation data available")

        fig = go.Figure()

        # Histogram
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=50,
            name='Simulated',
            marker_color=self.colors['neutral'],
            opacity=0.7,
            hovertemplate='<b>Range:</b> %{x}<br>' +
                          '<b>Frequency:</b> %{y}<br>' +
                          '<extra></extra>'
        ))

        # Add actual value line
        fig.add_vline(
            x=actual_value,
            line_dash="dash",
            line_color=self.colors['profit'],
            line_width=3,
            annotation_text=f"Actual: {actual_value:,.2f}",
            annotation_position="top"
        )

        # Add percentile lines
        p5 = np.percentile(data, 5)
        p95 = np.percentile(data, 95)

        fig.add_vline(
            x=p5,
            line_dash="dot",
            line_color="orange",
            opacity=0.5,
            annotation_text="5th %ile",
            annotation_position="bottom left"
        )

        fig.add_vline(
            x=p95,
            line_dash="dot",
            line_color="orange",
            opacity=0.5,
            annotation_text="95th %ile",
            annotation_position="bottom right"
        )

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title="Frequency",
            template=self.theme,
            height=500,
            showlegend=False,
        )

        return fig

    def plot_monte_carlo_percentiles(
        self,
        simulation_results: Dict,
        title: str = "Monte Carlo Percentile Analysis"
    ) -> go.Figure:
        """
        Plot percentile comparison for multiple metrics.

        Args:
            simulation_results: Results from Monte Carlo simulation
            title: Chart title

        Returns:
            Plotly figure
        """
        metrics = ['Equity', 'Sharpe', 'Drawdown']
        percentiles = ['5th', '25th', '50th', '75th', '95th']

        # Get data for each metric
        equity_pcts = simulation_results.get('equity_percentiles', {})
        sharpe_pcts = simulation_results.get('sharpe_percentiles', {})
        dd_pcts = simulation_results.get('drawdown_percentiles', {})

        # Normalize for comparison (0-100 scale)
        def normalize(values, reverse=False):
            min_val = min(values)
            max_val = max(values)
            if max_val == min_val:
                return [50] * len(values)
            normalized = [(v - min_val) / (max_val - min_val) * 100 for v in values]
            if reverse:
                normalized = [100 - v for v in normalized]
            return normalized

        equity_vals = [equity_pcts.get(p, 0) for p in percentiles]
        sharpe_vals = [sharpe_pcts.get(p, 0) for p in percentiles]
        dd_vals = [abs(dd_pcts.get(p, 0)) for p in percentiles]

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Final Equity', 'Sharpe Ratio', 'Max Drawdown'),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )

        # Equity
        fig.add_trace(
            go.Bar(
                x=percentiles,
                y=equity_vals,
                name='Equity',
                marker_color=self.colors['profit'],
                text=[f'${v:,.0f}' for v in equity_vals],
                textposition='auto',
            ),
            row=1, col=1
        )

        # Sharpe
        fig.add_trace(
            go.Bar(
                x=percentiles,
                y=sharpe_vals,
                name='Sharpe',
                marker_color=self.colors['neutral'],
                text=[f'{v:.2f}' for v in sharpe_vals],
                textposition='auto',
            ),
            row=1, col=2
        )

        # Drawdown
        fig.add_trace(
            go.Bar(
                x=percentiles,
                y=dd_vals,
                name='Drawdown',
                marker_color=self.colors['loss'],
                text=[f'{v:.1f}%' for v in dd_vals],
                textposition='auto',
            ),
            row=1, col=3
        )

        fig.update_layout(
            title=title,
            template=self.theme,
            height=400,
            showlegend=False,
        )

        return fig

    def plot_cost_attribution(
        self,
        cost_attribution: Dict,
        title: str = "Cost Attribution Analysis"
    ) -> go.Figure:
        """
        Plot cost breakdown pie chart and waterfall.

        Args:
            cost_attribution: Results from cost attribution analysis
            title: Chart title

        Returns:
            Plotly figure
        """
        total_slippage = cost_attribution.get('total_slippage', 0)
        total_fees = cost_attribution.get('total_fees', 0)
        total_commission = cost_attribution.get('total_commission', 0)

        if total_slippage + total_fees + total_commission == 0:
            return self._empty_chart("No cost data available")

        # Pie chart of cost breakdown
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Cost Components', 'Impact on Returns'),
            specs=[[{"type": "pie"}, {"type": "bar"}]]
        )

        # Cost components pie
        fig.add_trace(
            go.Pie(
                labels=['Slippage', 'Fees', 'Commission'],
                values=[total_slippage, total_fees, total_commission],
                marker_colors=[self.colors['loss'], self.colors['benchmark'], '#ff69b4'],
                hovertemplate='<b>%{label}</b><br>' +
                              'Amount: $%{value:,.2f}<br>' +
                              'Percentage: %{percent}<br>' +
                              '<extra></extra>'
            ),
            row=1, col=1
        )

        # Impact waterfall
        gross_pnl = cost_attribution.get('gross_pnl', 0)
        net_pnl = cost_attribution.get('net_pnl', 0)
        total_costs = cost_attribution.get('total_costs', 0)

        fig.add_trace(
            go.Waterfall(
                x=['Gross P&L', 'Slippage', 'Fees', 'Commission', 'Net P&L'],
                y=[gross_pnl, -total_slippage, -total_fees, -total_commission, net_pnl],
                measure=['absolute', 'relative', 'relative', 'relative', 'total'],
                text=[f'${gross_pnl:,.0f}', f'-${total_slippage:,.0f}',
                      f'-${total_fees:,.0f}', f'-${total_commission:,.0f}', f'${net_pnl:,.0f}'],
                textposition='auto',
                connector={"line": {"color": "gray"}},
            ),
            row=1, col=2
        )

        fig.update_layout(
            title=title,
            template=self.theme,
            height=500,
            showlegend=False,
        )

        return fig

    def plot_cost_by_ticker(
        self,
        cost_attribution: Dict,
        top_n: int = 10,
        title: str = "Cost Attribution by Ticker"
    ) -> go.Figure:
        """
        Plot costs by ticker.

        Args:
            cost_attribution: Results from cost attribution analysis
            top_n: Number of tickers to show
            title: Chart title

        Returns:
            Plotly figure
        """
        ticker_analysis = cost_attribution.get('ticker_analysis', {})

        if not ticker_analysis:
            return self._empty_chart("No ticker cost data available")

        # Extract top tickers by cost
        tickers = list(ticker_analysis.keys())[:top_n]
        costs = [ticker_analysis[t]['total_costs'] for t in tickers]
        cost_pcts = [ticker_analysis[t]['cost_as_pct_gross'] for t in tickers]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Total Costs', 'Costs as % of Gross P&L'),
        )

        # Total costs
        fig.add_trace(
            go.Bar(
                x=tickers,
                y=costs,
                name='Total Costs',
                marker_color=self.colors['loss'],
                text=[f'${c:,.0f}' for c in costs],
                textposition='auto',
            ),
            row=1, col=1
        )

        # Cost percentage
        fig.add_trace(
            go.Bar(
                x=tickers,
                y=cost_pcts,
                name='Cost %',
                marker_color=self.colors['benchmark'],
                text=[f'{p:.1f}%' for p in cost_pcts],
                textposition='auto',
            ),
            row=1, col=2
        )

        fig.update_xaxes(title_text="Ticker", row=1, col=1)
        fig.update_xaxes(title_text="Ticker", row=1, col=2)
        fig.update_yaxes(title_text="Cost ($)", row=1, col=1)
        fig.update_yaxes(title_text="Cost (%)", row=1, col=2)

        fig.update_layout(
            title=title,
            template=self.theme,
            height=500,
            showlegend=False,
        )

        return fig

    def plot_winlose_attribution(
        self,
        winlose_attribution: Dict,
        title: str = "Win/Loss Attribution"
    ) -> go.Figure:
        """
        Plot win/loss attribution by various factors.

        Args:
            winlose_attribution: Results from win/loss attribution analysis
            title: Chart title

        Returns:
            Plotly figure
        """
        by_ticker = winlose_attribution.get('by_ticker', pd.DataFrame())

        if by_ticker.empty:
            return self._empty_chart("No win/loss attribution data available")

        # Top 10 tickers by total P&L
        top_10 = by_ticker.head(10)

        tickers = top_10.index.tolist()
        total_pnls = top_10['total_pnl'].values
        win_rates = top_10['win_rate'].values

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Total P&L by Ticker', 'Win Rate by Ticker'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )

        # P&L bars
        colors_pnl = [self.colors['profit'] if p > 0 else self.colors['loss'] for p in total_pnls]

        fig.add_trace(
            go.Bar(
                x=tickers,
                y=total_pnls,
                marker_color=colors_pnl,
                text=[f'${p:,.0f}' for p in total_pnls],
                textposition='auto',
                name='P&L',
            ),
            row=1, col=1
        )

        # Win rate bars
        colors_wr = [self.colors['profit'] if w > 50 else self.colors['loss'] for w in win_rates]

        fig.add_trace(
            go.Bar(
                x=tickers,
                y=win_rates,
                marker_color=colors_wr,
                text=[f'{w:.1f}%' for w in win_rates],
                textposition='auto',
                name='Win Rate',
            ),
            row=1, col=2
        )

        # Add 50% reference line to win rate
        fig.add_hline(y=50, line_dash="dash", line_color="white", opacity=0.5, row=1, col=2)

        fig.update_xaxes(title_text="Ticker", row=1, col=1)
        fig.update_xaxes(title_text="Ticker", row=1, col=2)
        fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
        fig.update_yaxes(title_text="Win Rate (%)", row=1, col=2)

        fig.update_layout(
            title=title,
            template=self.theme,
            height=500,
            showlegend=False,
        )

        return fig

    def plot_attribution_by_exit_reason(
        self,
        winlose_attribution: Dict,
        title: str = "Attribution by Exit Reason"
    ) -> go.Figure:
        """
        Plot performance attribution by exit reason.

        Args:
            winlose_attribution: Results from win/loss attribution analysis
            title: Chart title

        Returns:
            Plotly figure
        """
        by_exit_reason = winlose_attribution.get('by_exit_reason', pd.DataFrame())

        if by_exit_reason.empty:
            return self._empty_chart("No exit reason data available")

        exit_reasons = by_exit_reason.index.tolist()
        total_pnls = by_exit_reason['total_pnl'].values
        trade_counts = by_exit_reason['trade_count'].values
        win_rates = by_exit_reason['win_rate'].values

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('P&L by Exit Reason', 'Win Rate by Exit Reason'),
            vertical_spacing=0.15
        )

        # P&L by exit reason
        colors_pnl = [self.colors['profit'] if p > 0 else self.colors['loss'] for p in total_pnls]

        fig.add_trace(
            go.Bar(
                x=exit_reasons,
                y=total_pnls,
                marker_color=colors_pnl,
                text=[f'${p:,.0f}<br>({int(c)} trades)' for p, c in zip(total_pnls, trade_counts)],
                textposition='auto',
                name='P&L',
            ),
            row=1, col=1
        )

        # Win rate
        colors_wr = [self.colors['profit'] if w > 50 else self.colors['loss'] for w in win_rates]

        fig.add_trace(
            go.Bar(
                x=exit_reasons,
                y=win_rates,
                marker_color=colors_wr,
                text=[f'{w:.1f}%' for w in win_rates],
                textposition='auto',
                name='Win Rate',
            ),
            row=2, col=1
        )

        # Add 50% reference line
        fig.add_hline(y=50, line_dash="dash", line_color="white", opacity=0.5, row=2, col=1)

        fig.update_xaxes(title_text="Exit Reason", row=2, col=1)
        fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
        fig.update_yaxes(title_text="Win Rate (%)", row=2, col=1)

        fig.update_layout(
            title=title,
            template=self.theme,
            height=700,
            showlegend=False,
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
