"""
Interactive Backtesting Dashboard

Real-time monitoring and analysis of algorithmic trading strategies.

Features:
- Live backtest monitoring
- Comprehensive performance metrics
- Interactive visualizations
- Configuration interface
- Multi-strategy support
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dashboard.data_interface import BacktestDataInterface
from dashboard.metrics import MetricsCalculator
from dashboard.visualizations import DashboardVisualizations
from dashboard.monte_carlo import MonteCarloSimulator
from dashboard.attribution import AttributionAnalyzer


# Page configuration
st.set_page_config(
    page_title="Algorithmic Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .positive {
        color: #00ff00;
    }
    .negative {
        color: #ff0000;
    }
    .neutral {
        color: #4a9eff;
    }
    .stMetric {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


class BacktestDashboard:
    """Main dashboard application."""

    def __init__(self):
        """Initialize dashboard."""
        self.data_interface = None
        self.metrics_calculator = MetricsCalculator()
        self.visualizations = DashboardVisualizations(theme='plotly_dark')
        self.monte_carlo = MonteCarloSimulator(n_simulations=1000)
        self.attribution_analyzer = AttributionAnalyzer()

        # Initialize session state
        if 'mode' not in st.session_state:
            st.session_state.mode = 'replay'

        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False

        if 'selected_backtest' not in st.session_state:
            st.session_state.selected_backtest = None

    def run(self):
        """Run the dashboard application."""
        st.title("📈 Algorithmic Trading Dashboard")

        # Sidebar
        self.render_sidebar()

        # Initialize data interface
        self.data_interface = BacktestDataInterface(mode=st.session_state.mode)

        # Load data
        equity_curve, trades, position_values = self.load_data()

        if len(equity_curve) == 0:
            st.warning("⚠️ No backtest data available. Please run a backtest first.")
            self.render_quick_start_guide()
            return

        # Calculate metrics
        returns = equity_curve.pct_change().dropna()
        benchmark_returns = self.load_benchmark(equity_curve.index[0], equity_curve.index[-1])

        metrics = self.metrics_calculator.calculate_all_metrics(
            equity_curve=equity_curve,
            trades=trades,
            benchmark_returns=benchmark_returns
        )

        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Overview",
            "📈 Performance",
            "💼 Trades",
            "🎯 Risk Analysis",
            "🎲 Monte Carlo",
            "📉 Attribution",
            "⚙️ Configuration"
        ])

        with tab1:
            self.render_overview_tab(equity_curve, trades, metrics, benchmark_returns)

        with tab2:
            self.render_performance_tab(equity_curve, returns, trades, metrics)

        with tab3:
            self.render_trades_tab(trades, equity_curve)

        with tab4:
            self.render_risk_tab(equity_curve, returns, metrics, position_values)

        with tab5:
            self.render_monte_carlo_tab(trades)

        with tab6:
            self.render_attribution_tab(trades)

        with tab7:
            self.render_configuration_tab()

    def render_sidebar(self):
        """Render sidebar with controls."""
        st.sidebar.title("⚙️ Dashboard Controls")

        # Mode selection
        st.sidebar.subheader("Mode")
        mode = st.sidebar.radio(
            "Select Mode",
            options=['replay', 'live'],
            format_func=lambda x: "📂 Replay Mode" if x == 'replay' else "🔴 Live Mode",
            key='mode'
        )

        if mode == 'replay':
            # Backtest selection
            st.sidebar.subheader("Select Backtest")

            data_interface_temp = BacktestDataInterface(mode='replay')
            available_backtests = data_interface_temp.list_available_backtests()

            if available_backtests:
                backtest_names = [b['name'] for b in available_backtests]
                selected = st.sidebar.selectbox(
                    "Available Backtests",
                    options=backtest_names,
                    index=0
                )
                st.session_state.selected_backtest = selected

                # Show backtest info
                selected_info = next(b for b in available_backtests if b['name'] == selected)
                st.sidebar.info(
                    f"**Modified:** {selected_info['modified'].strftime('%Y-%m-%d %H:%M')}\n\n"
                    f"**Size:** {selected_info['size'] / 1024:.1f} KB"
                )
            else:
                st.sidebar.warning("No backtests found")

        else:
            st.sidebar.info("Live mode: Dashboard will update in real-time during backtest execution.")

        # Refresh controls
        st.sidebar.subheader("Refresh Settings")
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh

        if auto_refresh:
            refresh_interval = st.sidebar.slider(
                "Refresh Interval (seconds)",
                min_value=1,
                max_value=60,
                value=5
            )
            st.sidebar.info(f"Auto-refreshing every {refresh_interval} seconds")

        if st.sidebar.button("🔄 Refresh Now"):
            st.rerun()

        # Info
        st.sidebar.markdown("---")
        st.sidebar.subheader("ℹ️ About")
        st.sidebar.info(
            "**Algorithmic Trading Dashboard**\n\n"
            "Real-time monitoring and analysis of trading strategies.\n\n"
            "Version: 0.1.0"
        )

    def load_data(self):
        """Load backtest data."""
        if st.session_state.mode == 'replay':
            # Load specific backtest if selected
            results = self.data_interface.load_backtest_results(
                st.session_state.selected_backtest
            )
            equity_curve = results.get('equity_curve', pd.Series())
            trades = results.get('trades', [])
            position_values = results.get('position_values', pd.Series())
        else:
            # Live mode - get current data
            equity_curve = self.data_interface.get_equity_curve()
            trades = self.data_interface.get_trades()
            position_values = self.data_interface.get_position_values()

        return equity_curve, trades, position_values

    def load_benchmark(self, start_date, end_date):
        """Load benchmark data."""
        try:
            benchmark_returns = self.metrics_calculator.get_benchmark_data(
                start_date=start_date,
                end_date=end_date,
                ticker='SPY'
            )
            return benchmark_returns
        except Exception as e:
            st.warning(f"Could not load benchmark data: {e}")
            return None

    def render_overview_tab(self, equity_curve, trades, metrics, benchmark_returns):
        """Render overview tab with key metrics and charts."""
        st.header("Overview")

        # Top metrics row
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            total_return = metrics.get('total_return_pct', 0)
            color = 'positive' if total_return > 0 else 'negative'
            st.metric(
                "Total Return",
                f"{total_return:.2f}%",
                delta=f"{total_return:.2f}%"
            )

        with col2:
            sharpe = metrics.get('sharpe_ratio', 0)
            color = 'positive' if sharpe > 1 else 'negative'
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.2f}",
                delta="Good" if sharpe > 1 else "Poor"
            )

        with col3:
            max_dd = metrics.get('max_drawdown_pct', 0)
            st.metric(
                "Max Drawdown",
                f"{max_dd:.2f}%",
                delta=f"{max_dd:.2f}%",
                delta_color="inverse"
            )

        with col4:
            win_rate = metrics.get('win_rate_pct', 0)
            st.metric(
                "Win Rate",
                f"{win_rate:.1f}%",
                delta="Good" if win_rate > 50 else "Poor"
            )

        with col5:
            total_trades = metrics.get('total_trades', 0)
            st.metric(
                "Total Trades",
                f"{total_trades}",
                delta=None
            )

        # Equity curve with benchmark
        st.subheader("Equity Curve")

        if benchmark_returns is not None and len(benchmark_returns) > 0:
            # Create benchmark equity curve
            benchmark_equity = (1 + benchmark_returns).cumprod() * equity_curve.iloc[0]
            fig_equity = self.visualizations.plot_equity_curve(
                equity_curve=equity_curve,
                trades=trades,
                benchmark=benchmark_equity
            )
        else:
            fig_equity = self.visualizations.plot_equity_curve(
                equity_curve=equity_curve,
                trades=trades
            )

        st.plotly_chart(fig_equity, use_container_width=True)

        # Second row: Drawdown and risk metrics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Drawdown Analysis")
            fig_dd = self.visualizations.plot_drawdown(equity_curve)
            st.plotly_chart(fig_dd, use_container_width=True)

        with col2:
            st.subheader("Risk-Adjusted Performance")
            fig_metrics = self.visualizations.plot_pnl_metrics(metrics)
            st.plotly_chart(fig_metrics, use_container_width=True)

        # Summary statistics
        st.subheader("Summary Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Return Metrics**")
            st.write(f"**CAGR:** {metrics.get('cagr_pct', 0):.2f}%")
            st.write(f"**Total Return:** {metrics.get('total_return_pct', 0):.2f}%")
            st.write(f"**Daily Avg:** {metrics.get('daily_return_mean', 0)*100:.3f}%")
            st.write(f"**Daily Std:** {metrics.get('daily_return_std', 0)*100:.3f}%")

        with col2:
            st.markdown("**Risk Metrics**")
            st.write(f"**Sharpe:** {metrics.get('sharpe_ratio', 0):.2f}")
            st.write(f"**Sortino:** {metrics.get('sortino_ratio', 0):.2f}")
            st.write(f"**Omega:** {metrics.get('omega_ratio', 0):.2f}")
            st.write(f"**Calmar:** {metrics.get('calmar_ratio', 0):.2f}")

        with col3:
            st.markdown("**Trade Statistics**")
            st.write(f"**Win Rate:** {metrics.get('win_rate_pct', 0):.1f}%")
            st.write(f"**Profit Factor:** {metrics.get('profit_factor', 0):.2f}")
            st.write(f"**Avg Win:** ${metrics.get('avg_win', 0):,.2f}")
            st.write(f"**Avg Loss:** ${metrics.get('avg_loss', 0):,.2f}")

    def render_performance_tab(self, equity_curve, returns, trades, metrics):
        """Render performance analysis tab."""
        st.header("Performance Analysis")

        # Rolling metrics
        st.subheader("Rolling Metrics (30-day)")
        fig_rolling = self.visualizations.plot_rolling_metrics(returns, window=30)
        st.plotly_chart(fig_rolling, use_container_width=True)

        # Returns distribution
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Returns Distribution")
            fig_hist = self.visualizations.plot_returns_histogram(returns)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            st.subheader("Win Rate Analysis")
            fig_win = self.visualizations.plot_win_rate_analysis(trades)
            st.plotly_chart(fig_win, use_container_width=True)

        # Monthly heatmap
        st.subheader("Trade Performance Heatmap")
        fig_heatmap = self.visualizations.plot_trade_heatmap(trades)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Distribution statistics
        st.subheader("Distribution Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Skewness", f"{metrics.get('skewness', 0):.3f}")

        with col2:
            st.metric("Kurtosis", f"{metrics.get('kurtosis', 0):.3f}")

        with col3:
            st.metric("Best Day", f"{metrics.get('best_day', 0)*100:.2f}%")

        with col4:
            st.metric("Worst Day", f"{metrics.get('worst_day', 0)*100:.2f}%")

    def render_trades_tab(self, trades, equity_curve):
        """Render trades analysis tab."""
        st.header("Trade Analysis")

        # Trade summary
        closed_trades = [t for t in trades if t.get('action') == 'EXIT']

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Trades", len(closed_trades))

        with col2:
            winning = len([t for t in closed_trades if t.get('pnl', 0) > 0])
            st.metric("Winning Trades", winning)

        with col3:
            losing = len([t for t in closed_trades if t.get('pnl', 0) < 0])
            st.metric("Losing Trades", losing)

        with col4:
            total_pnl = sum([t.get('pnl', 0) for t in closed_trades])
            st.metric("Total P&L", f"${total_pnl:,.2f}")

        # Outlier trades
        st.subheader("Top Outlier Trades")
        fig_outliers = self.visualizations.plot_outlier_trades(trades, n_outliers=10)
        st.plotly_chart(fig_outliers, use_container_width=True)

        # Trade table
        st.subheader("Trade Log")

        if closed_trades:
            df = pd.DataFrame(closed_trades)

            # Select and format columns
            display_cols = ['date', 'ticker', 'action', 'shares', 'price', 'pnl', 'exit_reason']
            display_cols = [col for col in display_cols if col in df.columns]

            df_display = df[display_cols].copy()

            # Format date
            if 'date' in df_display.columns:
                df_display['date'] = pd.to_datetime(df_display['date']).dt.strftime('%Y-%m-%d')

            # Format numbers
            if 'shares' in df_display.columns:
                df_display['shares'] = df_display['shares'].apply(lambda x: f"{x:.0f}")

            if 'price' in df_display.columns:
                df_display['price'] = df_display['price'].apply(lambda x: f"${x:.2f}")

            if 'pnl' in df_display.columns:
                df_display['pnl'] = df_display['pnl'].apply(lambda x: f"${x:,.2f}")

            # Display with color coding
            st.dataframe(
                df_display,
                use_container_width=True,
                height=400
            )

            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Trade Log",
                data=csv,
                file_name=f"trade_log_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No closed trades yet.")

    def render_risk_tab(self, equity_curve, returns, metrics, position_values):
        """Render risk analysis tab."""
        st.header("Risk Analysis")

        # Risk metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "VaR (95%)",
                f"{metrics.get('var_95', 0)*100:.2f}%",
                help="Value at Risk at 95% confidence"
            )

        with col2:
            st.metric(
                "CVaR (95%)",
                f"{metrics.get('cvar_95', 0)*100:.2f}%",
                help="Conditional Value at Risk (Expected Shortfall)"
            )

        with col3:
            st.metric(
                "Max Drawdown Duration",
                f"{metrics.get('max_drawdown_duration_days', 0):.0f} days"
            )

        with col4:
            st.metric(
                "Underwater Time",
                f"{metrics.get('underwater_pct', 0):.1f}%",
                help="Percentage of time in drawdown"
            )

        # Leverage and utilization
        if len(position_values) > 0:
            st.subheader("Leverage & Capital Utilization")
            fig_leverage = self.visualizations.plot_leverage_utilization(
                equity_curve=equity_curve,
                position_values=position_values
            )
            st.plotly_chart(fig_leverage, use_container_width=True)
        else:
            st.info("No position value data available for leverage analysis.")

        # Drawdown details
        st.subheader("Drawdown Details")

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Max Drawdown:** {metrics.get('max_drawdown_pct', 0):.2f}%")
            st.write(f"**Start Date:** {metrics.get('max_drawdown_start', 'N/A')}")
            st.write(f"**End Date:** {metrics.get('max_drawdown_end', 'N/A')}")

        with col2:
            st.write(f"**Duration:** {metrics.get('max_drawdown_duration_days', 0):.0f} days")
            st.write(f"**Underwater %:** {metrics.get('underwater_pct', 0):.1f}%")

    def render_monte_carlo_tab(self, trades):
        """Render Monte Carlo simulation tab."""
        st.header("🎲 Monte Carlo Simulation")

        st.markdown("""
        Monte Carlo simulation resamples trades to test strategy robustness.
        This helps determine if results are due to skill or luck.
        """)

        if not trades or len([t for t in trades if t.get('action') == 'EXIT']) < 10:
            st.warning("⚠️ Need at least 10 closed trades for meaningful Monte Carlo simulation.")
            return

        # Simulation controls
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("Simulation Parameters")

        with col2:
            if st.button("▶️ Run Simulation", type="primary"):
                st.session_state.run_mc = True

        n_sims = st.slider(
            "Number of Simulations",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="More simulations = more accurate but slower"
        )

        # Run simulation
        if st.session_state.get('run_mc', False):
            with st.spinner("Running Monte Carlo simulation..."):
                self.monte_carlo.n_simulations = n_sims
                simulation_results = self.monte_carlo.run_simulation(trades)

                if 'error' in simulation_results:
                    st.error(simulation_results['error'])
                    return

                # Robustness assessment
                robustness = self.monte_carlo.assess_robustness(simulation_results)

                st.markdown("---")
                st.subheader("Robustness Assessment")

                # Display robustness score
                col1, col2, col3 = st.columns(3)

                with col1:
                    score = robustness['robustness_score']
                    color = '🟢' if score >= 80 else '🟡' if score >= 60 else '🟠' if score >= 40 else '🔴'
                    st.metric(
                        "Robustness Score",
                        f"{score}/100 {color}",
                        delta=robustness['assessment'].split(' - ')[0]
                    )

                with col2:
                    st.metric(
                        "Probability of Profit",
                        f"{simulation_results['prob_profit']:.1%}",
                        delta="Good" if simulation_results['prob_profit'] > 0.75 else "Poor"
                    )

                with col3:
                    st.metric(
                        "P(Sharpe > 1)",
                        f"{simulation_results['prob_sharpe_gt_1']:.1%}",
                        delta="Good" if simulation_results['prob_sharpe_gt_1'] > 0.5 else "Poor"
                    )

                # Robustness factors
                st.markdown("**Assessment Factors:**")
                for factor in robustness['factors']:
                    if 'WARNING' in factor:
                        st.warning(factor.replace('WARNING: ', '⚠️ '))
                    else:
                        st.success(f"✓ {factor}")

                st.markdown("---")
                st.subheader("Simulation Results")

                # Distribution plots
                col1, col2 = st.columns(2)

                with col1:
                    fig_equity = self.visualizations.plot_monte_carlo_distribution(
                        simulation_results,
                        metric='equity',
                        title="Final Equity Distribution"
                    )
                    st.plotly_chart(fig_equity, use_container_width=True)

                with col2:
                    fig_sharpe = self.visualizations.plot_monte_carlo_distribution(
                        simulation_results,
                        metric='sharpe',
                        title="Sharpe Ratio Distribution"
                    )
                    st.plotly_chart(fig_sharpe, use_container_width=True)

                # Drawdown distribution
                fig_dd = self.visualizations.plot_monte_carlo_distribution(
                    simulation_results,
                    metric='drawdown',
                    title="Max Drawdown Distribution"
                )
                st.plotly_chart(fig_dd, use_container_width=True)

                # Percentile analysis
                st.subheader("Percentile Analysis")
                fig_percentiles = self.visualizations.plot_monte_carlo_percentiles(simulation_results)
                st.plotly_chart(fig_percentiles, use_container_width=True)

                # Detailed statistics
                st.subheader("Detailed Statistics")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Final Equity Percentiles**")
                    eq_pct = simulation_results['equity_percentiles']
                    st.write(f"5th: ${eq_pct['5th']:,.0f}")
                    st.write(f"25th: ${eq_pct['25th']:,.0f}")
                    st.write(f"50th (Median): ${eq_pct['50th']:,.0f}")
                    st.write(f"75th: ${eq_pct['75th']:,.0f}")
                    st.write(f"95th: ${eq_pct['95th']:,.0f}")
                    st.write(f"**Actual: ${simulation_results['actual_equity']:,.0f}**")
                    st.write(f"Percentile Rank: {simulation_results['actual_equity_percentile']:.1f}%")

                with col2:
                    st.markdown("**Sharpe Ratio Percentiles**")
                    sh_pct = simulation_results['sharpe_percentiles']
                    st.write(f"5th: {sh_pct['5th']:.2f}")
                    st.write(f"25th: {sh_pct['25th']:.2f}")
                    st.write(f"50th (Median): {sh_pct['50th']:.2f}")
                    st.write(f"75th: {sh_pct['75th']:.2f}")
                    st.write(f"95th: {sh_pct['95th']:.2f}")
                    st.write(f"**Actual: {simulation_results['actual_sharpe']:.2f}**")
                    st.write(f"Percentile Rank: {simulation_results['actual_sharpe_percentile']:.1f}%")

                with col3:
                    st.markdown("**Probabilities**")
                    st.write(f"Profit: {simulation_results['prob_profit']:.1%}")
                    st.write(f"10% Return: {simulation_results['prob_10pct_return']:.1%}")
                    st.write(f"20% Return: {simulation_results['prob_20pct_return']:.1%}")
                    st.write(f"DD > -20%: {simulation_results['prob_drawdown_gt_20pct']:.1%}")
                    st.write(f"Sharpe > 1: {simulation_results['prob_sharpe_gt_1']:.1%}")

                st.markdown("---")
                st.info("""
                **Interpretation:**
                - If actual results are in top 25% (75th percentile+), strategy is robust
                - High probability of profit (>75%) indicates consistent performance
                - Low result variability (low CV) suggests stable strategy
                - If actual results rank low (<25th percentile), may have gotten lucky
                """)

        else:
            st.info("👆 Click 'Run Simulation' to start Monte Carlo analysis")

    def render_attribution_tab(self, trades):
        """Render attribution analysis tab."""
        st.header("📉 Performance Attribution")

        st.markdown("""
        Understand what drives your returns - costs, specific trades, timing, and exit decisions.
        """)

        if not trades or len([t for t in trades if t.get('action') == 'EXIT']) < 1:
            st.warning("⚠️ No closed trades available for attribution analysis.")
            return

        # Run attribution analysis
        cost_attribution = self.attribution_analyzer.analyze_cost_attribution(trades)
        winlose_attribution = self.attribution_analyzer.analyze_win_lose_attribution(trades)

        # Create sub-tabs for different attribution types
        subtab1, subtab2 = st.tabs(["💰 Cost Attribution", "🎯 Win/Loss Attribution"])

        with subtab1:
            st.subheader("Cost Attribution Analysis")

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total Costs",
                    f"${cost_attribution['total_costs']:,.2f}",
                    delta=f"{cost_attribution['costs_as_pct_gross']:.1f}% of gross"
                )

            with col2:
                st.metric(
                    "Slippage",
                    f"${cost_attribution['total_slippage']:,.2f}",
                    delta=f"{cost_attribution['slippage_pct_of_costs']:.1f}% of costs"
                )

            with col3:
                st.metric(
                    "Fees",
                    f"${cost_attribution['total_fees']:,.2f}",
                    delta=f"{cost_attribution['fees_pct_of_costs']:.1f}% of costs"
                )

            with col4:
                st.metric(
                    "Commission",
                    f"${cost_attribution['total_commission']:,.2f}",
                    delta=f"{cost_attribution['commission_pct_of_costs']:.1f}% of costs"
                )

            # Cost breakdown charts
            st.markdown("---")
            st.subheader("Cost Breakdown")

            fig_costs = self.visualizations.plot_cost_attribution(cost_attribution)
            st.plotly_chart(fig_costs, use_container_width=True)

            # Cost by ticker
            st.subheader("Costs by Ticker")
            fig_ticker_costs = self.visualizations.plot_cost_by_ticker(cost_attribution, top_n=10)
            st.plotly_chart(fig_ticker_costs, use_container_width=True)

            # Cost impact summary
            st.markdown("---")
            st.subheader("Cost Impact Summary")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Gross vs Net Performance**")
                st.write(f"Gross P&L: ${cost_attribution['gross_pnl']:,.2f}")
                st.write(f"Total Costs: -${cost_attribution['total_costs']:,.2f}")
                st.write(f"Net P&L: ${cost_attribution['net_pnl']:,.2f}")
                st.write(f"**Cost Impact: {cost_attribution['costs_as_pct_gross']:.2f}% of gross returns**")

            with col2:
                st.markdown("**Per Trade Costs**")
                st.write(f"Avg Cost/Trade: ${cost_attribution['avg_cost_per_trade']:,.2f}")

                if cost_attribution['costs_as_pct_gross'] > 20:
                    st.error("⚠️ Costs are eating >20% of gross returns!")
                elif cost_attribution['costs_as_pct_gross'] > 10:
                    st.warning("⚠️ Costs are eating >10% of gross returns")
                else:
                    st.success("✓ Costs are reasonable (<10% of gross)")

        with subtab2:
            st.subheader("Win/Loss Attribution Analysis")

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total Trades",
                    f"{winlose_attribution['total_trades']}"
                )

            with col2:
                st.metric(
                    "Win Rate",
                    f"{winlose_attribution['win_rate']:.1f}%",
                    delta="Good" if winlose_attribution['win_rate'] > 50 else "Poor"
                )

            with col3:
                st.metric(
                    "Winning Trades",
                    f"{winlose_attribution['winning_trades']}",
                    delta=f"{winlose_attribution['winning_trades']/winlose_attribution['total_trades']*100:.1f}%"
                )

            with col4:
                st.metric(
                    "Total P&L",
                    f"${winlose_attribution['total_pnl']:,.2f}"
                )

            # Win/Loss by ticker
            st.markdown("---")
            st.subheader("Attribution by Ticker")
            fig_winlose = self.visualizations.plot_winlose_attribution(winlose_attribution)
            st.plotly_chart(fig_winlose, use_container_width=True)

            # Attribution by exit reason
            st.subheader("Attribution by Exit Reason")
            fig_exit = self.visualizations.plot_attribution_by_exit_reason(winlose_attribution)
            st.plotly_chart(fig_exit, use_container_width=True)

            # Concentration analysis
            st.markdown("---")
            st.subheader("P&L Concentration")

            col1, col2 = st.columns(2)

            with col1:
                top10_pct = winlose_attribution['concentration_top10pct']
                st.metric(
                    "Top 10% of Trades",
                    f"{top10_pct:.1f}%",
                    delta=f"of total P&L"
                )

                if top10_pct > 80:
                    st.warning("⚠️ Very concentrated - top 10% of trades drive >80% of returns")
                elif top10_pct > 60:
                    st.info("ℹ️ Moderately concentrated - top 10% drive >60% of returns")
                else:
                    st.success("✓ Well diversified returns")

            with col2:
                top20_pct = winlose_attribution['concentration_top20pct']
                st.metric(
                    "Top 20% of Trades",
                    f"{top20_pct:.1f}%",
                    delta="of total P&L"
                )

            # Detailed attribution table
            st.markdown("---")
            st.subheader("Detailed Attribution by Ticker")

            by_ticker = winlose_attribution['by_ticker']
            if not by_ticker.empty:
                st.dataframe(
                    by_ticker.head(20),
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No ticker data available")

    def render_configuration_tab(self):
        """Render configuration interface."""
        st.header("⚙️ Strategy Configuration")

        st.info("🚧 Configuration interface coming soon!")

        st.markdown("""
        This tab will allow you to configure:

        - **Strategy Selection**: Choose from available strategies
        - **Date Range**: Set backtest start and end dates
        - **Timeframe**: Daily, intraday, or tick-level data
        - **Risk Parameters**: Position sizing, stop losses, etc.
        - **Entry/Exit Rules**: Configure signal parameters
        - **Screening Criteria**: Filters for stock selection

        For now, please modify the `config.py` file directly.
        """)

        # Show current config if available
        results = self.data_interface.load_backtest_results()
        config = results.get('config', {})

        if config:
            st.subheader("Current Configuration")
            st.json(config)

    def render_quick_start_guide(self):
        """Render quick start guide when no data is available."""
        st.subheader("🚀 Quick Start Guide")

        st.markdown("""
        ### How to use this dashboard:

        1. **Run a backtest** using one of these methods:
           ```bash
           python main.py
           ```

        2. **The dashboard will automatically load** the most recent backtest results

        3. **Use the sidebar** to:
           - Switch between replay and live mode
           - Select different backtests to analyze
           - Configure auto-refresh settings

        4. **Explore the tabs**:
           - 📊 **Overview**: Key metrics and equity curve
           - 📈 **Performance**: Detailed performance analysis
           - 💼 **Trades**: Individual trade analysis
           - 🎯 **Risk**: Risk metrics and drawdown analysis
           - ⚙️ **Configuration**: Strategy settings (coming soon)

        ### Need help?
        - Check the documentation in `/docs`
        - Review example backtests in `/outputs`
        """)


def main():
    """Main entry point."""
    dashboard = BacktestDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
