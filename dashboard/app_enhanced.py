"""
Enhanced Interactive Backtesting Dashboard - Central Command Center

NEW FEATURES:
- Strategy selection and execution
- Parameter configuration UI
- Backtest runner (run backtests from dashboard)
- Real-time progress tracking
- Integrated results viewing

This is your single entry point for all operations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dashboard.data_interface import BacktestDataInterface
from dashboard.metrics import MetricsCalculator
from dashboard.visualizations import DashboardVisualizations
from dashboard.monte_carlo import MonteCarloSimulator
from dashboard.attribution import AttributionAnalyzer
from dashboard.strategy_registry import get_registry
from dashboard.backtest_executor import get_executor, BacktestStatus


# Page configuration
st.set_page_config(
    page_title="X-Dividend Trading Dashboard",
    page_icon="💎",
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
    .strategy-card {
        background-color: #2a2a2a;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4a9eff;
        margin: 10px 0;
    }
    .recommended {
        border-left-color: #00ff00 !important;
    }
</style>
""", unsafe_allow_html=True)


class EnhancedBacktestDashboard:
    """Enhanced dashboard with strategy management and execution."""

    def __init__(self):
        """Initialize dashboard."""
        self.data_interface = None
        self.metrics_calculator = MetricsCalculator()
        self.visualizations = DashboardVisualizations(theme='plotly_dark')
        self.monte_carlo = MonteCarloSimulator(n_simulations=1000)
        self.attribution_analyzer = AttributionAnalyzer()
        self.strategy_registry = get_registry()
        self.backtest_executor = get_executor()

        # Initialize session state
        if 'mode' not in st.session_state:
            st.session_state.mode = 'manage'  # Default to strategy management

        if 'selected_strategy' not in st.session_state:
            st.session_state.selected_strategy = 'xdiv_fixed'  # Default to recommended

        if 'selected_backtest' not in st.session_state:
            st.session_state.selected_backtest = None

        if 'current_run_id' not in st.session_state:
            st.session_state.current_run_id = None

    def run(self):
        """Run the dashboard application."""
        st.title("💎 X-Dividend Trading Dashboard")
        st.caption("Central Command Center for All Operations")

        # Sidebar
        self.render_sidebar()

        # Main content based on mode
        if st.session_state.mode == 'manage':
            self.render_strategy_management()
        elif st.session_state.mode == 'results':
            self.render_results_view()

    def render_sidebar(self):
        """Render sidebar with controls."""
        st.sidebar.title("⚙️ Dashboard Controls")

        # Mode selection
        st.sidebar.subheader("Mode")
        mode = st.sidebar.radio(
            "Select Mode",
            options=['manage', 'results'],
            format_func=lambda x: "🚀 Strategy Manager" if x == 'manage' else "📊 View Results",
            key='mode'
        )

        st.sidebar.markdown("---")

        # Quick stats
        st.sidebar.subheader("📈 Quick Stats")

        try:
            data_interface = BacktestDataInterface(mode='replay')
            available_backtests = data_interface.list_available_backtests()

            st.sidebar.metric("Available Backtests", len(available_backtests))

            # Recent runs
            recent_runs = self.backtest_executor.get_recent_runs(limit=5)
            completed_runs = [r for r in recent_runs if r.status == BacktestStatus.COMPLETED]
            st.sidebar.metric("Completed Runs", len(completed_runs))

        except:
            pass

        st.sidebar.markdown("---")

        # Info
        st.sidebar.subheader("ℹ️ About")
        st.sidebar.info(
            "**X-Dividend Dashboard v3.0**\n\n"
            "Your central command center for dividend capture strategies.\n\n"
            "🚀 Run backtests\n"
            "⚙️ Configure parameters\n"
            "📊 Analyze results\n"
            "🎯 Optimize strategies"
        )

    def render_strategy_management(self):
        """Render strategy management and execution tab."""
        st.header("🚀 Strategy Management & Execution")

        # Create two columns
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Select Strategy")

            # List available strategies
            strategies = self.strategy_registry.list_strategies()

            for strategy in strategies:
                # Create strategy card
                is_recommended = 'recommended' in strategy.tags
                card_class = "strategy-card recommended" if is_recommended else "strategy-card"

                with st.container():
                    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)

                    # Title with star if recommended
                    title = f"{strategy.display_name}"
                    if is_recommended:
                        st.markdown(f"### ⭐ {title}")
                    else:
                        st.markdown(f"### {title}")

                    st.caption(f"v{strategy.version} • {strategy.strategy_type}")
                    st.write(strategy.description)

                    # Tags
                    if strategy.tags:
                        tags_str = " • ".join([f"`{tag}`" for tag in strategy.tags[:3]])
                        st.markdown(tags_str)

                    # Select button
                    if st.button(
                        "Select" if st.session_state.selected_strategy != strategy.name else "✓ Selected",
                        key=f"select_{strategy.name}",
                        disabled=st.session_state.selected_strategy == strategy.name
                    ):
                        st.session_state.selected_strategy = strategy.name
                        st.rerun()

                    st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.subheader("Configure & Run")

            selected_strategy = self.strategy_registry.get_strategy(st.session_state.selected_strategy)

            if not selected_strategy:
                st.warning("Please select a strategy from the left panel.")
                return

            # Strategy info
            st.markdown(f"### {selected_strategy.display_name}")
            st.write(selected_strategy.description)

            # Parameter configuration
            st.markdown("---")
            st.subheader("Parameters")

            user_parameters = {}

            for param_name, param_def in selected_strategy.parameters.items():
                param_type = param_def.get('type', 'str')
                default_value = param_def.get('default')
                description = param_def.get('description', '')

                if param_type == 'float':
                    min_val = param_def.get('min', 0.0)
                    max_val = param_def.get('max', 1000000.0)
                    user_parameters[param_name] = st.number_input(
                        f"{param_name}",
                        min_value=min_val,
                        max_value=max_val,
                        value=float(default_value),
                        help=description
                    )

                elif param_type == 'int':
                    min_val = param_def.get('min', 0)
                    max_val = param_def.get('max', 1000000)
                    user_parameters[param_name] = st.number_input(
                        f"{param_name}",
                        min_value=min_val,
                        max_value=max_val,
                        value=int(default_value),
                        help=description
                    )

                elif param_type == 'date':
                    try:
                        default_date = datetime.strptime(str(default_value), '%Y-%m-%d').date()
                    except:
                        default_date = datetime.now().date()

                    user_parameters[param_name] = st.date_input(
                        f"{param_name}",
                        value=default_date,
                        help=description
                    ).strftime('%Y-%m-%d')

                elif param_type == 'bool':
                    user_parameters[param_name] = st.checkbox(
                        f"{param_name}",
                        value=bool(default_value),
                        help=description
                    )

                else:
                    user_parameters[param_name] = st.text_input(
                        f"{param_name}",
                        value=str(default_value),
                        help=description
                    )

            # Execution
            st.markdown("---")
            st.subheader("Execute Backtest")

            # Check if backtest is running
            if st.session_state.current_run_id:
                current_run = self.backtest_executor.get_run(st.session_state.current_run_id)

                if current_run and current_run.status == BacktestStatus.RUNNING:
                    st.info("🏃 Backtest is currently running...")

                    # Progress bar
                    st.progress(current_run.progress, text=current_run.progress_message)

                    # Live logs
                    with st.expander("📋 Live Logs", expanded=True):
                        log_text = "\n".join(current_run.logs[-20:])  # Last 20 lines
                        st.code(log_text, language="text")

                    # Auto-refresh while running
                    time.sleep(2)
                    st.rerun()

                elif current_run and current_run.status == BacktestStatus.COMPLETED:
                    st.success(f"✅ Backtest completed successfully!")

                    runtime = (current_run.end_time - current_run.start_time).total_seconds()
                    st.write(f"**Runtime:** {runtime:.1f} seconds")

                    # Show logs
                    with st.expander("📋 Execution Logs"):
                        log_text = "\n".join(current_run.logs)
                        st.code(log_text, language="text")

                    # Button to view results
                    if st.button("📊 View Results", type="primary"):
                        st.session_state.mode = 'results'
                        st.rerun()

                    # Reset button
                    if st.button("🔄 Run Another Backtest"):
                        st.session_state.current_run_id = None
                        st.rerun()

                elif current_run and current_run.status == BacktestStatus.FAILED:
                    st.error(f"❌ Backtest failed: {current_run.error}")

                    with st.expander("📋 Error Details", expanded=True):
                        log_text = "\n".join(current_run.logs)
                        st.code(log_text, language="text")

                    if st.button("🔄 Try Again"):
                        st.session_state.current_run_id = None
                        st.rerun()

            else:
                # Ready to run
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("▶️ Run Backtest", type="primary", use_container_width=True):
                        # Create and execute run
                        try:
                            run_id = self.backtest_executor.create_run(
                                strategy_name=selected_strategy.name,
                                parameters=user_parameters
                            )

                            self.backtest_executor.execute_run(
                                run_id=run_id,
                                strategy_metadata=selected_strategy,
                                config_overrides=selected_strategy.config_overrides
                            )

                            st.session_state.current_run_id = run_id
                            st.rerun()

                        except Exception as e:
                            st.error(f"Failed to start backtest: {e}")

                with col2:
                    # Show recent runs
                    recent_runs = self.backtest_executor.get_recent_runs(limit=3)
                    if recent_runs:
                        st.caption(f"Recent runs: {len(recent_runs)}")

    def render_results_view(self):
        """Render results viewing interface (original dashboard tabs)."""
        st.header("📊 Results Analysis")

        # Initialize data interface
        self.data_interface = BacktestDataInterface(mode='replay')

        # Backtest selection
        available_backtests = self.data_interface.list_available_backtests()

        if not available_backtests:
            st.warning("⚠️ No backtest results available. Please run a backtest first.")
            if st.button("🚀 Go to Strategy Manager"):
                st.session_state.mode = 'manage'
                st.rerun()
            return

        # Selectbox for backtest
        backtest_names = [b['name'] for b in available_backtests]
        selected = st.selectbox(
            "Select Backtest to Analyze",
            options=backtest_names,
            index=0
        )
        st.session_state.selected_backtest = selected

        # Load data
        results = self.data_interface.load_backtest_results(selected)
        equity_curve = results.get('equity_curve', pd.Series())
        trades = results.get('trades', [])
        position_values = results.get('position_values', pd.Series())

        if len(equity_curve) == 0:
            st.warning("⚠️ No data in selected backtest.")
            return

        # Calculate metrics
        returns = equity_curve.pct_change().dropna()
        benchmark_returns = self.load_benchmark(equity_curve.index[0], equity_curve.index[-1])

        metrics = self.metrics_calculator.calculate_all_metrics(
            equity_curve=equity_curve,
            trades=trades,
            benchmark_returns=benchmark_returns
        )

        # Render tabs (simplified version)
        tab1, tab2, tab3 = st.tabs([
            "📊 Overview",
            "📈 Performance",
            "💼 Trades"
        ])

        with tab1:
            self.render_overview_tab(equity_curve, trades, metrics, benchmark_returns)

        with tab2:
            self.render_performance_tab(equity_curve, returns, trades, metrics)

        with tab3:
            self.render_trades_tab(trades, equity_curve)

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
        """Render overview tab (simplified)."""
        st.subheader("Key Metrics")

        # Top metrics row
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            total_return = metrics.get('total_return_pct', 0)
            st.metric(
                "Total Return",
                f"{total_return:.2f}%",
                delta=f"{total_return:.2f}%"
            )

        with col2:
            sharpe = metrics.get('sharpe_ratio', 0)
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
                f"{total_trades}"
            )

        # Equity curve
        st.subheader("Equity Curve")

        if benchmark_returns is not None and len(benchmark_returns) > 0:
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

    def render_performance_tab(self, equity_curve, returns, trades, metrics):
        """Render performance tab (simplified)."""
        st.subheader("Performance Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Return Metrics**")
            st.write(f"**CAGR:** {metrics.get('cagr_pct', 0):.2f}%")
            st.write(f"**Total Return:** {metrics.get('total_return_pct', 0):.2f}%")
            st.write(f"**Best Day:** {metrics.get('best_day', 0)*100:.2f}%")
            st.write(f"**Worst Day:** {metrics.get('worst_day', 0)*100:.2f}%")

        with col2:
            st.markdown("**Risk Metrics**")
            st.write(f"**Sharpe:** {metrics.get('sharpe_ratio', 0):.2f}")
            st.write(f"**Sortino:** {metrics.get('sortino_ratio', 0):.2f}")
            st.write(f"**Max DD:** {metrics.get('max_drawdown_pct', 0):.2f}%")
            st.write(f"**Volatility:** {metrics.get('daily_return_std', 0)*100:.3f}%")

        with col3:
            st.markdown("**Trade Statistics**")
            st.write(f"**Win Rate:** {metrics.get('win_rate_pct', 0):.1f}%")
            st.write(f"**Profit Factor:** {metrics.get('profit_factor', 0):.2f}")
            st.write(f"**Total Trades:** {metrics.get('total_trades', 0)}")
            st.write(f"**Avg Hold:** {metrics.get('avg_holding_days', 0):.1f} days")

    def render_trades_tab(self, trades, equity_curve):
        """Render trades tab (simplified)."""
        st.subheader("Trade Log")

        closed_trades = [t for t in trades if t.get('action') == 'EXIT']

        if not closed_trades:
            st.info("No closed trades yet.")
            return

        df = pd.DataFrame(closed_trades)

        # Select columns to display
        display_cols = ['date', 'ticker', 'shares', 'price', 'pnl', 'exit_reason']
        display_cols = [col for col in display_cols if col in df.columns]

        df_display = df[display_cols].copy()

        # Format
        if 'date' in df_display.columns:
            df_display['date'] = pd.to_datetime(df_display['date']).dt.strftime('%Y-%m-%d')

        if 'pnl' in df_display.columns:
            df_display['pnl'] = df_display['pnl'].apply(lambda x: f"${x:,.2f}")

        if 'price' in df_display.columns:
            df_display['price'] = df_display['price'].apply(lambda x: f"${x:.2f}")

        st.dataframe(df_display, use_container_width=True, height=400)

        # Download
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trade Log",
            data=csv,
            file_name=f"trade_log_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


def main():
    """Main entry point."""
    dashboard = EnhancedBacktestDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
