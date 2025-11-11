"""
Analytics Module for Dividend Capture Algorithm
Generates plots, reports, and performance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional
import warnings
import os
warnings.filterwarnings('ignore')

from config import analytics_config

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class PerformanceAnalytics:
    """
    Comprehensive performance analytics and visualization
    """
    
    def __init__(self, backtest_results: Dict):
        self.results = backtest_results
        self.equity_curve = backtest_results.get('equity_curve', pd.DataFrame())
        self.returns = backtest_results.get('returns', pd.Series())
        
    def generate_all_plots(self, save_dir: str = '/mnt/user-data/outputs'):
        """Generate all visualization plots"""

        # Create output directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        print("\n📊 Generating performance visualizations...")

        plots_generated = []
        
        # 1. Equity curve
        if analytics_config.plot_equity_curve and len(self.equity_curve) > 0:
            filepath = f"{save_dir}/equity_curve.png"
            self.plot_equity_curve(save_path=filepath)
            plots_generated.append(filepath)
        
        # 2. Drawdown
        if analytics_config.plot_drawdown and len(self.equity_curve) > 0:
            filepath = f"{save_dir}/drawdown.png"
            self.plot_drawdown(save_path=filepath)
            plots_generated.append(filepath)
        
        # 3. Monthly returns
        if analytics_config.plot_monthly_returns and 'monthly_returns' in self.results:
            filepath = f"{save_dir}/monthly_returns.png"
            self.plot_monthly_returns(save_path=filepath)
            plots_generated.append(filepath)
        
        # 4. Rolling metrics
        if analytics_config.plot_rolling_metrics and len(self.returns) > 0:
            filepath = f"{save_dir}/rolling_metrics.png"
            self.plot_rolling_metrics(save_path=filepath)
            plots_generated.append(filepath)
        
        # 5. Distribution analysis
        if len(self.returns) > 0:
            filepath = f"{save_dir}/returns_distribution.png"
            self.plot_returns_distribution(save_path=filepath)
            plots_generated.append(filepath)
        
        print(f"✅ Generated {len(plots_generated)} plots")
        
        return plots_generated
    
    def plot_equity_curve(self, save_path: Optional[str] = None):
        """Plot equity curve with benchmark comparison"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        # Main equity curve
        equity = self.equity_curve['portfolio_value']
        dates = self.equity_curve.index
        
        ax1.plot(dates, equity, linewidth=2, label='Strategy', color='#2E86AB')
        ax1.axhline(y=self.results['initial_capital'], color='gray', 
                   linestyle='--', linewidth=1, alpha=0.5, label='Initial Capital')
        
        # Fill area
        ax1.fill_between(dates, equity, self.results['initial_capital'], 
                        where=(equity >= self.results['initial_capital']),
                        alpha=0.3, color='green', interpolate=True)
        ax1.fill_between(dates, equity, self.results['initial_capital'],
                        where=(equity < self.results['initial_capital']),
                        alpha=0.3, color='red', interpolate=True)
        
        ax1.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Add performance annotations
        final_value = equity.iloc[-1]
        total_return = self.results['total_return_pct']
        sharpe = self.results['sharpe_ratio']
        
        textstr = f'Final Value: ${final_value:,.0f}\n'
        textstr += f'Total Return: {total_return:+.2f}%\n'
        textstr += f'Sharpe Ratio: {sharpe:.2f}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top', bbox=props)
        
        # Number of positions over time
        positions = self.equity_curve['num_positions']
        ax2.plot(dates, positions, linewidth=1.5, color='#A23B72')
        ax2.fill_between(dates, 0, positions, alpha=0.3, color='#A23B72')
        ax2.set_ylabel('# Positions', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Number of Active Positions', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Equity curve saved to {save_path}")
        
        plt.close()
    
    def plot_drawdown(self, save_path: Optional[str] = None):
        """Plot underwater (drawdown) chart"""
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        drawdown = self.results['drawdown'] * 100  # Convert to percentage
        dates = drawdown.index
        
        ax.fill_between(dates, 0, drawdown, color='red', alpha=0.3)
        ax.plot(dates, drawdown, color='darkred', linewidth=1.5)
        
        ax.set_title('Drawdown Analysis', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add max drawdown line
        max_dd = self.results['max_drawdown_pct']
        ax.axhline(y=max_dd, color='red', linestyle='--', 
                  linewidth=2, label=f'Max Drawdown: {max_dd:.2f}%')
        
        # Add target drawdown line
        ax.axhline(y=-15, color='orange', linestyle='--', 
                  linewidth=1, alpha=0.5, label='Target Max DD: -15%')
        
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Drawdown plot saved to {save_path}")
        
        plt.close()
    
    def plot_monthly_returns(self, save_path: Optional[str] = None):
        """Plot monthly returns heatmap"""
        
        monthly_returns = self.results['monthly_returns'] * 100  # Convert to percentage
        
        # Reshape into year x month grid
        monthly_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create heatmap
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=0, cbar_kws={'label': 'Return (%)'},
                   linewidths=0.5, ax=ax)
        
        ax.set_title('Monthly Returns Heatmap', fontsize=16, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        
        # Set month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_labels)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Monthly returns saved to {save_path}")
        
        plt.close()
    
    def plot_rolling_metrics(self, save_path: Optional[str] = None):
        """Plot rolling Sharpe and other metrics"""
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        returns = self.returns
        dates = returns.index
        
        # Calculate rolling metrics
        window = analytics_config.rolling_sharpe_window
        
        # 1. Rolling Sharpe
        rf_daily = (1 + analytics_config.risk_free_rate) ** (1/252) - 1
        excess_returns = returns - rf_daily
        rolling_sharpe = (excess_returns.rolling(window).mean() / 
                         excess_returns.rolling(window).std() * np.sqrt(252))
        
        axes[0].plot(dates, rolling_sharpe, linewidth=1.5, color='#2E86AB')
        axes[0].axhline(y=1.5, color='green', linestyle='--', alpha=0.5, label='Target: 1.5')
        axes[0].axhline(y=1.0, color='orange', linestyle='--', alpha=0.5, label='Good: 1.0')
        axes[0].axhline(y=0, color='red', linestyle='-', alpha=0.3)
        axes[0].set_title(f'Rolling Sharpe Ratio ({window}-day)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Sharpe Ratio', fontsize=10)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Rolling Volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
        axes[1].plot(dates, rolling_vol, linewidth=1.5, color='#A23B72')
        axes[1].axhline(y=15, color='green', linestyle='--', alpha=0.5, label='Target: 15%')
        axes[1].axhline(y=25, color='orange', linestyle='--', alpha=0.5, label='High: 25%')
        axes[1].set_title(f'Rolling Volatility ({window}-day)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Volatility (%)', fontsize=10)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Cumulative Returns
        cumulative_returns = (1 + returns).cumprod() - 1
        axes[2].plot(dates, cumulative_returns * 100, linewidth=2, color='#06A77D')
        axes[2].fill_between(dates, 0, cumulative_returns * 100, alpha=0.3, color='#06A77D')
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[2].set_title('Cumulative Returns', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Return (%)', fontsize=10)
        axes[2].set_xlabel('Date', fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Rolling metrics saved to {save_path}")
        
        plt.close()
    
    def plot_returns_distribution(self, save_path: Optional[str] = None):
        """Plot returns distribution and QQ plot"""
        
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        returns_pct = self.returns * 100
        
        # 1. Histogram with normal curve
        ax1 = fig.add_subplot(gs[0, :])
        
        n, bins, patches = ax1.hist(returns_pct, bins=50, density=True, 
                                    alpha=0.7, color='#2E86AB', edgecolor='black')
        
        # Fit normal distribution
        mu, sigma = returns_pct.mean(), returns_pct.std()
        x = np.linspace(returns_pct.min(), returns_pct.max(), 100)
        ax1.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * 
                np.exp(-(x - mu)**2 / (2 * sigma**2)),
                linewidth=2, color='red', label='Normal Distribution')
        
        ax1.axvline(x=mu, color='green', linestyle='--', linewidth=2, label=f'Mean: {mu:.3f}%')
        ax1.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Return (%)', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. QQ Plot
        ax2 = fig.add_subplot(gs[1, 0])
        
        from scipy import stats
        stats.probplot(returns_pct, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normality Test)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Box Plot
        ax3 = fig.add_subplot(gs[1, 1])
        
        ax3.boxplot(returns_pct, vert=True)
        ax3.set_title('Returns Box Plot', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Return (%)', fontsize=12)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Statistics Table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        stats_data = [
            ['Metric', 'Value'],
            ['Mean', f'{returns_pct.mean():.4f}%'],
            ['Std Dev', f'{returns_pct.std():.4f}%'],
            ['Skewness', f'{returns_pct.skew():.4f}'],
            ['Kurtosis', f'{returns_pct.kurtosis():.4f}'],
            ['Min', f'{returns_pct.min():.4f}%'],
            ['25%', f'{returns_pct.quantile(0.25):.4f}%'],
            ['Median', f'{returns_pct.median():.4f}%'],
            ['75%', f'{returns_pct.quantile(0.75):.4f}%'],
            ['Max', f'{returns_pct.max():.4f}%'],
        ]
        
        table = ax4.table(cellText=stats_data, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color header row
        for i in range(2):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Distribution analysis saved to {save_path}")
        
        plt.close()
    
    def generate_html_report(self, save_path: str = '/mnt/user-data/outputs/backtest_report.html'):
        """Generate comprehensive HTML report"""

        if not analytics_config.generate_html_report:
            return

        # Create output directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        print("\n📝 Generating HTML report...")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dividend Capture Strategy - Backtest Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 40px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2E86AB;
                    border-bottom: 3px solid #2E86AB;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #A23B72;
                    margin-top: 30px;
                }}
                .metric-grid {{
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background: #f9f9f9;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #2E86AB;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #666;
                    margin-bottom: 5px;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #333;
                }}
                .positive {{ color: #06A77D; }}
                .negative {{ color: #D62828; }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #2E86AB;
                    color: white;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .assessment {{
                    padding: 15px;
                    margin: 20px 0;
                    border-radius: 5px;
                    font-weight: bold;
                }}
                .excellent {{ background-color: #d4edda; border-left: 4px solid #28a745; }}
                .good {{ background-color: #fff3cd; border-left: 4px solid #ffc107; }}
                .needs-improvement {{ background-color: #f8d7da; border-left: 4px solid #dc3545; }}
                .timestamp {{
                    color: #999;
                    font-size: 12px;
                    text-align: right;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📈 Dividend Capture Strategy - Backtest Report</h1>
                <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>💰 Performance Summary</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-label">Total Return</div>
                        <div class="metric-value {'positive' if self.results['total_return'] > 0 else 'negative'}">
                            {self.results['total_return_pct']:+.2f}%
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Annual Return</div>
                        <div class="metric-value {'positive' if self.results['annual_return'] > 0 else 'negative'}">
                            {self.results['annual_return_pct']:+.2f}%
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Sharpe Ratio</div>
                        <div class="metric-value">
                            {self.results['sharpe_ratio']:.2f}
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Sortino Ratio</div>
                        <div class="metric-value">
                            {self.results['sortino_ratio']:.2f}
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Max Drawdown</div>
                        <div class="metric-value negative">
                            {self.results['max_drawdown_pct']:.2f}%
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value">
                            {self.results['win_rate_pct']:.1f}%
                        </div>
                    </div>
                </div>
                
                <h2>📊 Trade Statistics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Total Trades</td>
                        <td>{self.results['total_trades']}</td>
                    </tr>
                    <tr>
                        <td>Average Win</td>
                        <td class="positive">{self.results['avg_win_pct']:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Average Loss</td>
                        <td class="negative">{self.results['avg_loss_pct']:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Profit Factor</td>
                        <td>{self.results['profit_factor']:.2f}</td>
                    </tr>
                    <tr>
                        <td>Average Holding Period</td>
                        <td>{self.results['avg_holding_days']:.1f} days</td>
                    </tr>
                </table>
                
                <h2>⚠️ Risk Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Annual Volatility</td>
                        <td>{self.results['volatility_annual']*100:.2f}%</td>
                    </tr>
                    <tr>
                        <td>VaR 95%</td>
                        <td>{self.results['var_95_pct']:.2f}%</td>
                    </tr>
                    <tr>
                        <td>CVaR 95%</td>
                        <td>{self.results['cvar_95_pct']:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Calmar Ratio</td>
                        <td>{self.results['calmar_ratio']:.2f}</td>
                    </tr>
                </table>
                
                <h2>🎯 Strategy Assessment</h2>
                """
        
        # Assessment boxes
        if self.results['sharpe_ratio'] >= 1.5:
            html += '<div class="assessment excellent">✅ EXCELLENT: Sharpe ratio exceeds target (>1.5)</div>'
        elif self.results['sharpe_ratio'] >= 1.0:
            html += '<div class="assessment good">✓ GOOD: Sharpe ratio above 1.0</div>'
        else:
            html += '<div class="assessment needs-improvement">⚠️ NEEDS IMPROVEMENT: Sharpe ratio below 1.0</div>'
        
        if self.results['win_rate_pct'] >= 55:
            html += '<div class="assessment excellent">✅ EXCELLENT: Win rate exceeds target (>55%)</div>'
        else:
            html += '<div class="assessment needs-improvement">⚠️ NEEDS IMPROVEMENT: Win rate below 55%</div>'
        
        if self.results['annual_return_pct'] >= 10:
            html += '<div class="assessment excellent">✅ EXCELLENT: Annual return exceeds target (>10%)</div>'
        else:
            html += '<div class="assessment needs-improvement">⚠️ NEEDS IMPROVEMENT: Annual return below 10%</div>'
        
        html += """
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(save_path, 'w') as f:
            f.write(html)
        
        print(f"✅ HTML report generated: {save_path}")
        
        return save_path


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_performance_report(backtest_results: Dict, 
                             output_dir: str = '/mnt/user-data/outputs'):
    """Create comprehensive performance report with all plots and HTML"""
    
    analytics = PerformanceAnalytics(backtest_results)
    
    # Generate all plots
    plots = analytics.generate_all_plots(save_dir=output_dir)
    
    # Generate HTML report
    html_report = analytics.generate_html_report(
        save_path=f"{output_dir}/backtest_report.html"
    )
    
    print(f"\n✅ Performance report complete!")
    print(f"   - {len(plots)} plots generated")
    print(f"   - HTML report: {html_report}")
    
    return plots, html_report


if __name__ == '__main__':
    # This would normally be called with backtest results
    print("Analytics module loaded. Use create_performance_report() with backtest results.")
