"""
Risk Management Module for Dividend Capture Algorithm
Handles position sizing, portfolio constraints, and risk monitoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats

from config import risk_config, backtest_config, analytics_config


class RiskManager:
    """
    Manages portfolio-level risk and position sizing
    """
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.cash = initial_capital
        self.peak_capital = initial_capital
        
        # Risk tracking
        self.daily_returns = []
        self.monthly_returns = []
        
    def calculate_position_size(self, signal: Dict, current_positions: Dict,
                                current_price: float) -> int:
        """
        Calculate optimal position size using Kelly Criterion

        Args:
            signal: Entry signal with expected return
            current_positions: Dictionary of current positions
            current_price: Current stock price

        Returns:
            Number of shares to buy
        """
        # Validate inputs
        if current_price <= 0 or self.current_capital <= 0:
            return 0

        # Kelly Criterion: f* = (p*b - q) / b
        # Where p = win rate, b = win/loss ratio, q = 1-p

        p = risk_config.estimated_win_rate
        b = risk_config.estimated_win_loss_ratio
        q = 1 - p

        # Avoid division by zero
        if b == 0:
            return 0

        kelly_fraction = (p * b - q) / b
        kelly_adjusted = kelly_fraction * risk_config.kelly_safety_factor

        # Maximum position size based on Kelly
        kelly_position_value = self.current_capital * kelly_adjusted

        # Apply maximum position percentage constraint
        max_position_value = self.current_capital * risk_config.max_position_pct

        # Take the minimum of Kelly and max position
        target_position_value = min(kelly_position_value, max_position_value)

        # Check available cash
        available_cash = self.cash * (1 - risk_config.min_cash_reserve)
        target_position_value = min(target_position_value, available_cash)

        # Convert to shares
        shares = int(target_position_value / current_price)
        
        # Ensure we don't exceed max positions
        if len(current_positions) >= risk_config.max_positions:
            return 0
        
        return max(0, shares)
    
    def check_sector_limits(self, signal: Dict, current_positions: Dict) -> bool:
        """
        Check if adding position would violate sector concentration limits
        
        Args:
            signal: Entry signal with sector info
            current_positions: Dictionary of current positions
            
        Returns:
            True if position is allowed, False otherwise
        """
        sector = signal.get('sector', 'Unknown')
        
        # Calculate current sector exposure
        sector_value = 0
        total_value = 0
        
        for ticker, pos in current_positions.items():
            pos_value = pos.get('position_value', 0)
            total_value += pos_value
            
            if pos.get('sector') == sector:
                sector_value += pos_value
        
        # Add proposed position
        proposed_value = signal.get('position_value', 0)
        new_sector_value = sector_value + proposed_value
        new_total_value = total_value + proposed_value
        
        # Check limit
        if new_total_value > 0:
            sector_exposure = new_sector_value / new_total_value
            if sector_exposure > risk_config.max_sector_exposure:
                print(f"⚠️  Sector limit exceeded: {sector} would be {sector_exposure*100:.1f}%")
                return False
        
        return True
    
    def calculate_portfolio_beta(self, current_positions: Dict, 
                                market_returns: pd.Series) -> float:
        """
        Calculate portfolio beta relative to market
        
        Args:
            current_positions: Dictionary of current positions
            market_returns: Market return series (e.g., SPY)
            
        Returns:
            Portfolio beta
        """
        if len(current_positions) == 0:
            return 0
        
        # Weight by position value
        total_value = sum(pos.get('position_value', 0) for pos in current_positions.values())
        
        if total_value == 0:
            return 0
        
        # Weighted average beta
        portfolio_beta = 0
        for ticker, pos in current_positions.items():
            weight = pos.get('position_value', 0) / total_value
            stock_beta = pos.get('beta', 1.0)  # From fundamentals
            portfolio_beta += weight * stock_beta
        
        return portfolio_beta
    
    def calculate_portfolio_var(self, returns: pd.Series, 
                               confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) using historical simulation
        
        Args:
            returns: Historical returns series
            confidence: Confidence level (default 95%)
            
        Returns:
            VaR as a decimal (e.g., 0.02 for 2%)
        """
        if len(returns) < 30:
            return 0
        
        # Historical VaR
        var = np.percentile(returns, (1 - confidence) * 100)
        
        return abs(var)
    
    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall)
        
        Args:
            returns: Historical returns series
            confidence: Confidence level
            
        Returns:
            CVaR as a decimal
        """
        if len(returns) < 30:
            return 0
        
        var_threshold = np.percentile(returns, (1 - confidence) * 100)
        cvar = returns[returns <= var_threshold].mean()
        
        return abs(cvar)
    
    def check_risk_limits(self, current_positions: Dict, 
                         daily_returns: pd.Series) -> Tuple[bool, List[str]]:
        """
        Check if portfolio is within risk limits
        
        Args:
            current_positions: Dictionary of current positions
            daily_returns: Recent daily returns
            
        Returns:
            Tuple of (is_within_limits, list_of_violations)
        """
        violations = []
        
        # 1. Check VaR limit
        if len(daily_returns) >= 30:
            var_95 = self.calculate_portfolio_var(daily_returns, 0.95)
            if var_95 > risk_config.max_portfolio_var_95:
                violations.append(f"VaR 95% ({var_95*100:.2f}%) exceeds limit ({risk_config.max_portfolio_var_95*100:.2f}%)")
        
        # 2. Check drawdown
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_drawdown > risk_config.max_drawdown_pct:
            violations.append(f"Drawdown ({current_drawdown*100:.2f}%) exceeds limit ({risk_config.max_drawdown_pct*100:.2f}%)")
        
        # 3. Check daily loss limit
        if len(daily_returns) > 0:
            today_return = daily_returns.iloc[-1]
            if today_return < -risk_config.max_daily_loss_pct:
                violations.append(f"Daily loss ({today_return*100:.2f}%) exceeds limit ({risk_config.max_daily_loss_pct*100:.2f}%)")
        
        # 4. Check position count
        if len(current_positions) > risk_config.max_positions:
            violations.append(f"Position count ({len(current_positions)}) exceeds limit ({risk_config.max_positions})")
        
        # 5. Check cash reserve
        cash_pct = self.cash / self.current_capital if self.current_capital > 0 else 0
        if cash_pct < risk_config.min_cash_reserve:
            violations.append(f"Cash reserve ({cash_pct*100:.1f}%) below minimum ({risk_config.min_cash_reserve*100:.1f}%)")
        
        is_within_limits = len(violations) == 0
        
        return is_within_limits, violations
    
    def calculate_correlation_matrix(self, position_returns: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Calculate correlation matrix for current positions
        
        Args:
            position_returns: Dictionary mapping ticker to return series
            
        Returns:
            Correlation matrix DataFrame
        """
        if len(position_returns) == 0:
            return pd.DataFrame()
        
        # Build returns DataFrame
        returns_df = pd.DataFrame(position_returns)
        
        # Calculate correlation
        corr_matrix = returns_df.corr()
        
        return corr_matrix
    
    def check_correlation_limits(self, position_returns: Dict[str, pd.Series]) -> bool:
        """
        Check if portfolio correlation is within acceptable limits
        
        Args:
            position_returns: Dictionary mapping ticker to return series
            
        Returns:
            True if within limits, False otherwise
        """
        if len(position_returns) < 2:
            return True
        
        corr_matrix = self.calculate_correlation_matrix(position_returns)
        
        # Calculate average pairwise correlation
        # Exclude diagonal (self-correlation = 1)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        avg_correlation = corr_matrix.values[mask].mean()
        
        if avg_correlation > risk_config.max_avg_correlation:
            print(f"⚠️  High correlation detected: {avg_correlation:.2f} "
                  f"(limit: {risk_config.max_avg_correlation:.2f})")
            return False
        
        return True
    
    def update_capital(self, new_capital: float):
        """Update capital and track peak for drawdown calculation"""
        self.current_capital = new_capital
        
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital
    
    def update_cash(self, amount: float):
        """Update cash position"""
        self.cash += amount
    
    def get_risk_metrics(self, returns: pd.Series) -> Dict:
        """
        Calculate comprehensive risk metrics
        
        Args:
            returns: Return series
            
        Returns:
            Dictionary of risk metrics
        """
        if len(returns) == 0:
            return {}
        
        metrics = {
            'current_capital': self.current_capital,
            'cash': self.cash,
            'peak_capital': self.peak_capital,
            'current_drawdown': (self.peak_capital - self.current_capital) / self.peak_capital,
            
            # Return metrics
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'avg_daily_return': returns.mean(),
            'daily_volatility': returns.std(),
            'annualized_volatility': returns.std() * np.sqrt(252),
            
            # Risk metrics
            'var_95': self.calculate_portfolio_var(returns, 0.95),
            'cvar_95': self.calculate_cvar(returns, 0.95),
            
            # Downside metrics
            'negative_returns': returns[returns < 0],
            'downside_deviation': returns[returns < 0].std() if len(returns[returns < 0]) > 0 else 0,
        }
        
        # Calculate Sharpe (if we have enough data)
        if len(returns) >= 30:
            risk_free_daily = (1 + analytics_config.risk_free_rate) ** (1/252) - 1
            excess_returns = returns - risk_free_daily
            sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
            metrics['sharpe_ratio'] = sharpe
            
            # Sortino
            downside_std = returns[returns < 0].std()
            sortino = excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
            metrics['sortino_ratio'] = sortino
        
        return metrics
    
    def should_stop_trading(self, returns: pd.Series) -> Tuple[bool, str]:
        """
        Check if trading should be stopped due to risk limits
        
        Args:
            returns: Recent return series
            
        Returns:
            Tuple of (should_stop, reason)
        """
        # Circuit breaker: Max drawdown
        current_dd = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_dd > risk_config.max_drawdown_pct:
            return True, f"Maximum drawdown reached: {current_dd*100:.2f}%"
        
        # Daily loss limit
        if len(returns) > 0:
            today_return = returns.iloc[-1]
            if today_return < -risk_config.max_daily_loss_pct:
                return True, f"Daily loss limit reached: {today_return*100:.2f}%"
        
        # Monthly loss limit (check last 21 trading days)
        if len(returns) >= 21:
            monthly_return = (1 + returns.iloc[-21:]).prod() - 1
            if monthly_return < -risk_config.max_monthly_loss_pct:
                return True, f"Monthly loss limit reached: {monthly_return*100:.2f}%"
        
        return False, ""
    
    def get_position_allocation(self, signals: List[Dict], 
                               current_positions: Dict) -> List[Dict]:
        """
        Allocate capital across multiple signals
        
        Args:
            signals: List of entry signals
            current_positions: Current positions
            
        Returns:
            List of signals with position_size added
        """
        # Calculate available slots
        available_slots = risk_config.max_positions - len(current_positions)
        
        if available_slots <= 0:
            return []
        
        # Take top N signals
        top_signals = signals[:available_slots]
        
        # Calculate available capital
        available_cash = self.cash * (1 - risk_config.min_cash_reserve)
        
        # Allocate equally with Kelly adjustment
        allocated_signals = []
        
        for signal in top_signals:
            ticker = signal['ticker']
            price = signal['entry_price']
            
            # Calculate position size
            shares = self.calculate_position_size(signal, current_positions, price)
            
            if shares > 0:
                position_value = shares * price
                
                # Check if we have enough cash
                if position_value <= available_cash:
                    signal['shares'] = shares
                    signal['position_value'] = position_value
                    allocated_signals.append(signal)
                    
                    # Update available cash
                    available_cash -= position_value
        
        return allocated_signals


# ============================================================================
# TESTING
# ============================================================================

def test_risk_manager():
    """Test risk manager functionality"""
    print("\n" + "="*80)
    print("TESTING RISK MANAGER")
    print("="*80 + "\n")
    
    rm = RiskManager(initial_capital=100_000)
    
    # Test 1: Position sizing
    print("1️⃣  Testing position sizing...")
    test_signal = {
        'ticker': 'AAPL',
        'entry_price': 150.0,
        'expected_return': 0.025,
        'sector': 'Technology'
    }
    
    shares = rm.calculate_position_size(test_signal, {}, 150.0)
    print(f"   Position size for AAPL @ $150: {shares} shares (${shares * 150:,.0f})")
    
    # Test 2: Risk metrics
    print("\n2️⃣  Testing risk metrics...")
    # Simulate some returns
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.015, 100))
    
    metrics = rm.get_risk_metrics(returns)
    print(f"   Daily Vol: {metrics['daily_volatility']*100:.2f}%")
    print(f"   Annual Vol: {metrics['annualized_volatility']*100:.2f}%")
    print(f"   VaR 95%: {metrics['var_95']*100:.2f}%")
    print(f"   Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
    
    # Test 3: Risk limits
    print("\n3️⃣  Testing risk limits...")
    within_limits, violations = rm.check_risk_limits({}, returns)
    if within_limits:
        print("   ✅ All risk limits satisfied")
    else:
        print(f"   ⚠️  Violations: {violations}")
    
    print("\n✅ Risk Manager tests completed")


if __name__ == '__main__':
    test_risk_manager()
