"""
Strategy Module for Dividend Capture Algorithm
Generates entry/exit signals and manages positions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from config import (
    entry_config, exit_config, risk_config, screening_config,
    TECHNICAL_ANALYSIS_CONSTANTS, DIVIDEND_STRATEGY_CONSTANTS
)
from data_manager import DataManager


class DividendCaptureStrategy:
    """
    Implements dividend capture with mean reversion logic
    """
    
    def __init__(self, data_manager: DataManager):
        self.dm = data_manager
        self.positions = {}  # Currently held positions
        self.pending_orders = {}  # Orders waiting to fill
        self.position_history = []  # Closed positions
        
    def generate_entry_signals(self, screened_stocks: pd.DataFrame, 
                               current_date: str) -> List[Dict]:
        """
        Generate entry signals from screened stocks
        
        Args:
            screened_stocks: DataFrame from screening process
            current_date: Current date (YYYY-MM-DD)
            
        Returns:
            List of entry signals with ticker, score, and entry price
        """
        if len(screened_stocks) == 0:
            return []
        
        signals = []
        current_dt = pd.to_datetime(current_date)
        
        print(f"\n🎯 Generating entry signals for {len(screened_stocks)} candidates...")
        
        for idx, row in screened_stocks.iterrows():
            ticker = row['ticker']

            # Get recent price data for technical analysis
            lookback_days = TECHNICAL_ANALYSIS_CONSTANTS['LOOKBACK_DAYS_DEFAULT']
            lookback_start = (current_dt - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            prices = self.dm.get_stock_prices(ticker, lookback_start, current_date)

            min_history = TECHNICAL_ANALYSIS_CONSTANTS['MIN_PRICE_HISTORY_DAYS']
            if len(prices) < min_history:  # Need sufficient history
                continue
            
            # Calculate technical indicators
            prices = self.dm.calculate_technical_indicators(prices)
            
            # Apply entry filters
            if not self._passes_entry_filters(prices, row):
                continue
            
            # Calculate entry price
            current_price = prices['close'].iloc[-1]
            
            # Estimate mean reversion parameters
            mr_params = self.dm.estimate_mean_reversion_params(prices['close'])
            
            # Calculate expected return
            expected_return = self._calculate_expected_return(
                current_price, row, mr_params, prices
            )
            
            # Build signal
            signal = {
                'ticker': ticker,
                'entry_date': current_date,
                'entry_price': current_price,
                'ex_div_date': row['ex_date'],
                'days_to_ex_div': row['days_to_ex_div'],
                'dividend_amount': row['amount'],
                'dividend_yield': row['yield'],
                'quality_score': row['quality_score'],
                'expected_return': expected_return,
                'sector': row.get('sector', 'Unknown'),
                
                # Technical context
                'rsi': prices['rsi_14'].iloc[-1],
                'z_score': prices['z_score'].iloc[-1],
                'momentum_20': prices['momentum_20'].iloc[-1],
                'volatility_20': prices['volatility_20'].iloc[-1],
                
                # Mean reversion
                'mr_theta': mr_params['theta'],
                'mr_mu': mr_params['mu'],
                'mr_sigma': mr_params['sigma'],
                
                # Stop loss
                'stop_loss': self._calculate_stop_loss(current_price, row['amount']),
            }
            
            signals.append(signal)
        
        # Sort by expected return
        signals = sorted(signals, key=lambda x: x['expected_return'], reverse=True)
        
        print(f"✅ Generated {len(signals)} entry signals")
        
        return signals
    
    def _passes_entry_filters(self, prices: pd.DataFrame, stock_info: pd.Series) -> bool:
        """
        Check if stock passes entry filters
        """
        if len(prices) < 20:
            return False

        latest = prices.iloc[-1]

        # RSI filter (always apply basic RSI check from screening config)
        rsi = latest['rsi_14']
        if pd.isna(rsi) or rsi < screening_config.min_rsi or rsi > screening_config.max_rsi:
            return False

        # Z-score filter (mean reversion signal)
        if entry_config.use_z_score_filter:
            z_score = latest['z_score']
            if pd.isna(z_score):
                return False
            if z_score < entry_config.z_score_min or z_score > entry_config.z_score_max:
                return False
        
        # Momentum filter
        if entry_config.require_positive_momentum:
            momentum = latest['momentum_20']
            if pd.isna(momentum) or momentum <= 0:
                return False
        
        # Volatility filter
        if entry_config.use_volatility_filter:
            volatility = latest['volatility_20']
            if pd.isna(volatility) or volatility > entry_config.max_realized_vol:
                return False
        
        # Days to ex-div must be in preferred window
        if stock_info['days_to_ex_div'] not in entry_config.preferred_entry_days:
            return False
        
        return True
    
    def _calculate_expected_return(self, price: float, stock_info: pd.Series,
                                   mr_params: Dict, prices: pd.DataFrame) -> float:
        """
        Calculate expected return from position
        Combines dividend capture + mean reversion
        """
        # Validate price
        if price <= 0:
            return 0

        dividend_amount = stock_info['amount']

        # Component 1: Dividend capture (historical inefficiency)
        # Capture the inefficiency portion of dividend drop
        alpha_capture = DIVIDEND_STRATEGY_CONSTANTS['DIVIDEND_CAPTURE_ALPHA']
        dividend_return = (dividend_amount * alpha_capture) / price

        # Component 2: Mean reversion potential
        # Distance from long-term mean (standardized)
        mu = mr_params['mu']
        sigma = mr_params['sigma']

        if sigma > 0 and mu > 0 and price > 0:
            # Calculate z-score consistently: (current - mean) / std
            # Since OU params are in log space, work in log space
            log_price = np.log(price)
            log_mu = np.log(mu)
            current_z = (log_price - log_mu) / sigma if sigma > 0 else 0
        else:
            current_z = 0

        # If trading below mean (negative z), expect reversion upward (positive return)
        # If trading above mean (positive z), expect reversion downward (negative return)
        mr_sensitivity = DIVIDEND_STRATEGY_CONSTANTS['MEAN_REVERSION_SENSITIVITY']
        mean_reversion_return = -current_z * mr_sensitivity

        # Component 3: Momentum continuation (small factor)
        momentum = prices['momentum_20'].iloc[-1]
        if not pd.isna(momentum):
            momentum_factor = DIVIDEND_STRATEGY_CONSTANTS['MOMENTUM_CONTINUATION_FACTOR']
            momentum_return = momentum * momentum_factor
        else:
            momentum_return = 0
        
        # Total expected return
        total_return = dividend_return + mean_reversion_return + momentum_return
        
        return total_return
    
    def _calculate_stop_loss(self, entry_price: float, dividend_amount: float) -> float:
        """Calculate stop loss price"""

        # Validate input
        if entry_price <= 0:
            return 0

        # Hard stop percentage
        hard_stop = entry_price * (1 - exit_config.hard_stop_pct)

        # Dividend-based stop (1x dividend)
        if exit_config.use_dividend_stop:
            div_stop = entry_price - dividend_amount
            return max(hard_stop, div_stop)  # Tighter of the two

        return hard_stop
    
    def check_exit_signals(self, current_date: str) -> List[Dict]:
        """
        Check all open positions for exit signals
        
        Args:
            current_date: Current date (YYYY-MM-DD)
            
        Returns:
            List of exit signals
        """
        exit_signals = []
        current_dt = pd.to_datetime(current_date)
        
        for ticker, position in self.positions.items():
            
            # Get current price
            prices = self.dm.get_stock_prices(
                ticker,
                (current_dt - timedelta(days=30)).strftime('%Y-%m-%d'),
                current_date
            )
            
            if len(prices) == 0:
                continue
            
            current_price = prices['close'].iloc[-1]

            # Calculate P&L
            entry_price = position['entry_price']
            if entry_price > 0:
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = 0
            
            # Days held
            entry_dt = pd.to_datetime(position['entry_date'])
            days_held = (current_dt - entry_dt).days
            
            # Days since ex-dividend
            ex_div_dt = pd.to_datetime(position['ex_div_date'])
            days_since_ex_div = (current_dt - ex_div_dt).days
            
            # Check exit conditions
            exit_reason = None
            
            # 1. Stop loss
            if current_price <= position['stop_loss']:
                exit_reason = 'stop_loss'
            
            # 2. Profit target
            elif pnl_pct >= exit_config.profit_target_absolute:
                exit_reason = 'profit_target_absolute'
            
            elif pnl_pct >= position['dividend_yield'] * exit_config.profit_target_multiple:
                exit_reason = 'profit_target_dividend_multiple'
            
            # 3. Time-based exits
            elif days_held >= exit_config.max_holding_days:
                exit_reason = 'max_holding_period'
            
            # 4. Post-dividend logic (only after ex-div date)
            elif current_dt >= ex_div_dt:
                
                # Calculate price drop on ex-div
                if days_since_ex_div == 0:
                    # Check price drop on ex-div date
                    pre_ex_price = entry_price
                    price_drop_pct = (pre_ex_price - current_price) / pre_ex_price
                    div_drop_expected = position['dividend_amount'] / pre_ex_price
                    
                    if price_drop_pct < exit_config.dividend_adjustment_threshold_low * div_drop_expected:
                        # Price didn't drop much, hold for mean reversion
                        pass
                    elif price_drop_pct > exit_config.dividend_adjustment_threshold_high * div_drop_expected:
                        # Price dropped too much, exit immediately
                        exit_reason = 'excessive_div_drop'
                
                # Mean reversion exit
                if days_since_ex_div >= exit_config.min_holding_days:
                    # Check if price recovered
                    if exit_config.use_entry_plus_dividend:
                        if current_price >= entry_price + position['dividend_amount']:
                            exit_reason = 'mean_reversion_complete'
                    
                    # VWAP cross
                    if exit_config.use_vwap_cross and len(prices) >= exit_config.vwap_period:
                        prices_with_ind = self.dm.calculate_technical_indicators(prices)
                        vwap = prices_with_ind['vwap'].iloc[-1]
                        if current_price >= vwap:
                            exit_reason = 'vwap_cross'
            
            # 5. Trailing stop (if activated)
            if exit_config.trailing_stop_enabled and 'trailing_stop' in position:
                if current_price <= position['trailing_stop']:
                    exit_reason = 'trailing_stop'
            
            # Update trailing stop if in profit
            if exit_config.trailing_stop_enabled and pnl_pct >= exit_config.trailing_stop_activation:
                new_trailing_stop = current_price * (1 - exit_config.trailing_stop_distance)
                if 'trailing_stop' not in position or new_trailing_stop > position['trailing_stop']:
                    position['trailing_stop'] = new_trailing_stop
            
            # Generate exit signal if reason found
            if exit_reason:
                exit_signals.append({
                    'ticker': ticker,
                    'exit_date': current_date,
                    'exit_price': current_price,
                    'exit_reason': exit_reason,
                    'days_held': days_held,
                    'pnl_pct': pnl_pct,
                    **position
                })
        
        return exit_signals
    
    def open_position(self, signal: Dict) -> bool:
        """
        Open a new position
        
        Args:
            signal: Entry signal dictionary
            
        Returns:
            True if position opened successfully
        """
        ticker = signal['ticker']
        
        if ticker in self.positions:
            print(f"⚠️  Position already exists for {ticker}")
            return False
        
        # Add position
        self.positions[ticker] = signal
        
        print(f"📈 OPENED: {ticker} @ ${signal['entry_price']:.2f} "
              f"(Div: ${signal['dividend_amount']:.2f}, Score: {signal['quality_score']:.1f})")
        
        return True
    
    def close_position(self, exit_signal: Dict) -> Dict:
        """
        Close an existing position
        
        Args:
            exit_signal: Exit signal dictionary
            
        Returns:
            Closed position dictionary with P&L
        """
        ticker = exit_signal['ticker']
        
        if ticker not in self.positions:
            print(f"⚠️  No position found for {ticker}")
            return None
        
        # Calculate final P&L
        entry_price = exit_signal['entry_price']
        exit_price = exit_signal['exit_price']

        if entry_price > 0:
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = 0

        pnl_amount = (exit_price - entry_price)
        
        # Build closed position record
        closed_position = {
            **exit_signal,
            'pnl_pct': pnl_pct,
            'pnl_amount_per_share': pnl_amount,
        }
        
        # Add to history
        self.position_history.append(closed_position)
        
        # Remove from active positions
        del self.positions[ticker]
        
        print(f"📉 CLOSED: {ticker} @ ${exit_price:.2f} "
              f"(P&L: {pnl_pct*100:+.2f}%, Reason: {exit_signal['exit_reason']})")
        
        return closed_position
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio statistics"""
        
        num_positions = len(self.positions)
        
        if num_positions == 0:
            return {
                'num_positions': 0,
                'total_exposure': 0,
                'sectors': {},
                'avg_quality_score': 0,
                'avg_days_to_ex_div': 0
            }
        
        # Sector breakdown
        sectors = {}
        total_quality = 0
        total_days_to_ex_div = 0
        
        for ticker, pos in self.positions.items():
            sector = pos.get('sector', 'Unknown')
            sectors[sector] = sectors.get(sector, 0) + 1
            total_quality += pos['quality_score']
            total_days_to_ex_div += pos['days_to_ex_div']
        
        return {
            'num_positions': num_positions,
            'sectors': sectors,
            'avg_quality_score': total_quality / num_positions,
            'avg_days_to_ex_div': total_days_to_ex_div / num_positions
        }
    
    def get_trade_statistics(self) -> Dict:
        """Get statistics from closed trades"""
        
        if len(self.position_history) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'avg_pnl': 0,
                'profit_factor': 0,
                'avg_holding_days': 0
            }
        
        trades = pd.DataFrame(self.position_history)
        
        # Calculate statistics
        total_trades = len(trades)
        wins = trades[trades['pnl_pct'] > 0]
        losses = trades[trades['pnl_pct'] <= 0]
        
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0
        avg_pnl = trades['pnl_pct'].mean()
        
        # Profit factor
        total_wins = wins['pnl_pct'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        avg_holding_days = trades['days_held'].mean()
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_pnl': avg_pnl,
            'profit_factor': profit_factor,
            'avg_holding_days': avg_holding_days
        }


# ============================================================================
# TESTING
# ============================================================================

def test_strategy():
    """Test strategy functionality"""
    print("\n" + "="*80)
    print("TESTING STRATEGY MODULE")
    print("="*80 + "\n")
    
    dm = DataManager()
    strategy = DividendCaptureStrategy(dm)
    
    # Get some dividend candidates
    div_cal = dm.get_dividend_calendar('2024-01-01', '2024-03-31')
    
    if len(div_cal) > 0:
        # Screen stocks for 2024-01-15
        screened = dm.screen_stocks(div_cal, '2024-01-15')
        
        if len(screened) > 0:
            print(f"\n✅ Screened {len(screened)} stocks")
            
            # Generate entry signals
            signals = strategy.generate_entry_signals(screened, '2024-01-15')
            
            if len(signals) > 0:
                print(f"\n✅ Generated {len(signals)} entry signals")
                print("\nTop 3 signals:")
                for i, sig in enumerate(signals[:3]):
                    print(f"   {i+1}. {sig['ticker']}: "
                          f"Expected Return: {sig['expected_return']*100:.2f}%, "
                          f"Quality: {sig['quality_score']:.1f}")
    
    print("\n✅ Strategy tests completed")


if __name__ == '__main__':
    test_strategy()
