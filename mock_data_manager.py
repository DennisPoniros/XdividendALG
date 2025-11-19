"""
Mock Data Manager for Testing Strategy Without External Data
Generates synthetic but realistic dividend and price data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class MockDataManager:
    """
    Generates realistic synthetic dividend and price data for backtesting
    """

    def __init__(self, seed=42):
        """Initialize with random seed for reproducibility."""
        np.random.seed(seed)

        # Universe of dividend-paying stocks
        self.tickers = [
            # High quality dividend stocks
            'AAPL', 'MSFT', 'JPM', 'JNJ', 'PG', 'KO', 'PEP', 'WMT', 'VZ', 'T',
            'XOM', 'CVX', 'BAC', 'WFC', 'GS', 'USB', 'PNC', 'TFC', 'C', 'MS',
            'HD', 'LOW', 'TGT', 'COST', 'MCD', 'NKE', 'SBUX', 'DIS', 'CMCSA', 'NFLX',
            'ABT', 'LLY', 'UNH', 'ABBV', 'BMY', 'MRK', 'PFE', 'AMGN', 'GILD', 'CVS',
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'PCG', 'ED', 'ES', 'FE',
        ]

        # Sector mapping
        self.sectors = {
            'AAPL': 'Technology', 'MSFT': 'Technology',
            'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials',
            'JNJ': 'Healthcare', 'ABT': 'Healthcare', 'LLY': 'Healthcare', 'UNH': 'Healthcare',
            'PG': 'Consumer', 'KO': 'Consumer', 'PEP': 'Consumer', 'WMT': 'Consumer',
            'XOM': 'Energy', 'CVX': 'Energy',
            'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities',
        }

        # Cache for generated data
        self.price_cache = {}
        self.dividend_cache = None

    def get_dividend_calendar(self, start_date: str, end_date: str, lookback_days: int = 30) -> pd.DataFrame:
        """
        Generate synthetic dividend calendar

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            lookback_days: Not used in mock

        Returns:
            DataFrame with dividend events
        """

        if self.dividend_cache is not None:
            # Filter cached data
            df = self.dividend_cache
            df = df[(df['ex_dividend_date'] >= start_date) & (df['ex_dividend_date'] <= end_date)]
            return df.reset_index(drop=True)

        # Generate dividend calendar for full backtest period
        events = []

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Each stock pays quarterly dividends
        for ticker in self.tickers:
            # Random start month (0-2 for quarterly offset)
            offset_months = np.random.randint(0, 3)

            # Generate quarterly dividends
            current_date = start_dt + timedelta(days=offset_months * 30)

            while current_date <= end_dt:
                # Dividend amount based on ticker characteristics
                base_yield = np.random.uniform(0.02, 0.05)  # 2-5% annual
                quarterly_yield = base_yield / 4

                # Ex-dividend date
                ex_date = current_date.strftime('%Y-%m-%d')

                events.append({
                    'ticker': ticker,
                    'ex_dividend_date': ex_date,
                    'ex_date': ex_date,  # Alternative field name
                    'dividend_amount': quarterly_yield * 100,  # Assume $100 stock price
                    'amount': quarterly_yield * 100,
                    'sector': self.sectors.get(ticker, 'Unknown'),
                })

                # Next quarter
                current_date += timedelta(days=90)

        # Create DataFrame
        df = pd.DataFrame(events)

        # Cache it
        self.dividend_cache = df.copy()

        # Filter to requested range
        df = df[(df['ex_dividend_date'] >= start_date) & (df['ex_dividend_date'] <= end_date)]

        return df.reset_index(drop=True)

    def get_stock_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate synthetic price data with realistic mean reversion

        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """

        cache_key = f"{ticker}_{start_date}_{end_date}"

        if cache_key in self.price_cache:
            return self.price_cache[cache_key].copy()

        # Generate business days
        dates = pd.date_range(start=start_date, end=end_date, freq='B')

        if len(dates) == 0:
            return pd.DataFrame()

        # Stock characteristics
        base_price = np.random.uniform(50, 200)
        annual_vol = np.random.uniform(0.15, 0.35)
        daily_vol = annual_vol / np.sqrt(252)

        # Mean reversion parameters
        mean_reversion_speed = np.random.uniform(0.05, 0.15)  # theta
        long_term_mean = base_price

        # Generate price series with mean reversion (Ornstein-Uhlenbeck)
        prices = [base_price]

        for i in range(1, len(dates)):
            # Mean reversion: price pulled toward long-term mean
            drift = mean_reversion_speed * (long_term_mean - prices[-1])

            # Random shock
            shock = np.random.normal(0, daily_vol * prices[-1])

            # New price
            new_price = prices[-1] + drift + shock

            # Ensure price stays positive
            new_price = max(new_price, 1.0)

            prices.append(new_price)

        # Create OHLCV data
        prices = np.array(prices)

        # High/Low with realistic spread
        highs = prices * (1 + np.abs(np.random.normal(0, 0.005, len(prices))))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.005, len(prices))))

        # Volume
        avg_volume = np.random.uniform(500_000, 5_000_000)
        volumes = np.random.uniform(0.7, 1.3, len(prices)) * avg_volume

        # Create DataFrame
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, len(prices))),
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes,
        }, index=dates)

        # Cache it
        self.price_cache[cache_key] = df.copy()

        return df

    def calculate_technical_indicators(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators

        Args:
            prices: DataFrame with OHLC data

        Returns:
            DataFrame with added technical indicators
        """

        df = prices.copy()

        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], period=14)

        # Z-score (mean reversion signal)
        df['z_score'] = self._calculate_zscore(df['close'], period=20)

        # Volatility
        df['returns'] = df['close'].pct_change()
        df['volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(252)

        # Momentum
        df['momentum_20'] = df['close'].pct_change(20)

        # VWAP (simplified)
        df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)

        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_zscore(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate z-score for mean reversion."""
        rolling_mean = prices.rolling(period).mean()
        rolling_std = prices.rolling(period).std()

        zscore = (prices - rolling_mean) / rolling_std

        return zscore

    def estimate_mean_reversion_params(self, prices: pd.Series) -> Dict:
        """
        Estimate mean reversion parameters

        Args:
            prices: Price series

        Returns:
            Dictionary with theta, mu, sigma
        """

        if len(prices) < 20:
            return {'theta': 0.1, 'mu': prices.mean(), 'sigma': 0.2}

        # Simple estimates
        mu = prices.mean()
        sigma = prices.std() / mu if mu > 0 else 0.2

        # Estimate theta from autocorrelation
        returns = prices.pct_change().dropna()
        if len(returns) > 1:
            acf_1 = returns.autocorr(lag=1)
            theta = -np.log(abs(acf_1)) if abs(acf_1) > 0 else 0.1
            theta = np.clip(theta, 0.05, 0.5)
        else:
            theta = 0.1

        return {
            'theta': theta,
            'mu': mu,
            'sigma': sigma,
        }

    def screen_stocks(self, dividend_calendar: pd.DataFrame, current_date: str) -> pd.DataFrame:
        """
        Mock screening - just adds required fields

        Args:
            dividend_calendar: Dividend events
            current_date: Current date

        Returns:
            Screened stocks with required fields
        """

        if len(dividend_calendar) == 0:
            return pd.DataFrame()

        current_dt = pd.to_datetime(current_date)

        # Calculate days to ex-div
        dividend_calendar = dividend_calendar.copy()
        dividend_calendar['ex_date'] = pd.to_datetime(dividend_calendar['ex_dividend_date'])
        dividend_calendar['days_to_ex_div'] = (dividend_calendar['ex_date'] - current_dt).dt.days

        # Filter to upcoming dividends
        upcoming = dividend_calendar[
            (dividend_calendar['days_to_ex_div'] >= -5) &
            (dividend_calendar['days_to_ex_div'] <= 20)
        ].copy()

        if len(upcoming) == 0:
            return pd.DataFrame()

        # Add mock quality score
        upcoming['quality_score'] = np.random.uniform(60, 90, len(upcoming))

        # Add yield (mock)
        upcoming['yield'] = upcoming['amount'] / 100  # Assume $100 stock

        return upcoming

    def get_fundamentals(self, ticker: str) -> Optional[Dict]:
        """
        Mock fundamentals

        Args:
            ticker: Stock ticker

        Returns:
            Dictionary with fundamental data
        """

        return {
            'market_cap': np.random.uniform(5e9, 500e9),
            'pe_ratio': np.random.uniform(10, 30),
            'dividend_yield': np.random.uniform(0.02, 0.05),
            'payout_ratio': np.random.uniform(0.3, 0.7),
            'roe': np.random.uniform(0.08, 0.20),
            'debt_to_equity': np.random.uniform(0.2, 0.8),
        }


if __name__ == '__main__':
    # Test mock data
    print("="*80)
    print("MOCK DATA MANAGER TEST")
    print("="*80)

    dm = MockDataManager()

    # Test dividend calendar
    div_cal = dm.get_dividend_calendar('2023-01-01', '2023-03-31')
    print(f"\nGenerated {len(div_cal)} dividend events")
    print(div_cal.head(10))

    # Test price data
    prices = dm.get_stock_prices('AAPL', '2023-01-01', '2023-01-31')
    print(f"\nGenerated {len(prices)} days of price data for AAPL")
    print(prices.head())

    # Test technicals
    prices_with_ind = dm.calculate_technical_indicators(prices)
    print(f"\nAdded technical indicators")
    print(prices_with_ind[['close', 'rsi_14', 'z_score', 'volatility_20']].tail())

    print("\n" + "="*80)
    print("✅ Mock data generation working correctly")
    print("="*80)
