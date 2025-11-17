"""
Data Manager for Dividend Capture Algorithm
Handles data fetching from Alpaca (prices) and yfinance (fundamentals/dividends)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from logger import get_logger

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("⚠️  alpaca-trade-api not installed. Install with: pip install alpaca-trade-api")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️  yfinance not installed. Install with: pip install yfinance")

from config import (
    ALPACA_CONFIG, data_config, screening_config, 
    scoring_config, mean_reversion_config
)


class DataManager:
    """
    Unified data manager for fetching and processing market data
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.alpaca_api = None
        self.data_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_call_count = 0
        self._initialize_alpaca()
    
    def _initialize_alpaca(self):
        """Initialize Alpaca API connection"""
        if not ALPACA_AVAILABLE:
            self.logger.warning("Alpaca API not available - package not installed")
            return

        try:
            self.alpaca_api = tradeapi.REST(
                key_id=ALPACA_CONFIG['API_KEY'],
                secret_key=ALPACA_CONFIG['SECRET_KEY'],
                base_url=ALPACA_CONFIG['BASE_URL']
            )
            # Test connection
            account = self.alpaca_api.get_account()
            self.logger.info(f"Alpaca connected - Account Status: {account.status}")
        except Exception as e:
            self.logger.error(f"Alpaca connection failed: {e}")
            self.logger.warning("Please check your API credentials in config.py")
            self.alpaca_api = None
    
    def get_dividend_calendar(self, start_date: str, end_date: str, 
                             lookback_days: int = 30) -> pd.DataFrame:
        """
        Get dividend calendar with ex-dividend dates
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            lookback_days: Days to look ahead for upcoming dividends
            
        Returns:
            DataFrame with columns: [ticker, ex_date, pay_date, amount, yield]
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance required for dividend data")
        
        self.logger.info(f"Fetching dividend calendar: {start_date} to {end_date}")
        
        # Get list of potential dividend-paying stocks
        tickers = self._get_dividend_stock_universe()
        
        dividend_calendar = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                
                # Get dividend history
                dividends = stock.dividends
                if len(dividends) == 0:
                    continue
                
                # Get actions (includes ex-dates)
                actions = stock.actions
                
                # Filter to date range
                mask = (actions.index >= start_date) & (actions.index <= end_date)
                relevant_divs = actions[mask]
                
                if len(relevant_divs) > 0:
                    for date, row in relevant_divs.iterrows():
                        if row.get('Dividends', 0) > 0:
                            # Calculate ANNUALIZED yield using trailing 12-month dividends
                            hist = stock.history(start=date - timedelta(days=5),
                                               end=date)
                            if len(hist) > 0:
                                price = hist['Close'].iloc[-1]
                                # Validate price before calculating yield
                                if price > 0:
                                    # Get trailing 12-month dividends for accurate annual yield
                                    try:
                                        ttm_start = date - timedelta(days=365)
                                        ttm_dividends = dividends[(dividends.index >= ttm_start) & (dividends.index <= date)]
                                        annual_dividend = ttm_dividends.sum() if len(ttm_dividends) > 0 else row['Dividends'] * 4  # Fallback: assume quarterly
                                        div_yield = annual_dividend / price
                                    except:
                                        # Fallback: assume quarterly payments (most common)
                                        div_yield = (row['Dividends'] * 4) / price
                                else:
                                    div_yield = 0
                            else:
                                div_yield = 0

                            dividend_calendar.append({
                                'ticker': ticker,
                                'ex_date': date,
                                'amount': row['Dividends'],
                                'yield': div_yield  # Now properly annualized
                            })
            
            except Exception as e:
                # Silently skip problematic tickers
                continue
        
        df = pd.DataFrame(dividend_calendar)

        if len(df) > 0:
            # Convert ex_date to timezone-naive to avoid comparison issues
            df['ex_date'] = pd.to_datetime(df['ex_date']).dt.tz_localize(None)
            df = df.sort_values('ex_date').reset_index(drop=True)

            # Count events per ticker
            events_per_ticker = df.groupby('ticker').size()
            multi_event_tickers = events_per_ticker[events_per_ticker > 1]

            self.logger.info(f"Found {len(df)} dividend events across {df['ticker'].nunique()} stocks")
            if len(multi_event_tickers) > 0:
                self.logger.info(f"  {len(multi_event_tickers)} stocks with multiple events: {dict(multi_event_tickers)}")
        else:
            self.logger.warning("No dividend events found in specified date range")

        return df
    
    def _get_dividend_stock_universe(self) -> List[str]:
        """
        Get initial universe of dividend-paying stocks
        Using a curated list of high-quality dividend stocks
        """
        # S&P 500 Dividend Aristocrats and common dividend payers
        dividend_universe = [
            # Dividend Aristocrats (consistent 25+ year history)
            'JNJ', 'PG', 'KO', 'PEP', 'WMT', 'MCD', 'MMM', 'CAT', 'XOM', 'CVX',
            'IBM', 'VZ', 'T', 'CSCO', 'INTC', 'GIS', 'KMB', 'CL', 'CLX', 'SYY',
            'ABT', 'ABBV', 'BMY', 'LLY', 'MRK', 'PFE', 'MDT', 'UNH', 'TMO',
            
            # High-quality dividend payers
            'AAPL', 'MSFT', 'JPM', 'BAC', 'WFC', 'C', 'USB', 'PNC', 'TFC',
            'HD', 'LOW', 'TGT', 'COST', 'SO', 'DUK', 'NEE', 'D', 'AEP',
            'MO', 'PM', 'BTI', 'UPS', 'FDX', 'LMT', 'RTX', 'BA', 'GD',
            
            # REITs
            'O', 'SPG', 'VICI', 'AMT', 'PSA', 'DLR', 'WELL',
            
            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL',
            
            # Financials
            'JPM', 'BAC', 'C', 'WFC', 'BLK', 'MS', 'GS', 'SCHW',
            
            # Consumer Staples
            'PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS', 'K'
        ]
        
        # Remove duplicates
        return list(set(dividend_universe))
    
    def get_stock_prices(self, ticker: str, start_date: str,
                        end_date: str, timeframe: str = '1Day') -> pd.DataFrame:
        """
        Get historical price data for a ticker
        Implements caching to reduce API calls

        Args:
            ticker: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: '1Day', '1Hour', '5Min', etc.

        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        cache_key = f"{ticker}_{start_date}_{end_date}_{timeframe}"
        if cache_key in self.data_cache:
            self.cache_hits += 1
            self.logger.debug(f"Cache hit for {ticker} ({start_date} to {end_date})")
            return self.data_cache[cache_key].copy()

        self.cache_misses += 1
        self.api_call_count += 1

        # Try Alpaca first, fallback to yfinance
        if self.alpaca_api:
            try:
                barset = self.alpaca_api.get_bars(
                    ticker,
                    timeframe,
                    start=start_date,
                    end=end_date,
                    adjustment='all'
                ).df

                if len(barset) > 0:
                    # Normalize timezone to avoid comparison issues
                    if barset.index.tz is not None:
                        barset.index = barset.index.tz_localize(None)
                    self.data_cache[cache_key] = barset.copy()
                    self.logger.debug(f"Fetched {len(barset)} bars for {ticker} from Alpaca")
                    return barset
            except Exception as e:
                self.logger.warning(f"Alpaca fetch failed for {ticker}: {e}")

        # Fallback to yfinance
        if YFINANCE_AVAILABLE:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
                if len(df) > 0:
                    df.columns = [c.lower() for c in df.columns]
                    # Normalize timezone to avoid comparison issues
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    self.data_cache[cache_key] = df.copy()
                    self.logger.debug(f"Fetched {len(df)} bars for {ticker} from yfinance")
                    return df
                else:
                    self.logger.warning(f"No data returned for {ticker}")
            except Exception as e:
                self.logger.error(f"yfinance fetch failed for {ticker}: {e}")

        self.logger.error(f"Failed to fetch data for {ticker} from all sources")
        return pd.DataFrame()
    
    def get_fundamentals(self, ticker: str) -> Dict:
        """
        Get fundamental data for quality scoring
        
        Returns:
            Dictionary with fundamental metrics
        """
        if not YFINANCE_AVAILABLE:
            return {}
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            fundamentals = {
                'market_cap': info.get('marketCap', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                
                # Dividend metrics
                'dividend_yield': info.get('dividendYield', 0),
                'payout_ratio': info.get('payoutRatio', 0),
                'dividend_rate': info.get('dividendRate', 0),
                'ex_dividend_date': info.get('exDividendDate', None),
                
                # Financial health
                'debt_to_equity': info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else None,
                'roe': info.get('returnOnEquity', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                
                # Growth
                'earnings_growth': info.get('earningsGrowth', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                
                # Volume
                'avg_volume': info.get('averageVolume', 0),
                'volume': info.get('volume', 0),
                
                # Beta
                'beta': info.get('beta', 1.0),
                
                # Short interest
                'short_ratio': info.get('shortRatio', 0),
                'short_percent': info.get('shortPercentOfFloat', 0),
            }
            
            # Get dividend history for growth rate
            dividends = stock.dividends
            if len(dividends) >= 12:  # At least 3 years of quarterly dividends
                recent_divs = dividends[-4:].sum()  # Last year
                old_divs = dividends[-16:-12].sum()  # 3 years ago
                if old_divs > 0:
                    cagr = (recent_divs / old_divs) ** (1/3) - 1
                    fundamentals['dividend_growth_3y'] = cagr
                else:
                    fundamentals['dividend_growth_3y'] = 0
            else:
                fundamentals['dividend_growth_3y'] = 0
            
            return fundamentals
            
        except Exception as e:
            print(f"⚠️  Failed to get fundamentals for {ticker}: {e}")
            return {}
    
    def calculate_technical_indicators(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for strategy
        
        Args:
            prices: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        df = prices.copy()
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Volatility
        df['returns'] = df['close'].pct_change()
        df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], period=14)

        # Z-Score (for mean reversion) - uses price std, not returns std
        price_std = df['close'].rolling(window=20).std()
        df['z_score'] = (df['close'] - df['sma_20']) / price_std.where(price_std > 0, np.nan)
        
        # VWAP
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Momentum
        df['momentum_20'] = df['close'].pct_change(periods=20)
        
        # Bollinger Bands
        std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['sma_20'] + (2 * std_20)
        df['bb_lower'] = df['sma_20'] - (2 * std_20)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def estimate_mean_reversion_params(self, prices: pd.Series) -> Dict:
        """
        Estimate Ornstein-Uhlenbeck process parameters
        
        Args:
            prices: Price series
            
        Returns:
            Dictionary with theta (mean reversion speed), mu (long-term mean), sigma
        """
        # Log prices for OU estimation
        log_prices = np.log(prices.dropna())
        
        if len(log_prices) < mean_reversion_config.ou_estimation_window:
            return {'theta': 0, 'mu': prices.mean(), 'sigma': 0}
        
        # AR(1) regression: dP_t = alpha + beta*P_{t-1} + e_t
        # theta = -beta, mu = -alpha/beta
        
        y = log_prices.diff().dropna()
        X = log_prices.shift(1).dropna()
        
        # Align
        y = y[X.index]
        
        if len(y) < 30:
            return {'theta': 0, 'mu': prices.mean(), 'sigma': 0}
        
        # OLS regression
        X_mean = X.mean()
        y_mean = y.mean()
        
        beta = ((X - X_mean) * (y - y_mean)).sum() / ((X - X_mean) ** 2).sum()
        alpha = y_mean - beta * X_mean
        
        theta = -beta
        mu = -alpha / beta if beta != 0 else log_prices.mean()
        
        # Estimate sigma from residuals
        residuals = y - (alpha + beta * X)
        sigma = residuals.std()
        
        return {
            'theta': max(mean_reversion_config.min_mean_reversion_speed,
                        min(theta, mean_reversion_config.max_mean_reversion_speed)),
            'mu': np.exp(mu),  # Convert back from log
            'sigma': sigma
        }
    
    def get_batch_prices(self, tickers: List[str], start_date: str, 
                        end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Get prices for multiple tickers efficiently
        
        Args:
            tickers: List of tickers
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        prices = {}
        
        print(f"📊 Fetching prices for {len(tickers)} tickers...")
        
        # Use yfinance download for batch efficiency
        if YFINANCE_AVAILABLE:
            try:
                data = yf.download(tickers, start=start_date, end=end_date, 
                                 group_by='ticker', auto_adjust=True, 
                                 progress=False, threads=True)
                
                for ticker in tickers:
                    try:
                        if len(tickers) == 1:
                            ticker_data = data
                        else:
                            ticker_data = data[ticker]
                        
                        if not ticker_data.empty:
                            ticker_data.columns = [c.lower() for c in ticker_data.columns]
                            prices[ticker] = ticker_data
                    except:
                        continue
                
                print(f"✅ Successfully fetched {len(prices)}/{len(tickers)} tickers")
                
            except Exception as e:
                print(f"⚠️  Batch download failed: {e}")
        
        return prices
    
    def screen_stocks(self, dividend_calendar: pd.DataFrame, 
                     current_date: str) -> pd.DataFrame:
        """
        Apply screening criteria to dividend calendar
        
        Args:
            dividend_calendar: DataFrame with dividend events
            current_date: Current date for screening (YYYY-MM-DD)
            
        Returns:
            Filtered DataFrame with quality scores
        """
        # Ensure current_date is timezone-naive for consistent date math
        current_dt = pd.to_datetime(current_date).tz_localize(None)

        # Calculate days to ex-dividend (ex_date already normalized to tz-naive in get_dividend_calendar)
        dividend_calendar['days_to_ex_div'] = (
            dividend_calendar['ex_date'] - current_dt
        ).dt.days
        
        # Filter by entry window
        mask = (
            (dividend_calendar['days_to_ex_div'] >= screening_config.min_days_to_ex_div) &
            (dividend_calendar['days_to_ex_div'] <= screening_config.max_days_to_ex_div)
        )
        
        candidates = dividend_calendar[mask].copy()
        
        if len(candidates) == 0:
            return pd.DataFrame()
        
        print(f"🔍 Screening {len(candidates)} dividend candidates...")

        # Get fundamentals for each candidate
        screened_stocks = []
        failed_reasons = {}

        for idx, row in candidates.iterrows():
            ticker = row['ticker']

            # Get fundamentals
            fundamentals = self.get_fundamentals(ticker)

            if not fundamentals:
                failed_reasons[ticker] = "No fundamentals data"
                continue

            # Apply filters (returns tuple: passed, reason)
            passed, reason = self._passes_screening_filters(fundamentals, row)
            if not passed:
                failed_reasons[ticker] = reason
                continue

            # Calculate quality score
            quality_score = self._calculate_quality_score(fundamentals, row)

            if quality_score >= screening_config.min_quality_score:
                screened_stocks.append({
                    **row,
                    'quality_score': quality_score,
                    **fundamentals
                })
            else:
                failed_reasons[ticker] = f"Quality score {quality_score:.1f} < {screening_config.min_quality_score}"

        result = pd.DataFrame(screened_stocks)

        if len(result) > 0:
            result = result.sort_values('quality_score', ascending=False)
            print(f"✅ {len(result)} stocks passed screening")
        else:
            print("⚠️  No stocks passed screening criteria")
            # Show why stocks failed
            if failed_reasons:
                print("\n📋 Screening failures:")
                for ticker, reason in list(failed_reasons.items())[:10]:  # Show first 10
                    print(f"   {ticker}: {reason}")
        
        return result
    
    def _passes_screening_filters(self, fundamentals: Dict, div_info: pd.Series) -> tuple:
        """
        Check if stock passes all screening filters

        Returns:
            tuple: (passed: bool, reason: str)
        """

        # Market cap
        market_cap = fundamentals.get('market_cap', 0)
        if market_cap < screening_config.min_market_cap:
            return False, f"Market cap ${market_cap/1e9:.2f}B < ${screening_config.min_market_cap/1e9:.1f}B"

        # Volume
        volume = fundamentals.get('avg_volume', 0)
        if volume < screening_config.min_avg_volume:
            return False, f"Volume {volume:,} < {screening_config.min_avg_volume:,}"

        # Dividend yield
        div_yield = div_info.get('yield', 0)
        if div_yield < screening_config.min_dividend_yield:
            return False, f"Div yield {div_yield*100:.2f}% < {screening_config.min_dividend_yield*100:.1f}%"
        if div_yield > screening_config.max_dividend_yield:
            return False, f"Div yield {div_yield*100:.2f}% > {screening_config.max_dividend_yield*100:.1f}%"

        # Payout ratio
        payout = fundamentals.get('payout_ratio', 0)
        if payout > screening_config.max_acceptable_payout:
            return False, f"Payout ratio {payout*100:.1f}% > {screening_config.max_acceptable_payout*100:.1f}%"

        # Financial health
        debt_to_equity = fundamentals.get('debt_to_equity')
        if debt_to_equity and debt_to_equity > screening_config.max_debt_to_equity:
            return False, f"Debt/Equity {debt_to_equity:.2f} > {screening_config.max_debt_to_equity:.2f}"

        roe = fundamentals.get('roe', 0)
        if roe < screening_config.min_roe:
            return False, f"ROE {roe*100:.1f}% < {screening_config.min_roe*100:.1f}%"
        
        pe = fundamentals.get('pe_ratio', 0)
        if pe > screening_config.max_pe_ratio and pe > 0:
            return False, f"P/E {pe:.1f} > {screening_config.max_pe_ratio:.1f}"

        # Beta
        beta = fundamentals.get('beta', 1.0)
        if beta > screening_config.max_beta:
            return False, f"Beta {beta:.2f} > {screening_config.max_beta:.2f}"

        return True, "Passed all filters"
    
    def _calculate_quality_score(self, fundamentals: Dict, div_info: pd.Series) -> float:
        """
        Calculate quality score (0-100) based on multiple factors
        """
        scores = {}
        
        # 1. Payout ratio score (25%)
        payout = fundamentals.get('payout_ratio') or 0
        if screening_config.optimal_payout_min <= payout <= screening_config.optimal_payout_max:
            payout_score = 100
        else:
            # Penalty for being outside optimal range
            distance = min(
                abs(payout - screening_config.optimal_payout_min),
                abs(payout - screening_config.optimal_payout_max)
            )
            payout_score = max(0, 100 - distance * 200)  # 50% penalty per 0.1 distance

        scores['payout'] = payout_score * scoring_config.payout_weight

        # 2. Growth score (25%)
        div_growth = fundamentals.get('dividend_growth_3y') or 0
        growth_score = min(100, max(0, div_growth * 1000))  # 10% growth = 100 points
        scores['growth'] = growth_score * scoring_config.growth_weight
        
        # 3. Financial health score (25%)
        # Sub-scores (handle None values explicitly)
        debt_to_equity = fundamentals.get('debt_to_equity') or 0.5
        debt_score = max(0, 100 - (debt_to_equity / screening_config.max_debt_to_equity) * 100)

        roe = fundamentals.get('roe') or 0
        roe_score = min(100, (roe / screening_config.min_roe) * 100) if screening_config.min_roe > 0 else 0

        pe = fundamentals.get('pe_ratio') or 15
        if pe > 0:
            pe_score = max(0, 100 - ((pe - 15) / screening_config.max_pe_ratio) * 100)
        else:
            pe_score = 50
        
        financial_score = (
            debt_score * scoring_config.debt_score_weight +
            roe_score * scoring_config.roe_score_weight +
            pe_score * scoring_config.pe_score_weight
        )
        scores['financial'] = financial_score * scoring_config.financial_weight
        
        # 4. Technical score (25%)
        beta = fundamentals.get('beta') or 1.0
        beta_score = max(0, 100 - ((beta - 0.5) / screening_config.max_beta) * 100)

        short_interest = fundamentals.get('short_percent') or 0
        short_score = max(0, 100 - (short_interest / screening_config.max_short_interest) * 100) if screening_config.max_short_interest > 0 else 100
        
        technical_score = (beta_score + short_score) / 2
        scores['technical'] = technical_score * scoring_config.technical_weight
        
        # Total score
        total_score = sum(scores.values())

        return round(total_score, 2)

    def get_cache_stats(self) -> Dict:
        """Get cache statistics for monitoring API efficiency"""
        total_calls = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_calls * 100) if total_calls > 0 else 0

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_requests': total_calls,
            'hit_rate_pct': round(hit_rate, 2),
            'api_calls': self.api_call_count,
            'cached_items': len(self.data_cache)
        }

    def clear_cache(self):
        """Clear data cache to free memory"""
        self.data_cache.clear()
        self.logger.info("Data cache cleared")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def test_data_manager():
    """Test data manager functionality"""
    print("\n" + "="*80)
    print("TESTING DATA MANAGER")
    print("="*80 + "\n")
    
    dm = DataManager()
    
    # Test 1: Dividend calendar
    print("\n1️⃣  Testing dividend calendar...")
    div_cal = dm.get_dividend_calendar('2024-01-01', '2024-03-31')
    if len(div_cal) > 0:
        print(div_cal.head())
    
    # Test 2: Stock prices
    print("\n2️⃣  Testing price fetching...")
    prices = dm.get_stock_prices('AAPL', '2024-01-01', '2024-03-31')
    if len(prices) > 0:
        print(f"   Fetched {len(prices)} price bars for AAPL")
        print(prices.head())
    
    # Test 3: Fundamentals
    print("\n3️⃣  Testing fundamentals...")
    fundamentals = dm.get_fundamentals('AAPL')
    print(f"   Market Cap: ${fundamentals.get('market_cap', 0)/1e9:.2f}B")
    print(f"   Dividend Yield: {fundamentals.get('dividend_yield', 0)*100:.2f}%")
    print(f"   P/E Ratio: {fundamentals.get('pe_ratio', 0):.2f}")
    
    # Test 4: Technical indicators
    if len(prices) > 0:
        print("\n4️⃣  Testing technical indicators...")
        prices_with_indicators = dm.calculate_technical_indicators(prices)
        print(f"   RSI (latest): {prices_with_indicators['rsi_14'].iloc[-1]:.2f}")
        print(f"   Z-Score: {prices_with_indicators['z_score'].iloc[-1]:.2f}")
    
    print("\n✅ Data Manager tests completed")


if __name__ == '__main__':
    test_data_manager()
