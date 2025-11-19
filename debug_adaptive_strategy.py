"""
Debug script to trace why Adaptive strategy finds no trades
"""

from mock_data_manager import MockDataManager
from strategy_adaptive_div_mr import AdaptiveDividendMeanReversionStrategy
from datetime import timedelta
import pandas as pd

# Initialize
dm = MockDataManager()
strategy = AdaptiveDividendMeanReversionStrategy(dm)

# Test a single date
test_date = '2023-01-05'
print(f"\n{'='*80}")
print(f"DEBUGGING ADAPTIVE STRATEGY - {test_date}")
print(f"{'='*80}\n")

# Step 1: Get dividend calendar
print("Step 1: Fetching dividend calendar...")
current_dt = pd.to_datetime(test_date).tz_localize(None)
lookback = (current_dt - timedelta(days=5)).strftime('%Y-%m-%d')
lookahead = (current_dt + timedelta(days=15)).strftime('%Y-%m-%d')

div_calendar = dm.get_dividend_calendar(lookback, lookahead)
print(f"Found {len(div_calendar)} dividend events")
if len(div_calendar) > 0:
    print(div_calendar[['ticker', 'ex_dividend_date', 'dividend_amount']].head(10))

# Step 2: Process each dividend event
print(f"\n{'='*80}")
print("Step 2: Processing dividend events...")
print(f"{'='*80}\n")

for idx, event in div_calendar.head(5).iterrows():
    ticker = event['ticker']
    ex_div_date = pd.to_datetime(event.get('ex_dividend_date', event.get('ex_date'))).tz_localize(None)
    dividend_amount = event.get('dividend_amount', event.get('amount', 0))

    print(f"\n--- Processing {ticker} ---")
    print(f"Ex-div date: {ex_div_date}")
    print(f"Dividend: ${dividend_amount:.4f}")

    # Calculate days to ex-div
    days_to_ex_div = (ex_div_date - current_dt).days
    print(f"Days to ex-div: {days_to_ex_div}")

    # Check if in entry window
    pre_min, pre_max = strategy.params['pre_div_days']
    post_min, post_max = strategy.params['post_div_days']

    if pre_min <= days_to_ex_div <= pre_max:
        signal_type = 'pre_dividend'
        print(f"✓ In pre-dividend window ({pre_min} to {pre_max})")
    elif post_min <= days_to_ex_div <= post_max:
        signal_type = 'post_dividend'
        print(f"✓ In post-dividend window ({post_min} to {post_max})")
    else:
        print(f"✗ NOT in any entry window (pre: {pre_min} to {pre_max}, post: {post_min} to {post_max})")
        continue

    # Get current price
    try:
        current_price = dm.get_current_price(ticker, test_date)
        print(f"Current price: ${current_price:.2f}" if current_price else "✗ Price not available")
        if not current_price or current_price <= 0:
            continue
    except Exception as e:
        print(f"✗ Error getting price: {e}")
        continue

    # Calculate dividend yield
    div_yield = (dividend_amount / current_price) if current_price > 0 else 0
    print(f"Dividend yield: {div_yield*100:.2f}%")
    print(f"Min required: {strategy.params['min_div_yield']*100:.2f}%")

    if div_yield < strategy.params['min_div_yield']:
        print(f"✗ Dividend yield too low")
        continue
    else:
        print(f"✓ Dividend yield OK")

    # Get fundamentals
    try:
        fundamentals = dm.get_fundamentals(ticker)
        market_cap = fundamentals.get('market_cap', 0)
        avg_volume = fundamentals.get('avg_volume', 0)

        print(f"Market cap: ${market_cap/1e9:.2f}B (min: ${strategy.params['min_market_cap']/1e9:.2f}B)")
        print(f"Avg volume: {avg_volume:,.0f} (min: {strategy.params['min_volume']:,.0f})")

        if market_cap < strategy.params['min_market_cap']:
            print(f"✗ Market cap too small")
            continue
        if avg_volume < strategy.params['min_volume']:
            print(f"✗ Volume too low")
            continue

        print(f"✓ Fundamentals OK")
    except Exception as e:
        print(f"✗ Error getting fundamentals: {e}")
        continue

    # Get technical indicators
    try:
        indicators = dm.calculate_technical_indicators(ticker, test_date, lookback_days=60)

        if not indicators:
            print(f"✗ Could not calculate indicators")
            continue

        z_score = indicators.get('z_score', 0)
        rsi = indicators.get('rsi', 50)
        atr = indicators.get('atr', current_price * 0.02)
        current_volume = indicators.get('current_volume', 0)
        avg_volume_20d = indicators.get('avg_volume', 1)

        print(f"\nTechnical Indicators:")
        print(f"  Z-score: {z_score:.2f}")
        print(f"  RSI: {rsi:.2f}")
        print(f"  ATR: ${atr:.2f}")

        # Check signal-specific filters
        if signal_type == 'pre_dividend':
            print(f"\nPre-dividend filters:")
            print(f"  Z-score threshold: {strategy.params['pre_div_zscore']}")
            print(f"  RSI threshold: {strategy.params['pre_div_rsi']}")

            if z_score >= strategy.params['pre_div_zscore']:
                print(f"  ✗ Z-score not low enough ({z_score:.2f} >= {strategy.params['pre_div_zscore']})")
                continue
            else:
                print(f"  ✓ Z-score OK")

            if rsi >= strategy.params['pre_div_rsi']:
                print(f"  ✗ RSI not low enough ({rsi:.2f} >= {strategy.params['pre_div_rsi']})")
                continue
            else:
                print(f"  ✓ RSI OK")

            # Volume confirmation
            volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 0
            print(f"  Volume ratio: {volume_ratio:.2f}x (min: {strategy.params['volume_spike_threshold']}x)")

            if volume_ratio < strategy.params['volume_spike_threshold']:
                print(f"  ✗ Volume too low")
                continue
            else:
                print(f"  ✓ Volume OK")

        print(f"\n✅ {ticker} PASSES ALL FILTERS!")

    except Exception as e:
        print(f"✗ Error calculating indicators: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*80}")
print("Debug complete")
print(f"{'='*80}\n")
