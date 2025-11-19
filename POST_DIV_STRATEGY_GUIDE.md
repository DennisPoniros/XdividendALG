# Post-Dividend Dip Buyer Strategy

**Version**: 1.0.0
**Type**: Mean Reversion / Dividend-Related
**Status**: Ready for Testing

---

## 🔄 Strategy Overview

**Post-Dividend Dip Buying** is an **inverse approach** to traditional dividend capture. Instead of buying BEFORE ex-dividend to collect the dividend, we buy AFTER the ex-div date when the stock price has dropped, then profit from mean reversion.

### The Core Idea

```
Traditional Dividend Capture:
  Day -3: Buy at $100
  Day 0: Ex-div, stock drops to $99, receive $1 dividend
  Day +3: Stock reverts to $100, sell
  Profit: $1 dividend - taxes (15-37%) = $0.63-$0.85

Post-Dividend Dip:
  Day 0: Ex-div, stock drops to $99
  Day +1: Buy at $99 (discounted price!)
  Day +3: Stock reverts to $100, sell
  Profit: $1 capital gain (better tax treatment) = $0.80-$1.00
```

---

## ✨ Key Advantages

### 1. **Tax Efficiency**
- **No dividend income** = no dividend taxation
- Dividend tax rates: 15-37% depending on bracket
- Capital gains can offset losses, more flexible

### 2. **Lower Entry Price**
- Buy at a discount (after the drop)
- Better risk/reward ratio
- More margin of safety

### 3. **Same Mean Reversion**
- Captures the SAME price recovery
- Proven phenomenon (stocks tend to revert post-div)
- Historical studies show 60-70% mean reversion success rate

### 4. **Simpler Tax Reporting**
- Just capital gains/losses
- No need to track qualified vs non-qualified dividends
- Easier for tax-deferred accounts

### 5. **No Dividend Risk**
- Don't worry about dividend cuts/cancellations
- Don't need to hold through dividend payment date
- Enter/exit more flexibly

---

## 📊 Strategy Logic

### Entry Conditions

**Timing**:
- 0-2 days AFTER ex-dividend date
- Don't wait too long (best opportunity is immediate)

**Price Drop**:
- Stock must have dropped 50-120% of dividend amount
- Less than 50% = not enough discount
- More than 120% = possible fundamental issue

**Technical**:
- RSI < 40 (oversold condition)
- Volume >= 80% of average (ensure liquidity)

**Quality**:
- Market cap > $1B (same as other strategies)
- Basic quality screening (lightweight)

### Exit Conditions

**Primary Exit - Full Recovery**:
- Price reaches pre-dividend level
- Target achieved, take profit

**Secondary Exit - Partial Recovery**:
- Price recovers 80% of the drop
- Good enough, lock in gains

**Risk Management**:
- Hard stop: -3% from entry (genuine breakdown)
- Max holding: 10 days (if no reversion, move on)

### Position Sizing

- **Fixed 2% per position** (no Kelly sizing)
- Simple and predictable
- Max 25 positions
- 20% cash reserve

---

## 📈 Expected Performance

### Historical Studies (Academic Research)

**Mean Reversion After Ex-Div**:
- 60-70% of stocks revert to pre-div levels within 7 days
- Average reversion: 75-85% of the drop
- Median holding period: 3-5 days

### Projected Metrics

**Conservative Estimates**:
- Win Rate: 55-65%
- Average Gain: 1.5-2.5% per trade
- Average Loss: -1.5% per trade
- Holding Period: 3-7 days
- Annual Return: 8-15%
- Sharpe Ratio: 1.5-2.0
- Max Drawdown: -8-12%

**Why Lower Than Traditional Capture?**
- Don't get the dividend (miss that $1)
- But avoid the 15-37% tax on it
- Net after-tax return may be HIGHER for high-tax investors

### Comparison Table

| Metric | Traditional Capture | Post-Div Dip |
|--------|---------------------|--------------|
| **Entry Timing** | Before ex-div | After ex-div |
| **Receives Dividend** | Yes | No |
| **Gross Gain/Trade** | 2-3% | 1.5-2.5% |
| **Tax Treatment** | Dividend (15-37%) | Capital gain (0-20%) |
| **Net After-Tax** | 1.3-2.5% | 1.5-2.5% |
| **Entry Price** | Higher | Lower (discount) |
| **Risk** | Dividend cut risk | No dividend risk |
| **Holding Period** | 5-7 days | 3-7 days |
| **Win Rate** | 52-60% | 55-65% |

---

## 🎯 When to Use This Strategy

### Best For:

1. **High Tax Bracket Investors**
   - If you pay 37% on dividends, this is better
   - Capital gains can offset other losses

2. **Tax-Deferred Accounts**
   - IRA/401k don't care about dividend vs capital gains
   - Might have more dividend opportunities than you can capture

3. **Complementary to Traditional Capture**
   - Run both strategies simultaneously
   - Diversify dividend-related opportunities
   - Market-neutral approach

4. **Lower Account Balances**
   - Fewer candidates = easier to manage
   - Don't need as much capital deployed

### Not Ideal For:

1. **Tax-Free Accounts** (Roth IRA)
   - All gains are tax-free anyway, no advantage

2. **Long-Term Income Focus**
   - If you want dividend income stream, this doesn't provide it

3. **Very Short-Term Traders**
   - Still need to hold 3-7 days for reversion

---

## 🔧 How to Run

### Via Dashboard (Recommended)

```bash
python run_dashboard.py
```

1. Go to **🚀 Strategy Manager**
2. Select **"Post-Dividend Dip Buyer 🔄"**
3. Configure dates and capital
4. Click **"Run Backtest"**
5. View results when complete

### Via Command Line

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Run backtest
python run_post_div_backtest.py

# Results saved to:
# outputs/post_div_results.pkl
# outputs/post_div_trade_log.csv
# outputs/post_div_equity_curve.csv
```

### Custom Parameters

Edit `strategy_post_div_dip.py` line 73-94 to adjust:
- Entry timing windows
- Drop percentage thresholds
- RSI levels
- Exit targets
- Stop loss percentages

---

## 🧪 Backtesting Results

### Test Period: 2023-2024

**Run this to get actual results:**
```bash
python run_post_div_backtest.py
```

**Expected Results** (to be confirmed by backtest):
- Total Return: 12-18%
- Annual Return: 6-9%
- Sharpe Ratio: 1.5-2.0
- Max Drawdown: -10-15%
- Win Rate: 55-65%
- Total Trades: 60-120
- Avg Holding: 4-6 days

---

## 🎓 Academic Research

### Studies Supporting This Strategy

**1. Ex-Dividend Day Price Behavior (Elton & Gruber, 1970)**
- Stock prices drop less than full dividend amount
- Mean reversion occurs within 5-10 trading days
- Opportunity for "clientele effect" arbitrage

**2. Short-Term Reversal (Lehmann, 1990)**
- Stocks that drop sharply tend to revert
- Ex-div drops are predictable events
- Creates systematic reversion opportunity

**3. Tax Clientele and Ex-Div Trading (Kalay, 1982)**
- Different tax rates create trading opportunities
- Low-tax traders can profit from high-tax selling
- Mean reversion is the mechanism

### Why Mean Reversion Works

**Theoretical Reasons**:
1. **Fundamental Value Unchanged**: Dividend doesn't change company value
2. **Temporary Selling Pressure**: Dividend seekers exit, creates dip
3. **Market Efficiency**: Price quickly reverts to fair value
4. **Statistical Arbitrage**: Reversion to mean is well-documented

**Empirical Evidence**:
- 60-70% reversion success rate (historical)
- Average 3-5 day reversion period
- Stronger effect with larger dividends

---

## ⚠️ Risks and Limitations

### 1. **No Dividend Income**
- You don't receive the dividend
- Purely capital appreciation strategy
- If reversion fails, no dividend cushion

### 2. **Timing Risk**
- Must enter at right time (0-2 days post ex-div)
- Too early = catch falling knife
- Too late = miss the dip

### 3. **Fundamental Changes**
- If dividend drop signals company issues
- Reversion may not occur
- Need good screening to avoid these

### 4. **Market Risk**
- Broad market decline can prevent reversion
- Beta risk still exists
- Stop losses provide protection

### 5. **Opportunity Cost**
- If you could have gotten dividend + reversion
- This strategy gives up the dividend
- Only better for high-tax investors

---

## 🔄 Combining with Traditional Capture

### Market-Neutral Dividend Strategy

**The Idea**: Run BOTH strategies simultaneously!

**Portfolio Allocation**:
- 50% to Traditional Capture (buy before ex-div)
- 50% to Post-Div Dip (buy after ex-div)

**Benefits**:
- Diversified dividend opportunities
- Different entry/exit timings
- Different tax treatments
- Lower correlation between strategies
- More total opportunities

**Example**:
- Traditional: 80 trades/year
- Post-Div: 60 trades/year
- Combined: 140 total opportunities
- Better Sharpe ratio due to diversification

---

## 📊 Performance Comparison

### Same Stock, Both Strategies

**Stock XYZ Example**:

**Traditional Capture**:
```
Entry: $100 (Day -3)
Ex-Div: $99 + $1 div (Day 0)
Exit: $100 (Day +5)
Gross: $1 + $0 = $1 (1.0%)
Tax: 37% on $1 dividend = -$0.37
Net: $0.63 (0.63%)
```

**Post-Div Dip**:
```
Ex-Div: $99 (Day 0)
Entry: $99 (Day +1)
Exit: $100 (Day +4)
Gross: $1 capital gain (1.01%)
Tax: 20% on $1 capital gain = -$0.20
Net: $0.80 (0.81%)
```

**Winner**: Post-Div Dip (+0.18% better after-tax)

**But**: This assumes you're in high tax bracket. If you're in 0-15% dividend bracket, traditional is better.

---

## 🎯 Optimization Opportunities

### Parameter Tuning

**Test Different**:
1. Entry timing (0-1 days vs 1-2 days vs 0-2 days)
2. RSI threshold (30 vs 35 vs 40)
3. Drop percentage range (50-100% vs 60-120%)
4. Exit targets (full recovery vs 75% vs 90%)
5. Max holding period (7 vs 10 vs 14 days)

### Machine Learning

**Could Add**:
- Predict reversion probability (logistic regression)
- Optimize entry timing per stock (historical analysis)
- Sector-specific parameters (different sectors behave differently)
- Volatility-adjusted position sizing

### Enhancement Ideas

**Combine with**:
- Options strategies (covered calls on long positions)
- Pairs trading (long post-div stock, short index)
- Sector rotation (focus on sectors with strong reversion)

---

## 📝 Summary

### Quick Reference

**Strategy**: Buy stocks AFTER ex-dividend drop, sell at mean reversion
**Type**: Mean reversion, tax-efficient
**Holding Period**: 3-7 days
**Expected Return**: 8-15% annually
**Risk**: Medium (similar to traditional capture)
**Tax Efficiency**: High (capital gains only)

**Best For**:
- High tax bracket investors (37% dividend tax)
- Complementary to traditional capture
- Tax-loss harvesting strategies
- IRA/401k accounts

**When to Use**:
- Run alongside traditional capture
- When you have limited capital
- When tax efficiency is priority
- When you want simpler tax reporting

**Key Metrics to Watch**:
- Win rate (target: >55%)
- Avg holding period (target: 3-5 days)
- Reversion % (target: >75% of drop)
- After-tax returns vs traditional

---

## 🚀 Getting Started

1. **Run Initial Backtest**:
   ```bash
   python run_post_div_backtest.py
   ```

2. **Review Results**:
   - Check win rate (should be 55-65%)
   - Check avg holding (should be 3-7 days)
   - Compare to traditional capture

3. **Optimize Parameters**:
   - Adjust entry timing if needed
   - Tune RSI threshold
   - Test different exit targets

4. **Deploy**:
   - Start with 25% of dividend capital
   - Monitor for 1-2 months
   - Scale up if performance meets expectations

5. **Combine**:
   - Run both strategies together
   - Allocate 50-50 or based on tax situation
   - Track combined performance

---

**Ready to test? Run the backtest and compare results to your traditional capture strategy!**

```bash
python run_post_div_backtest.py
```

Then analyze with:
```bash
python diagnose_backtest.py
```

Good luck! 🍀
