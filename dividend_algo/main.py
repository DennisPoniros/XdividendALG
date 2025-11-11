"""
Main execution script for Dividend Capture Trading Algorithm
Runs complete backtesting pipeline with analytics
"""

import sys
import warnings
warnings.filterwarnings('ignore')

from config import (
    data_config, backtest_config, analytics_config,
    validate_config, print_config_summary
)
from backtester import Backtester, run_train_test_split
from analytics import create_performance_report


def main():
    """
    Main execution function
    """
    
    print("\n" + "="*80)
    print("🎯 DIVIDEND CAPTURE ALGORITHM - COMPREHENSIVE BACKTEST")
    print("="*80 + "\n")
    
    # Step 1: Validate configuration
    print("1️⃣  Validating configuration...")
    validation = validate_config()

    if validation['errors']:
        print("❌ Configuration errors found:")
        for error in validation['errors']:
            print(f"   - {error}")
        return

    if validation['warnings']:
        print("\n⚠️  Configuration warnings:")
        for warning in validation['warnings']:
            print(f"   - {warning}")
        print("\nNote: Warnings won't prevent execution, but some features may not work.")

    print("\n✅ Configuration validated\n")
    print_config_summary()
    
    # Step 2: Choose backtest mode
    print("\n" + "="*80)
    print("SELECT BACKTEST MODE")
    print("="*80)
    print("1. Full Period Backtest (single run)")
    print("2. Train/Test Split (separate train & test periods)")
    print("3. Walk-Forward Analysis (rolling validation)")
    print("4. Quick Test (2024 only)")
    print("="*80)
    
    try:
        choice = input("\nEnter choice (1-4) [default: 2]: ").strip() or "2"
    except:
        choice = "2"  # Default for automated runs
    
    # Step 3: Run backtest based on choice
    if choice == "1":
        # Full period backtest
        print("\n📊 Running full period backtest...")
        bt = Backtester(
            data_config.train_start,
            data_config.test_end,
            backtest_config.initial_capital
        )
        results = bt.run_backtest(mode='full')
        
        if results:
            # Generate analytics
            plots, html_report = create_performance_report(results)
            
            # Export trade log
            bt.export_results('/mnt/user-data/outputs/trade_log.csv')
    
    elif choice == "2":
        # Train/test split
        print("\n📊 Running train/test split backtest...")
        train_results, test_results = run_train_test_split()
        
        if test_results:
            # Generate analytics for test period (out-of-sample)
            plots, html_report = create_performance_report(
                test_results,
                output_dir='/mnt/user-data/outputs'
            )
            
            print(f"\n✅ Test period results:")
            print(f"   - HTML Report: {html_report}")
    
    elif choice == "3":
        # Walk-forward analysis
        print("\n📊 Running walk-forward analysis...")
        bt = Backtester(
            data_config.train_start,
            data_config.test_end,
            backtest_config.initial_capital
        )
        wf_results = bt.run_backtest(mode='walk_forward')
        
        if wf_results:
            print("\n✅ Walk-forward analysis complete!")
            print(f"   Average Sharpe: {wf_results['avg_sharpe']:.2f}")
            print(f"   Consistency: {wf_results['consistency']*100:.1f}%")
    
    elif choice == "4":
        # Quick test - 2024 only
        print("\n📊 Running quick test (2024)...")
        bt = Backtester(
            '2024-01-01',
            '2024-10-31',
            backtest_config.initial_capital
        )
        results = bt.run_backtest(mode='full')
        
        if results:
            plots, html_report = create_performance_report(results)
            bt.export_results('/mnt/user-data/outputs/trade_log_2024.csv')
    
    else:
        print("❌ Invalid choice. Exiting.")
        return
    
    print("\n" + "="*80)
    print("🎉 BACKTEST PIPELINE COMPLETED")
    print("="*80)
    print("\n📁 Output files located in: /mnt/user-data/outputs/")
    print("   - Plots: equity_curve.png, drawdown.png, monthly_returns.png, etc.")
    print("   - Report: backtest_report.html")
    print("   - Trade log: trade_log.csv")
    print("\n" + "="*80 + "\n")


def quick_validation_run():
    """
    Quick validation run for testing the system
    Runs a short backtest to verify everything works
    """
    
    print("\n" + "="*80)
    print("🧪 QUICK VALIDATION RUN")
    print("="*80 + "\n")
    
    # Run short backtest
    bt = Backtester(
        '2024-09-01',
        '2024-10-31',
        50_000  # Smaller capital for quick test
    )
    
    results = bt.run_backtest(mode='full')
    
    if results and results.get('total_trades', 0) > 0:
        print("\n✅ VALIDATION PASSED - System is working correctly!")
        print(f"   - Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"   - Total Trades: {results['total_trades']}")
        print(f"   - Win Rate: {results['win_rate_pct']:.1f}%")
        return True
    else:
        print("\n⚠️  VALIDATION WARNING - No trades generated")
        print("   This might be normal if no dividend events occurred in test period")
        return False


def parameter_sweep():
    """
    Run parameter sweep to find optimal settings
    """
    
    print("\n" + "="*80)
    print("🔬 PARAMETER SWEEP ANALYSIS")
    print("="*80 + "\n")
    
    from config import entry_config, exit_config
    
    # Parameters to test
    entry_windows = [[3], [4], [5], [3, 4], [4, 5], [3, 4, 5]]
    profit_targets = [1.2, 1.5, 2.0]
    
    results_summary = []
    
    for entry_window in entry_windows:
        for profit_target in profit_targets:
            
            # Update config
            entry_config.preferred_entry_days = entry_window
            exit_config.profit_target_multiple = profit_target
            
            print(f"\n📊 Testing: Entry={entry_window}, Profit Target={profit_target}x div")
            
            # Run backtest
            bt = Backtester(
                '2023-01-01',
                '2024-10-31',
                backtest_config.initial_capital
            )
            
            results = bt.run_backtest(mode='full')
            
            if results:
                results_summary.append({
                    'entry_window': str(entry_window),
                    'profit_target': profit_target,
                    'sharpe': results['sharpe_ratio'],
                    'annual_return': results['annual_return_pct'],
                    'win_rate': results['win_rate_pct'],
                    'max_dd': results['max_drawdown_pct'],
                    'total_trades': results['total_trades']
                })
    
    # Display results
    if results_summary:
        import pandas as pd
        df = pd.DataFrame(results_summary)
        df = df.sort_values('sharpe', ascending=False)
        
        print("\n" + "="*80)
        print("🏆 PARAMETER SWEEP RESULTS (Sorted by Sharpe)")
        print("="*80 + "\n")
        print(df.to_string(index=False))
        print("\n" + "="*80 + "\n")
        
        # Save results
        import os
        output_dir = '/mnt/user-data/outputs'
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(f'{output_dir}/parameter_sweep.csv', index=False)
        print(f"✅ Results saved to: {output_dir}/parameter_sweep.csv")


if __name__ == '__main__':
    
    # Check if running in interactive mode
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == 'validate':
            # Quick validation
            success = quick_validation_run()
            sys.exit(0 if success else 1)
        
        elif mode == 'sweep':
            # Parameter sweep
            parameter_sweep()
        
        elif mode == 'test':
            # Quick test mode
            print("Running quick test mode...")
            bt = Backtester('2024-01-01', '2024-10-31', 100_000)
            results = bt.run_backtest(mode='full')
            
            if results:
                plots, html_report = create_performance_report(results)
                print(f"\n✅ Quick test complete. Report: {html_report}")
        
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: validate, sweep, test")
    
    else:
        # Run main interactive mode
        main()
