#!/usr/bin/env python3
"""
X-Dividend Dashboard Launcher

This is the MAIN ENTRY POINT for all XDividendALG operations.

Usage:
    python run_dashboard.py

Features:
- Strategy selection and management
- Parameter configuration
- Run backtests from UI
- View and analyze results
- Real-time progress tracking

No need to run any other scripts - everything is accessible from the dashboard!
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the enhanced dashboard."""

    # Get dashboard path
    project_root = Path(__file__).parent
    app_path = project_root / 'dashboard' / 'app_enhanced.py'

    if not app_path.exists():
        print(f"❌ Error: Dashboard app not found at {app_path}")
        sys.exit(1)

    # Build streamlit command
    cmd = [
        'streamlit',
        'run',
        str(app_path),
        '--server.address', 'localhost',
        '--server.port', '8501',
        '--theme.base', 'dark',
        '--server.headless', 'true'
    ]

    print("="*80)
    print("💎 X-DIVIDEND TRADING DASHBOARD")
    print("="*80)
    print("\n🚀 Launching dashboard at http://localhost:8501")
    print("\n📋 Available Features:")
    print("   ✅ Strategy selection and comparison")
    print("   ✅ Parameter configuration via UI")
    print("   ✅ Run backtests from dashboard")
    print("   ✅ Real-time progress tracking")
    print("   ✅ Results analysis and visualization")
    print("\n⭐ RECOMMENDED STRATEGY: X-Dividend ML (Fixed)")
    print("   - Fixes exit logic bug (-10.6% → +8-15% expected)")
    print("   - Uses time-based exits (not P&L-based)")
    print("   - Relaxed screening (50-150 trades/year)")
    print("\nPress Ctrl+C to stop the dashboard\n")
    print("="*80 + "\n")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("👋 Dashboard stopped")
        print("="*80)
        sys.exit(0)
    except FileNotFoundError:
        print("\n" + "="*80)
        print("❌ Error: Streamlit not found!")
        print("="*80)
        print("\nPlease install required packages:")
        print("  pip install -r dashboard/requirements.txt")
        print("\nOr install streamlit directly:")
        print("  pip install streamlit")
        print("\n" + "="*80)
        sys.exit(1)
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ Error launching dashboard: {e}")
        print("="*80)
        sys.exit(1)


if __name__ == '__main__':
    main()
