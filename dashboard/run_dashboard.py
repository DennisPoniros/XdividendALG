#!/usr/bin/env python3
"""
Launcher script for the backtesting dashboard.

Usage:
    python dashboard/run_dashboard.py
    python dashboard/run_dashboard.py --port 8501
    python dashboard/run_dashboard.py --host 0.0.0.0 --port 8080
"""

import subprocess
import sys
import argparse
from pathlib import Path


def main():
    """Launch the Streamlit dashboard."""
    parser = argparse.ArgumentParser(description="Launch the backtesting dashboard")
    parser.add_argument(
        '--host',
        default='localhost',
        help='Host to bind to (default: localhost)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port to run on (default: 8501)'
    )
    parser.add_argument(
        '--browser',
        action='store_true',
        help='Automatically open browser'
    )

    args = parser.parse_args()

    # Get dashboard path
    dashboard_dir = Path(__file__).parent
    app_path = dashboard_dir / 'app.py'

    # Build streamlit command
    cmd = [
        'streamlit',
        'run',
        str(app_path),
        '--server.address', args.host,
        '--server.port', str(args.port),
        '--theme.base', 'dark',
    ]

    if not args.browser:
        cmd.extend(['--server.headless', 'true'])

    print(f"🚀 Launching dashboard at http://{args.host}:{args.port}")
    print(f"📊 Dashboard app: {app_path}")
    print("\nPress Ctrl+C to stop the server\n")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped")
        sys.exit(0)
    except FileNotFoundError:
        print("\n❌ Error: Streamlit not found!")
        print("Please install required packages:")
        print("  pip install -r dashboard/requirements.txt")
        sys.exit(1)


if __name__ == '__main__':
    main()
