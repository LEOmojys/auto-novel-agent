#!/usr/bin/env python3
"""Agent web console entry point.

Usage:
    python webui.py                # http://127.0.0.1:7860
    python webui.py --port 8000
"""

from __future__ import annotations

import argparse

from novel_agent.webui.server import launch_console


def main() -> None:
    parser = argparse.ArgumentParser(description="Novel Agent web console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    launch_console(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
