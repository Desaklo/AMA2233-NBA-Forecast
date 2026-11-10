#!/usr/bin/env python3
"""Serve the forecast website from web/."""

from __future__ import annotations

import argparse
import http.server
import os
import socketserver
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "web"


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the NBA forecast website.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    os.chdir(ROOT)

    class Handler(http.server.SimpleHTTPRequestHandler):
        extensions_map = {
            **http.server.SimpleHTTPRequestHandler.extensions_map,
            ".js": "application/javascript",
            ".json": "application/json",
            ".css": "text/css",
        }

    with socketserver.ThreadingTCPServer((args.host, args.port), Handler) as httpd:
        httpd.allow_reuse_address = True
        url = f"http://{args.host}:{args.port}/"
        print(f"Serving {ROOT}")
        print(f"Open {url}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
