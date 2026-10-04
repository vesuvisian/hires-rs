#!/usr/bin/env python3
"""Serve `site/` with `application/wasm` (Zensical's preview server often does not).

Zensical bakes `site_url` into the build (GitHub Pages project path `/hires-rs/`).
This server mounts `site/` at that prefix so local URLs match production.
"""

from __future__ import annotations

import argparse
import mimetypes
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
ZENSICAL = ROOT / "zensical.toml"

mimetypes.add_type("application/wasm", ".wasm")


def pages_prefix() -> str:
    text = ZENSICAL.read_text(encoding="utf-8") if ZENSICAL.is_file() else ""
    match = re.search(r'^site_url\s*=\s*["\']([^"\']+)["\']', text, re.M)
    if not match:
        return ""
    path = urlparse(match.group(1)).path.rstrip("/")
    return path if path else ""


class Handler(SimpleHTTPRequestHandler):
    prefix = ""
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITE), **kwargs)

    def translate_path(self, path: str) -> str:
        prefix = self.prefix
        if prefix and (path == prefix or path.startswith(prefix + "/")):
            path = path[len(prefix) :] or "/"
        return super().translate_path(path)

    def do_GET(self) -> None:
        if self.prefix and self.path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", self.prefix + "/")
            self.end_headers()
            return
        super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not SITE.is_dir():
        raise SystemExit(
            f"{SITE} is missing. Run `zensical build` from the repo root first."
        )
    prefix = pages_prefix()
    Handler.prefix = prefix
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    origin = f"http://{args.host}:{args.port}"
    print(f"Serving {SITE} at {origin}{prefix or '/'}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
