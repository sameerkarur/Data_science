#!/usr/bin/env python3
"""Resolve every anchor in the built MkDocs site against the output tree.

Catches the two failure modes readers actually hit: links that 404, and links
that leave the site to render raw file contents (raw.githubusercontent.com,
github.com/blob, htmlpreview).
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urldefrag, urljoin

SITE = Path(sys.argv[1]).resolve()

RAW_HOSTS = ("raw.githubusercontent.com", "htmlpreview.github.io")


class Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = dict(attrs)
        if d.get("id"):
            self.ids.add(d["id"])
        if d.get("name"):
            self.ids.add(d["name"])
        if tag == "a" and d.get("href"):
            self.hrefs.append(d["href"])


pages = sorted(SITE.rglob("*.html"))
parsed: dict[Path, Collector] = {}
for page in pages:
    c = Collector()
    c.feed(page.read_text(encoding="utf-8", errors="replace"))
    parsed[page] = c

problems: dict[str, list[str]] = defaultdict(list)
external = 0
checked = 0

for page, coll in parsed.items():
    rel_page = page.relative_to(SITE)
    # URL of this page, used as the base for resolving relative hrefs.
    base = "/" + str(rel_page).replace("\\", "/")

    for href in coll.hrefs:
        if href.startswith(("mailto:", "javascript:", "tel:", "data:")):
            continue

        if any(host in href for host in RAW_HOSTS) or re.search(
            r"github\.com/[^/]+/[^/]+/blob/", href
        ):
            problems["raw-file link (renders as source code)"].append(
                f"{rel_page} -> {href}"
            )
            continue

        if href.startswith(("http://", "https://", "//")):
            external += 1
            continue

        target, frag = urldefrag(href)
        checked += 1

        if not target:
            # Pure in-page anchor.
            if frag and frag not in coll.ids:
                problems["dead in-page anchor"].append(f"{rel_page} -> #{frag}")
            continue

        resolved = urljoin(base, target)
        path = SITE / unquote(resolved.lstrip("/"))
        if resolved.endswith("/"):
            path = path / "index.html"

        if not path.exists():
            problems["broken link (404)"].append(f"{rel_page} -> {href}")
            continue

        if frag and path.suffix == ".html":
            other = parsed.get(path.resolve())
            if other is None:
                c = Collector()
                c.feed(path.read_text(encoding="utf-8", errors="replace"))
                other = c
                parsed[path.resolve()] = c
            if frag not in other.ids:
                problems["dead cross-page anchor"].append(f"{rel_page} -> {href}")

print(f"pages scanned:        {len(pages)}")
print(f"internal links checked: {checked}")
print(f"external links skipped: {external}")
total = sum(len(v) for v in problems.values())
print(f"problems:             {total}")

for kind, items in sorted(problems.items()):
    print(f"\n## {kind}  ({len(items)})")
    for item in items[:60]:
        print(f"  {item}")
    if len(items) > 60:
        print(f"  ... and {len(items) - 60} more")

sys.exit(1 if total else 0)
