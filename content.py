"""Loads the hand-written content of the dashboard from the `content/` folder.

Everything a portfolio manager writes (investment theses, journal entries,
watchlist, multi-asset view) lives in plain Markdown files, separate from the
analytics code, so it can be edited directly on GitHub without touching Python.

Journal entries use a small front-matter header:

    ---
    date: 2026-11-23            # date the note refers to (used for sorting / chart marker)
    title: Short title
    period_start: 2026-10-01    # optional: window analysed (stats are computed automatically)
    period_end: 2026-11-23      # optional
    written: 2026-11-23         # date the note was actually written
    tag: drawdown               # rally | drawdown | consolidation | outlook | review
    ---
"""
from __future__ import annotations

import re
from pathlib import Path

try:
    import markdown as _md

    def render_md(text: str) -> str:
        return _md.markdown(text, extensions=["extra", "sane_lists"])
except ImportError:  # pragma: no cover - fallback if the package is missing
    import html as _html

    def render_md(text: str) -> str:
        paras = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
        return "".join(f"<p>{_html.escape(p)}</p>" for p in paras)


FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    m = FRONT_MATTER.match(raw)
    if not m:
        return {}, raw
    meta = {}
    for line in m.group(1).splitlines():
        line = line.split("  #", 1)[0].strip()  # allow trailing "  # comments"
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, raw[m.end():]


def load_journal(folder: Path) -> list[dict]:
    """All journal entries, newest first."""
    if not folder.exists():
        return []
    entries = []
    for f in sorted(folder.glob("*.md")):
        meta, body = _parse_front_matter(f.read_text(encoding="utf-8"))
        if "date" not in meta:
            continue
        meta["html"] = render_md(body)
        meta["slug"] = f.stem
        entries.append(meta)
    return sorted(entries, key=lambda e: e["date"], reverse=True)


def load_sections(path: Path) -> list[dict]:
    """Split a Markdown file on '## ' headings -> [{title, key, html}]."""
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    _, raw = _parse_front_matter(raw)
    parts = re.split(r"^##\s+", raw, flags=re.M)
    intro = parts[0].strip()
    out = []
    if intro:
        out.append({"title": "", "key": "_intro", "html": render_md(intro)})
    for p in parts[1:]:
        title, _, body = p.partition("\n")
        title = title.strip()
        key = title.split()[0].upper() if title else ""
        out.append({"title": title, "key": key, "html": render_md(body.strip())})
    return out


def load_content(root: Path) -> dict:
    c = root / "content"
    return {
        "theses": load_sections(c / "theses.md"),
        "journal": load_journal(c / "journal"),
        "watchlist": load_sections(c / "watchlist.md"),
        "multi_asset": load_sections(c / "multi_asset.md"),
        "about": load_sections(c / "about.md"),
    }
