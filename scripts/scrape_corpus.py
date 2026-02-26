"""
Corpus scraper for AdapTeach — downloads Python educational content from
open-licensed sources and saves them in the corpus_raw JSON format.

Sources:
  - Python official tutorial (PSF Documentation License)
  - Automate the Boring Stuff 2e (CC BY-NC-SA 3.0)
  - Think Python 2e (CC BY-NC 3.0)

Usage:
  py scripts/scrape_corpus.py --out-dir data/corpus_raw/scraped/ --sources all
  py scripts/scrape_corpus.py --out-dir data/corpus_raw/scraped/ --sources python_docs
  py scripts/scrape_corpus.py --out-dir data/corpus_raw/scraped/ --sources atbs think_python
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    raise SystemExit(
        "Missing dependencies. Run:\n  pip install requests beautifulsoup4"
    )

TODAY = date.today().isoformat()
RATE_LIMIT_SECONDS = 1.2  # be polite

# ---------------------------------------------------------------------------
# Source definitions
# ---------------------------------------------------------------------------

PYTHON_DOCS_PAGES = [
    {
        "url": "https://docs.python.org/3/tutorial/introduction.html",
        "concept_tags": ["variables", "numbers", "strings", "operators", "data_types"],
        "difficulty": "intro",
    },
    {
        "url": "https://docs.python.org/3/tutorial/controlflow.html",
        "concept_tags": ["conditionals", "loops", "functions", "scope", "control_flow"],
        "difficulty": "intro",
    },
    {
        "url": "https://docs.python.org/3/tutorial/datastructures.html",
        "concept_tags": ["lists", "tuples", "sets", "dictionaries", "comprehensions"],
        "difficulty": "intro",
    },
    {
        "url": "https://docs.python.org/3/tutorial/modules.html",
        "concept_tags": ["modules", "imports", "scope"],
        "difficulty": "intro",
    },
    {
        "url": "https://docs.python.org/3/tutorial/inputoutput.html",
        "concept_tags": ["file_io", "strings", "formatting"],
        "difficulty": "intro",
    },
    {
        "url": "https://docs.python.org/3/tutorial/errors.html",
        "concept_tags": ["exceptions", "debugging", "error_handling"],
        "difficulty": "intro",
    },
    {
        "url": "https://docs.python.org/3/tutorial/classes.html",
        "concept_tags": ["OOP", "classes", "objects", "inheritance", "scope"],
        "difficulty": "intermediate",
    },
    {
        "url": "https://docs.python.org/3/tutorial/stdlib.html",
        "concept_tags": ["standard_library", "modules", "file_io"],
        "difficulty": "intro",
    },
    {
        "url": "https://docs.python.org/3/library/functions.html",
        "concept_tags": ["built_in_functions", "lists", "strings", "sorting"],
        "difficulty": "intro",
    },
    {
        "url": "https://docs.python.org/3/library/stdtypes.html",
        "concept_tags": ["data_types", "strings", "lists", "tuples", "dictionaries", "sets"],
        "difficulty": "intro",
    },
    {
        "url": "https://docs.python.org/3/library/exceptions.html",
        "concept_tags": ["exceptions", "error_handling", "debugging"],
        "difficulty": "intro",
    },
    {
        "url": "https://docs.python.org/3/glossary.html",
        "concept_tags": ["variables", "functions", "OOP", "scope", "data_types"],
        "difficulty": "intro",
    },
]

ATBS_PAGES = [
    {
        "url": "https://automatetheboringstuff.com/2e/chapter1/",
        "concept_tags": ["data_types", "variables", "operators", "strings", "numbers"],
        "difficulty": "intro",
    },
    {
        "url": "https://automatetheboringstuff.com/2e/chapter2/",
        "concept_tags": ["conditionals", "control_flow", "loops", "boolean"],
        "difficulty": "intro",
    },
    {
        "url": "https://automatetheboringstuff.com/2e/chapter3/",
        "concept_tags": ["functions", "scope", "return_values", "arguments"],
        "difficulty": "intro",
    },
    {
        "url": "https://automatetheboringstuff.com/2e/chapter4/",
        "concept_tags": ["lists", "indexing", "slicing", "loops"],
        "difficulty": "intro",
    },
    {
        "url": "https://automatetheboringstuff.com/2e/chapter5/",
        "concept_tags": ["dictionaries", "sets", "data_structures"],
        "difficulty": "intro",
    },
    {
        "url": "https://automatetheboringstuff.com/2e/chapter6/",
        "concept_tags": ["strings", "string_methods", "formatting"],
        "difficulty": "intro",
    },
    {
        "url": "https://automatetheboringstuff.com/2e/chapter7/",
        "concept_tags": ["regex", "strings", "pattern_matching"],
        "difficulty": "intermediate",
    },
    {
        "url": "https://automatetheboringstuff.com/2e/chapter8/",
        "concept_tags": ["file_io", "reading_writing", "pathlib"],
        "difficulty": "intro",
    },
    {
        "url": "https://automatetheboringstuff.com/2e/chapter9/",
        "concept_tags": ["file_io", "directories", "pathlib", "os"],
        "difficulty": "intro",
    },
    {
        "url": "https://automatetheboringstuff.com/2e/chapter10/",
        "concept_tags": ["debugging", "exceptions", "logging", "error_handling"],
        "difficulty": "intro",
    },
    {
        "url": "https://automatetheboringstuff.com/2e/chapter11/",
        "concept_tags": ["OOP", "classes", "objects"],
        "difficulty": "intermediate",
    },
]

THINK_PYTHON_PAGES = [
    {
        "url": "https://greenteapress.com/thinkpython2/html/thinkpython2002.html",
        "concept_tags": ["variables", "data_types", "operators", "expressions"],
        "difficulty": "intro",
    },
    {
        "url": "https://greenteapress.com/thinkpython2/html/thinkpython2003.html",
        "concept_tags": ["functions", "return_values", "arguments"],
        "difficulty": "intro",
    },
    {
        "url": "https://greenteapress.com/thinkpython2/html/thinkpython2004.html",
        "concept_tags": ["conditionals", "recursion", "control_flow"],
        "difficulty": "intro",
    },
    {
        "url": "https://greenteapress.com/thinkpython2/html/thinkpython2005.html",
        "concept_tags": ["functions", "recursion", "scope", "return_values"],
        "difficulty": "intro",
    },
    {
        "url": "https://greenteapress.com/thinkpython2/html/thinkpython2006.html",
        "concept_tags": ["loops", "iteration", "strings"],
        "difficulty": "intro",
    },
    {
        "url": "https://greenteapress.com/thinkpython2/html/thinkpython2007.html",
        "concept_tags": ["strings", "string_methods", "indexing", "slicing"],
        "difficulty": "intro",
    },
    {
        "url": "https://greenteapress.com/thinkpython2/html/thinkpython2008.html",
        "concept_tags": ["lists", "indexing", "slicing", "mutability"],
        "difficulty": "intro",
    },
    {
        "url": "https://greenteapress.com/thinkpython2/html/thinkpython2009.html",
        "concept_tags": ["tuples", "data_structures", "assignment"],
        "difficulty": "intro",
    },
    {
        "url": "https://greenteapress.com/thinkpython2/html/thinkpython2010.html",
        "concept_tags": ["dictionaries", "key_value", "data_structures"],
        "difficulty": "intro",
    },
    {
        "url": "https://greenteapress.com/thinkpython2/html/thinkpython2011.html",
        "concept_tags": ["tuples", "lists", "data_structures"],
        "difficulty": "intro",
    },
    {
        "url": "https://greenteapress.com/thinkpython2/html/thinkpython2012.html",
        "concept_tags": ["OOP", "classes", "objects"],
        "difficulty": "intermediate",
    },
    {
        "url": "https://greenteapress.com/thinkpython2/html/thinkpython2013.html",
        "concept_tags": ["OOP", "classes", "inheritance"],
        "difficulty": "intermediate",
    },
    {
        "url": "https://greenteapress.com/thinkpython2/html/thinkpython2014.html",
        "concept_tags": ["OOP", "special_methods", "operator_overloading"],
        "difficulty": "intermediate",
    },
    {
        "url": "https://greenteapress.com/thinkpython2/html/thinkpython2015.html",
        "concept_tags": ["recursion", "algorithms", "sorting"],
        "difficulty": "intermediate",
    },
    {
        "url": "https://greenteapress.com/thinkpython2/html/thinkpython2016.html",
        "concept_tags": ["recursion", "algorithms", "sorting", "searching"],
        "difficulty": "intermediate",
    },
    {
        "url": "https://greenteapress.com/thinkpython2/html/thinkpython2017.html",
        "concept_tags": ["debugging", "exceptions", "error_handling"],
        "difficulty": "intro",
    },
    {
        "url": "https://greenteapress.com/thinkpython2/html/thinkpython2020.html",
        "concept_tags": ["debugging", "error_handling", "common_errors"],
        "difficulty": "intro",
    },
]

ALL_SOURCES: dict[str, list[dict]] = {
    "python_docs": PYTHON_DOCS_PAGES,
    "atbs": ATBS_PAGES,
    "think_python": THINK_PYTHON_PAGES,
}

SOURCE_META: dict[str, dict[str, str]] = {
    "python_docs": {
        "license": "PSF Documentation License",
        "attribution": "Python Software Foundation",
    },
    "atbs": {
        "license": "CC BY-NC-SA 3.0",
        "attribution": "Al Sweigart, automatetheboringstuff.com",
    },
    "think_python": {
        "license": "CC BY-NC 3.0",
        "attribution": "Allen B. Downey, greenteapress.com",
    },
}

# ---------------------------------------------------------------------------
# HTML parsing helpers
# ---------------------------------------------------------------------------

def _fetch(url: str, retries: int = 3) -> str | None:
    headers = {"User-Agent": "AdapTeach-Corpus-Builder/1.0 (educational research)"}
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code == 200:
                return resp.text
            print(f"  HTTP {resp.status_code} for {url}")
            return None
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  Failed to fetch {url}: {e}")
                return None
    return None


def _extract_python_docs(soup: BeautifulSoup) -> tuple[str, str]:
    """Extract title + content from docs.python.org pages."""
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "Python Documentation"

    body = soup.find("div", class_="body") or soup.find("section") or soup.find("article")
    if body is None:
        body = soup.find("body")

    return title, _soup_to_text(body)


def _extract_atbs(soup: BeautifulSoup) -> tuple[str, str]:
    """Extract title + content from automatetheboringstuff.com pages.

    ATBS 2e pages are served as bare HTML fragments (no <html>/<body> wrapper),
    so soup.find('body') always returns None. We look for the calibre div that
    wraps each chapter, then fall back to the soup document root itself.
    """
    title_tag = soup.find("h2") or soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "Automate the Boring Stuff"

    body = (
        soup.find("div", id="content")
        or soup.find("main")
        or soup.find("article")
        or soup.find("div", class_="chapter")
        or soup.find("div", class_="calibre")
        or soup.find("body")
        or soup  # fragment fallback — soup IS the document root
    )
    return title, _soup_to_text(body)


def _extract_think_python(soup: BeautifulSoup) -> tuple[str, str]:
    """Extract title + content from greenteapress.com Think Python 2e pages.

    Think Python 2e HTML uses a flat old-style layout: content lives inside a
    <table> directly under <body>, with no semantic container divs. If <body>
    is missing (fragment), fall back to the soup root.
    """
    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "Think Python"

    body = (
        soup.find("body")
        or soup.find("table")  # TP2e content lives in a top-level table
        or soup  # fragment fallback
    )
    return title, _soup_to_text(body)


def _soup_to_text(tag: Any) -> str:
    """Convert a BeautifulSoup tag to clean plain text preserving code blocks."""
    if tag is None:
        return ""

    lines: list[str] = []

    for element in list(tag.descendants):
        if getattr(element, "name", None) in ("script", "style", "nav", "footer", "header", "aside"):
            element.decompose()

    for element in tag.children:
        _walk(element, lines)

    text = "\n".join(lines)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _walk(element: Any, lines: list[str]) -> None:
    """Recursively walk a BS4 element tree, preserving code blocks."""
    from bs4 import NavigableString, Tag

    if isinstance(element, NavigableString):
        text = str(element).strip()
        if text:
            lines.append(text)
        return

    if not isinstance(element, Tag):
        return

    tag_name = element.name

    if tag_name in ("script", "style", "nav", "footer", "aside", "header"):
        return

    if tag_name in ("pre", "code"):
        code_text = element.get_text()
        if tag_name == "pre":
            lines.append("\n```python\n" + code_text.rstrip() + "\n```\n")
        else:
            lines.append(f"`{code_text.strip()}`")
        return

    if tag_name in ("h1", "h2", "h3"):
        hashes = {"h1": "#", "h2": "##", "h3": "###"}[tag_name]
        lines.append(f"\n{hashes} {element.get_text(strip=True)}\n")
        return

    if tag_name in ("h4", "h5", "h6"):
        lines.append(f"\n#### {element.get_text(strip=True)}\n")
        return

    if tag_name == "p":
        text = element.get_text(separator=" ", strip=True)
        if text:
            lines.append(text + "\n")
        return

    if tag_name in ("ul", "ol"):
        for li in element.find_all("li", recursive=False):
            lines.append("- " + li.get_text(separator=" ", strip=True))
        lines.append("")
        return

    if tag_name in ("table", "tbody", "tr", "td", "th"):
        # Recurse into table cells so Think Python 2e (table-based layout) is extracted.
        # Navigation tables are already stripped by the decompose pass above.
        for child in element.children:
            _walk(child, lines)
        return

    # For everything else, recurse into children
    for child in element.children:
        _walk(child, lines)


# ---------------------------------------------------------------------------
# Main scrape logic
# ---------------------------------------------------------------------------

def _source_key(url: str) -> str:
    """Determine which source a URL belongs to."""
    if "docs.python.org" in url or "python.org/3/library" in url:
        return "python_docs"
    if "automatetheboringstuff.com" in url:
        return "atbs"
    if "greenteapress.com" in url or "thinkpython" in url:
        return "think_python"
    return "unknown"


def _extract_content(url: str, html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    source = _source_key(url)
    if source == "python_docs":
        return _extract_python_docs(soup)
    if source == "atbs":
        return _extract_atbs(soup)
    if source == "think_python":
        return _extract_think_python(soup)
    # Fallback: generic
    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url
    body = soup.find("body")
    return title, _soup_to_text(body)


def _make_doc_id(url: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", url.lower()).strip("-")[:60]
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"scraped-{slug}-{h}"


def _safe_filename(url: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", urlsplit(url).path.lower()).strip("_")[:60]
    h = hashlib.md5(url.encode()).hexdigest()[:6]
    return f"{slug}_{h}.json"


def scrape_page(url: str, meta: dict, out_dir: Path) -> bool:
    """Fetch, parse, and save one page. Returns True if saved."""
    out_file = out_dir / _safe_filename(url)
    if out_file.exists():
        print(f"  [skip] {out_file.name} already exists")
        return False

    print(f"  Fetching {url}")
    html = _fetch(url)
    if not html:
        return False

    title, content = _extract_content(url, html)

    if len(content) < 200:
        print(f"  [skip] too little content ({len(content)} chars)")
        return False

    source_key = _source_key(url)
    source_info = SOURCE_META.get(source_key, {"license": "unknown", "attribution": "unknown"})

    doc: dict[str, Any] = {
        "title": title,
        "content": content,
        "type": "tutorial",
        "concept_tags": meta.get("concept_tags", []),
        "difficulty": meta.get("difficulty", "intro"),
        "metadata": {
            "language": "en",
            "domain": "programming/python",
            "retrieved_at": TODAY,
            "intended_use": "educational corpus",
            "source": source_key,
        },
        "provenance": {
            "url": url,
            "license": source_info["license"],
            "attribution": source_info["attribution"],
            "retrieved_at": TODAY,
        },
        "ai_generated": False,
        "doc_id": _make_doc_id(url),
    }

    out_file.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [saved] {out_file.name}  ({len(content):,} chars)")
    return True


def run(sources: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    pages: list[tuple[dict, str]] = []
    for source_name in sources:
        source_pages = ALL_SOURCES.get(source_name, [])
        if not source_pages:
            print(f"Unknown source: {source_name}. Available: {list(ALL_SOURCES)}")
            continue
        for page_meta in source_pages:
            pages.append((page_meta, source_name))

    print(f"\nScraping {len(pages)} pages into {out_dir}\n")
    saved = 0
    skipped = 0

    for i, (meta, source_name) in enumerate(pages, 1):
        url = meta["url"]
        print(f"[{i}/{len(pages)}] {source_name}")
        result = scrape_page(url, meta, out_dir)
        if result:
            saved += 1
        else:
            skipped += 1
        if i < len(pages):
            time.sleep(RATE_LIMIT_SECONDS)

    print(f"\nDone. Saved: {saved}  Skipped/failed: {skipped}")
    print(f"Output: {out_dir.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Python educational corpus")
    parser.add_argument(
        "--out-dir",
        default="data/corpus_raw/scraped/",
        help="Output directory for scraped JSON files",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["all"],
        choices=["all", "python_docs", "atbs", "think_python"],
        help="Which sources to scrape",
    )
    args = parser.parse_args()

    sources = list(ALL_SOURCES.keys()) if "all" in args.sources else args.sources
    run(sources, Path(args.out_dir))


if __name__ == "__main__":
    main()
