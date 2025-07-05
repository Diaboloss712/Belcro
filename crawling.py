from __future__ import annotations

import argparse
import asyncio
import functools
import re
import textwrap
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import httpx
import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from langchain.schema import Document
from tqdm.asyncio import tqdm_asyncio

import config as CFG

# ---------------------------------------------------------------------------
# Config and Regex
# ---------------------------------------------------------------------------
DOC_VERSION: str = CFG.DOC_VERSION
_SITEMAP_URL = "https://getbootstrap.com/sitemap-0.xml"
_DOC_PATH_RE = re.compile(r"/docs/([^/]+)/([^/]+)/([^/]+)/?")

# ---------------------------------------------------------------------------
# 1. Sitemap helpers
# ---------------------------------------------------------------------------
@functools.cache
def load_sitemap() -> list[str]:
    """Return every <loc> URL from the single sitemap file."""
    xml = requests.get(_SITEMAP_URL, timeout=10).text
    soup = BeautifulSoup(xml, "xml")
    return [loc.text for loc in soup.find_all("loc")]


def detect_live_doc_version(urls: list[str]) -> str:
    """Extract the first version from URLs or fallback to config."""
    for u in urls:
        m = _DOC_PATH_RE.search(urlparse(u).path)
        if m:
            return m.group(1)
    # fallback if not found
    print(f"[crawler] could not detect live docs version; using DOC_VERSION={DOC_VERSION}")
    return DOC_VERSION

@functools.cache
def prefix_map() -> dict[str, List[str]]:
    """Return {section: [page_slug, ...]} mapping from sitemap."""
    global DOC_VERSION
    urls = load_sitemap()
    live_ver = detect_live_doc_version(urls)
    if live_ver != DOC_VERSION:
        print(f"[crawler] live docs version {live_ver} detected; overriding placeholder {DOC_VERSION} at runtime.")
        DOC_VERSION = live_ver

    mapping: dict[str, set[str]] = {}
    for u in urls:
        m = _DOC_PATH_RE.search(urlparse(u).path)
        if not m:
            continue
        _, section, slug = m.groups()
        if 'rtl' in slug.lower():
            continue
        mapping.setdefault(section, set()).add(slug)
    return {sec: sorted(slugs) for sec, slugs in mapping.items()}

# ---------------------------------------------------------------------------
# 2. URL builder
# ---------------------------------------------------------------------------

def build_url(section: str, page: str) -> str:
    return f"https://getbootstrap.com/docs/{DOC_VERSION}/{section}/{page}/"

# ---------------------------------------------------------------------------
# 3. Component name helper
# ---------------------------------------------------------------------------
def detect_component_name(url: str) -> str:
    """Extract component name from URL path."""
    m = re.search(r"/components/([^/]+)/?", url)
    return m.group(1) if m else "unknown"

# ---------------------------------------------------------------------------
# 4. HTML extraction helpers
# ---------------------------------------------------------------------------
ALLOWED_ATTRS = {
    "class", "type", "role",
    *[f"data-bs-{x}" for x in ["toggle", "dismiss", "backdrop", "target", "keyboard"]]
}

def extract_structure_classes(example: Tag, prefix: str) -> dict[str, List[str] | str]:
    classes: set[str] = set()
    for node in example.find_all(True, class_=True):
        for cls in node.get("class", []):
            if cls.startswith(prefix):
                classes.add(cls)
    # prune unwanted attributes
    for tag in example.descendants:
        if isinstance(tag, Tag):
            for attr in list(tag.attrs):
                if attr not in ALLOWED_ATTRS:
                    del tag.attrs[attr]
    return {"structure": sorted(classes), "html": textwrap.dedent(example.decode()).strip()}


def extract_doc_text(example: Tag) -> str:
    """
    1) If in <figure>, use <figcaption>
    2) Else, collect adjacent <p>,<ul>,<ol>,<dl>,<blockquote> until header/hr
    3) Fallback to example inner text
    """
    # figure caption
    fig = example.find_parent("figure")
    if fig:
        cap = fig.find("figcaption")
        if cap and cap.get_text(strip=True):
            return cap.get_text(" ", strip=True)
    # adjacent blocks
    parts: list[str] = []
    for sib in example.previous_siblings:
        if isinstance(sib, Tag):
            if sib.name in {"h1", "h2", "h3", "h4", "h5", "hr"}:
                break
            if sib.name in {"p", "ul", "ol", "dl", "blockquote"}:
                txt = sib.get_text(" ", strip=True)
                if txt:
                    parts.insert(0, txt)
    desc = " ".join(parts).strip()
    return desc or example.get_text(" ", strip=True)

# ---------------------------------------------------------------------------
# 5. Page parsing
# ---------------------------------------------------------------------------
def extract_component_docs(url: str, html: str) -> list[Document]:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main")
    if not main:
        return []

    component = detect_component_name(url)
    docs: list[Document] = []

    buffer: list[str] = []
    current_section: str | None = None

    for node in main.descendants:
        if not isinstance(node, Tag):
            continue
        # header marks new section
        if node.name in {"h1", "h2", "h3"}:
            current_section = node.get_text(strip=True)
            buffer.clear()
            continue
        # accumulate description blocks
        if node.name in {"p", "ul", "ol", "dl", "blockquote"}:
            txt = node.get_text(" ", strip=True)
            if txt:
                buffer.append(txt)
            continue
        # example block
        if node.name == "div" and "bd-example" in node.get("class", []):
            struct_info = extract_structure_classes(node, prefix=component)
            desc = "\n\n".join(buffer).strip() or extract_doc_text(node)
            header = (
                f"Component: {component}\n"
                f"Section: {current_section or 'unknown'}\n"
                f"Structure: {', '.join(struct_info['structure'])}\n\n"
            )
            docs.append(
                Document(
                    page_content=header + desc,
                    metadata={
                        "component": component,
                        "section": current_section,
                        "url": url,
                        "structure": struct_info["structure"],
                        "html": struct_info["html"],
                        "description": desc,
                    },
                )
            )
            buffer.clear()

    return docs

# ---------------------------------------------------------------------------
# 6. Crawling
# ---------------------------------------------------------------------------
async def fetch_html(session: httpx.AsyncClient, url: str) -> str | None:
    try:
        r = await session.get(url, timeout=10)
        r.raise_for_status()
        return r.text
    except Exception:
        return None

async def gather_pages(urls: list[str]) -> dict[str, str]:
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=20)) as client:
        pages = await tqdm_asyncio.gather(*(fetch_html(client, u) for u in urls), desc="Download")
    return {u: html for u, html in zip(urls, pages) if html}

async def crawl(outfile: str | None = None) -> list[Document]:
    pmap = prefix_map()
    urls = [build_url(sec, page) for sec, pages in pmap.items() for page in pages]
    html_map = await gather_pages(urls)
    docs: list[Document] = []
    for u, h in html_map.items():
        docs.extend(extract_component_docs(u, h))
    print("Total docs:", len(docs))
    if outfile:
        df = pd.DataFrame({
            "page_content": [d.page_content for d in docs],
            "metadata": [d.metadata for d in docs]
        })
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(outfile, compression="snappy")
    return docs

# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bootstrap docs crawler')
    parser.add_argument('-o', '--outfile', type=str, help='Parquet output path')
    args = parser.parse_args()
    result = asyncio.run(crawl(args.outfile))
    if result:
        import json
        print(json.dumps(result[0].metadata, indent=2, ensure_ascii=False))
