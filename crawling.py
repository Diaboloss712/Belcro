from __future__ import annotations

import argparse
import asyncio
import functools
import re
import textwrap
from typing import Dict, List
from urllib.parse import urlparse

import httpx
import requests
from bs4 import BeautifulSoup, Tag
from langchain.schema import Document
from tqdm.asyncio import tqdm_asyncio

import config as CFG

DOC_VERSION: str = CFG.DOC_VERSION


# ---------------------------------------------------------------------------
# 1. Sitemap helpers ---------------------------------------------------------
# ---------------------------------------------------------------------------

@functools.cache
def load_sitemap() -> list[str]:
    """Return every <loc> URL from the first sitemap file."""
    xml = requests.get("https://getbootstrap.com/sitemap-0.xml", timeout=10).text
    soup = BeautifulSoup(xml, "xml")
    return [loc.text for loc in soup.find_all("loc")]


_DOC_PATH_RE = re.compile(r"/docs/([^/]+)/([^/]+)/([^/]+)/?")


def detect_live_doc_version(urls: list[str]) -> str:
    """Extract the first <ver> part from a `/docs/<ver>/…` path in *urls*."""
    for u in urls:
        m = _DOC_PATH_RE.search(urlparse(u).path)
        if m:
            return m.group(1)
    raise RuntimeError("Could not detect Bootstrap docs version from sitemap")


@functools.cache
def prefix_map() -> dict[str, list[str]]:
    """Return {section: [page, …]} using the live sitemap and set DOC_VERSION."""
    global DOC_VERSION

    urls = load_sitemap()
    live_ver = detect_live_doc_version(urls)

    if live_ver != DOC_VERSION:
        print(f"[crawler] live docs version {live_ver} detected; "
              f"overriding placeholder {DOC_VERSION} at runtime.")
        DOC_VERSION = live_ver  # shadow only – don't write back to config

    mapping: dict[str, set[str]] = {}
    for u in urls:
        m = _DOC_PATH_RE.search(urlparse(u).path)
        if not m:
            continue
        _ver, section, page = m.groups()
        mapping.setdefault(section, set()).add(page)
    return {k: sorted(v) for k, v in mapping.items()}


# ---------------------------------------------------------------------------
# 2. HTML extraction helpers -------------------------------------------------
# ---------------------------------------------------------------------------

ALLOWED_ATTRS = {
    "class",
    "type",
    "role",
    *[f"data-bs-{x}" for x in [
        "toggle",
        "dismiss",
        "backdrop",
        "target",
        "keyboard",
    ]],
}


def detect_component_name(url: str) -> str:
    m = re.search(r"/components/([^/]+)/?", url)
    return m.group(1) if m else "unknown"


def extract_structure_classes(example_tag: Tag, prefix: str) -> Dict[str, List[str]]:
    classes: set[str] = set()
    for node in example_tag.find_all(class_=True):
        for cls in node.get("class", []):
            if cls.startswith(prefix):
                classes.add(cls)
    return {
        "structure": sorted(classes),
        "html": textwrap.dedent(example_tag.decode()).strip(),
    }


def extract_doc_text(example_tag: Tag, max_siblings: int = 5) -> str:
    """Return explanatory text preceding *example_tag* (with fallback)."""
    doc_parts: list[str] = []
    sib = example_tag.find_previous_sibling()
    while sib and len(doc_parts) < max_siblings:
        txt = sib.get_text(strip=True)
        if txt and sib.name not in {"script", "style"}:
            doc_parts.insert(0, txt)
        sib = sib.find_previous_sibling()

    if not doc_parts:
        fallback_txt = example_tag.get_text(" ", strip=True)
        if fallback_txt:
            doc_parts.append(fallback_txt)

    return " ".join(doc_parts)


# ---------------------------------------------------------------------------
# 3. Crawling ---------------------------------------------------------------
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


def build_url(section: str, page: str) -> str:
    return f"https://getbootstrap.com/docs/{DOC_VERSION}/{section}/{page}/"


def extract_component_docs(url: str, html: str) -> list[Document]:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main")
    if not main:
        return []

    component = detect_component_name(url)
    docs: list[Document] = []

    for ex in main.select(".bd-example"):
        body_text = extract_doc_text(ex)
        struct_info = extract_structure_classes(ex, prefix=component)

        header = (
            f"Component: {component}\n"
            f"Structure: {', '.join(struct_info['structure'])}\n\n"
        )
        docs.append(
            Document(
                page_content=header + body_text,
                metadata={
                    "component": component,
                    "url": url,
                    "structure": struct_info["structure"],
                    "html": struct_info["html"],
                },
            )
        )
    return docs


async def crawl(outfile: str | None) -> list[Document]:
    pmap = prefix_map()
    urls = [build_url(sec, page) for sec, pages in pmap.items() for page in pages if "rtl" not in page.lower()]

    html_map = await gather_pages(urls)

    docs: list[Document] = []
    for url, html in html_map.items():
        docs.extend(extract_component_docs(url, html))

    print("Total docs:", len(docs))

    if outfile:
        import pandas as pd
        df = pd.DataFrame(
            {
                "page_content": [d.page_content for d in docs],
                "metadata": [d.metadata for d in docs],
            }
        )
        df.to_parquet(outfile, compression="snappy")

    return docs