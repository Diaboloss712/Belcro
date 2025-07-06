from __future__ import annotations

import functools
import json
import re
import textwrap
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import httpx
import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from langchain.schema import Document
from tqdm.asyncio import tqdm_asyncio

import config as CFG

DOC_VERSION: str = CFG.DOC_VERSION
_SITEMAP_URL = "https://getbootstrap.com/sitemap-0.xml"
_DOC_PATH_RE = re.compile(r"/docs/([^/]+)/([^/]+)/([^/]+)/?")

@functools.cache
def load_sitemap() -> list[str]:
    xml = requests.get(_SITEMAP_URL, timeout=10).text
    soup = BeautifulSoup(xml, "xml")
    return [loc.text for loc in soup.find_all("loc")]

def detect_live_doc_version(urls: list[str]) -> str:
    for u in urls:
        m = _DOC_PATH_RE.search(urlparse(u).path)
        if m:
            return m.group(1)
    return DOC_VERSION

@functools.cache
def prefix_map() -> dict[str, List[str]]:
    global DOC_VERSION
    urls = load_sitemap()
    live_ver = detect_live_doc_version(urls)
    if live_ver != DOC_VERSION:
        print(f"[crawler] live docs version {live_ver} detected; overriding placeholder {DOC_VERSION}")
        DOC_VERSION = live_ver

    mapping: dict[str, set[str]] = {}
    for u in urls:
        m = _DOC_PATH_RE.search(urlparse(u).path)
        if not m:
            continue
        _, section, slug = m.groups()
        if 'rtl' in slug.lower() or section.lower() == 'examples':
            continue
        mapping.setdefault(section, set()).add(slug)

    return {sec: sorted(slugs) for sec, slugs in mapping.items()}

def build_url(section: str, page: str) -> str:
    return f"https://getbootstrap.com/docs/{DOC_VERSION}/{section}/{page}/"

def detect_component_name(url: str) -> str:
    m = _DOC_PATH_RE.search(urlparse(url).path)
    if m:
        return m.group(3)
    return urlparse(url).path.rstrip("/").split("/")[-1]

ALLOWED_ATTRS = {
    "class", "type", "role",
    *[f"data-bs-{x}" for x in ["toggle", "dismiss", "backdrop", "target", "keyboard"]]
}

def extract_structure_classes(example: Tag, prefix: str) -> dict[str, List[str] | str]:
    classes = set()
    for node in example.find_all(True, class_=True):
        for cls in node.get("class", []):
            classes.add(cls)

    # 필터링된 attribute만 유지
    for tag in example.descendants:
        if isinstance(tag, Tag):
            tag.attrs = {k: v for k, v in tag.attrs.items() if k in ALLOWED_ATTRS}

    return {
        "structure": sorted(classes),
        "example": textwrap.dedent(example.decode()).strip()
    }

def extract_doc_text(example: Tag) -> str:
    fig = example.find_parent("figure")
    if fig:
        cap = fig.find("figcaption")
        if cap and cap.get_text(strip=True):
            return cap.get_text(" ", strip=True)

    parts: list[str] = []
    for sib in example.previous_siblings:
        if not isinstance(sib, Tag):
            continue
        if sib.name in {"h1", "h2", "h3", "h4", "h5", "hr"}:
            break
        if sib.name in {"p", "ul", "ol", "dl", "blockquote"}:
            txt = sib.get_text(" ", strip=True)
            if txt:
                parts.insert(0, txt)

    desc = " ".join(parts).strip()
    return desc or example.get_text(" ", strip=True)

def extract_component_docs(url: str, html: str) -> list[Document]:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main")
    if not main:
        return []

    slug = detect_component_name(url)
    docs: list[Document] = []

    for example in main.select(".bd-example"):
        struct_info = extract_structure_classes(example, prefix=slug)
        desc = extract_doc_text(example)
        is_comp = urlparse(url).path.split("/")[3] == "components"

        docs.append(
            Document(
                page_content=slug,
                metadata={
                    "section": "",  # h-section 제거함 (불안정하므로)
                    "url": url,
                    "description": desc,
                    "structure": struct_info["structure"],
                    "example": struct_info["example"],
                    "keywords": [slug] + struct_info["structure"],
                    "is_component": is_comp,
                    "slug": slug
                },
                id=slug
            )
        )
    return docs

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
    all_urls = [build_url(sec, slug) for sec, slugs in pmap.items() for slug in slugs]
    urls = [u for u in all_urls if '/examples/' not in u and '/rtl/' not in u]

    html_map = await gather_pages(urls)
    docs: list[Document] = []

    for url, html in html_map.items():
        docs.extend(extract_component_docs(url, html))

    print("Total docs:", len(docs))

    if outfile:
        df = pd.DataFrame({
            "page_content": [d.page_content for d in docs],
            "metadata": [json.dumps(d.metadata, ensure_ascii=False) for d in docs],
        })
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(outfile, compression="snappy")

    return docs
