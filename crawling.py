from __future__ import annotations

import functools
import json
import re
from collections import Counter
from typing import List
from urllib.parse import urlparse

import httpx
import requests
from bs4 import BeautifulSoup, Tag
from langchain.schema import Document
from tqdm.asyncio import tqdm_asyncio


_DOC_PATH_RE = re.compile(r"/docs/([^/]+)/([^/]+)/([^/]+)/?")

@functools.cache
def load_sitemap() -> list[str]:
    try:
        xml = requests.get("https://getbootstrap.com/sitemap-0.xml", timeout=10).text
        soup = BeautifulSoup(xml, "xml")
        return [loc.text for loc in soup.find_all("loc")]
    except Exception as e:
        print(f"사이트맵 로드 실패: {e}")
        return []

def detect_live_doc_version(urls: list[str]) -> str:
    for u in urls:
        m = _DOC_PATH_RE.search(urlparse(u).path)
        if m:
            return m.group(1)
    raise RuntimeError("라이브 버전 문서 경로를 찾을 수 없습니다")

def prefix_map() -> tuple[dict[str, List[str]], str]:
    urls = load_sitemap()
    live_ver = detect_live_doc_version(urls)
    mapping: dict[str, List[str]] = {}
    for u in urls:
        m = _DOC_PATH_RE.search(urlparse(u).path)
        if not m:
            continue
        _, section, slug = m.groups()
        if slug.lower() == "rtl" or section.lower() == "examples":
            continue
        mapping.setdefault(section, []).append(slug)
    return {k: sorted(v) for k, v in mapping.items()}, live_ver

def extract_prefix_from_variables(soup) -> str | None:
    header = soup.find(lambda tag: tag.name in {"h2", "h3"} and "sass variables" in tag.get_text(strip=True).lower())
    if not header:
        return None
    code_block = header.find_next("pre", class_="language-scss")
    if not code_block:
        return None
    code = code_block.get_text()
    prefixes = []
    for line in code.splitlines():
        if line.strip().startswith("$"):
            var_name = line.strip().split(":")[0].lstrip("$")
            if "-" in var_name:
                prefix = var_name.split("-")[0]
                prefixes.append(prefix)
    return Counter(prefixes).most_common(1)[0][0] if prefixes else None

def extract_variables_table(soup, header_text, slug):
    section = soup.find(lambda tag: tag.name in ["h2", "h3"] and header_text.lower() in tag.get_text(strip=True).lower())
    if not section:
        return []
    code_block = section.find_next("pre", class_="language-scss")
    if not code_block:
        return []
    variables = []
    code = code_block.get_text()
    for line in code.splitlines():
        if line.strip().startswith("$"):
            var = line.strip().split(":")[0]
            var_norm = var.lstrip("$").lstrip("-")
            if var_norm.startswith(slug):
                variables.append(var_norm)
    return variables

def extract_horizontal(soup, slug):
    css_vars = extract_variables_table(soup, "Variables", slug)
    sass_vars = extract_variables_table(soup, "Sass variables", slug)
    all_vars = css_vars + sass_vars
    return sorted(set(all_vars))

def extract_structure(el: Tag, prefix: str, depth: int = 0, exclude_tags=("script", "style")) -> list[dict]:
    if el is None or el.name in exclude_tags:
        return []
    children_struct = []
    for child in el.find_all(recursive=False):
        classes = [c for c in (child.get("class") or []) if prefix in c]
        node = {
            "classes": classes,
            "depth": depth,
            "children": extract_structure(child, prefix, depth + 1, exclude_tags)
        }
        if classes or node["children"]:
            children_struct.append(node)
    return children_struct

def extract_keywords(soup: BeautifulSoup, slug: str, prefix: str) -> list[str]:
    raw_keywords = {slug, slug + "s", prefix}
    main = soup.find("main")
    if not main:
        return sorted(raw_keywords)
    for tag in main.find_all(["h1", "h2", "h3"]):
        text = tag.get_text(strip=True)
        if slug in text.lower() or prefix in text.lower():
            raw_keywords.add(text)
    for tag in main.find_all(True):
        for cls in tag.get("class") or []:
            if prefix in cls:
                raw_keywords.add(cls)
        id_ = tag.get("id")
        if id_ and prefix in id_:
            raw_keywords.add(id_)
        for attr, val in tag.attrs.items():
            if isinstance(val, str):
                if prefix in val:
                    raw_keywords.add(val)
                if attr.startswith("data-") and prefix in attr:
                    raw_keywords.add(attr)
    cleaned_keywords = set()
    for kw in raw_keywords:
        kw = kw.strip()
        if not kw or kw.lower().startswith("link to this section:") or kw.startswith("http"):
            continue
        if kw.startswith("#"):
            kw = kw[1:]
        cleaned_keywords.add(kw)
    return sorted(cleaned_keywords)

def extract_description(soup: BeautifulSoup) -> str:
    lead = soup.select_one("main .bd-content p.lead")
    if lead:
        return lead.get_text(strip=True)
    subtitle = soup.select_one("main .bd-subtitle p")
    if subtitle:
        return subtitle.get_text(strip=True)
    return ""

def extract_example(main: Tag) -> str:
    for class_name in ["bd-example", "bd-example-snippet", "bd-code-snippet", "bd-example-row"]:
        example = main.find("div", class_=class_name)
        if example:
            return str(example)
    pre = main.find("pre")
    if pre and pre.find("code"):
        return str(pre)
    return ""

async def crawl_one(section: str, slug: str, ver: str) -> Document | None:
    url = f"https://getbootstrap.com/docs/{ver}/{section}/{slug}/"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(url)
            html = res.text
    except Exception as e:
        print(f"{url} 요청 실패: {e}")
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")
        main = soup.find("main")
        if not main:
            return None
        example_html = extract_example(main)
        if not example_html:
            return None
        example_div = BeautifulSoup(example_html, "html.parser")
        prefix = extract_prefix_from_variables(soup) or slug
        description = extract_description(soup)
        hierarchy = extract_structure(example_div, prefix)
        horizontal = extract_horizontal(soup, prefix)
        keywords = extract_keywords(soup, slug, prefix)
        return Document(
            page_content=description,
            id=slug,
            metadata={
                "url": url,
                "section": section,
                "slug": slug,
                "description": description,
                "example": example_html,
                "hierarchy": json.dumps(hierarchy, ensure_ascii=False),
                "horizontal": json.dumps(horizontal, ensure_ascii=False),
                "is_component": True,
                "keywords": keywords,
            },
        )
    except Exception as e:
        print(f"{url} 처리 실패: {e}")
        return None

async def crawl(*, outfile: str | None = None) -> list[Document]:
    mapping, ver = prefix_map()
    docs: list[Document] = []
    tasks = [crawl_one(section, slug, ver) for section, slugs in mapping.items() for slug in slugs]
    for coro in tqdm_asyncio.as_completed(tasks, desc="Crawling pages", total=len(tasks)):
        doc = await coro
        if doc:
            docs.append(doc)
    if outfile:
        import pandas as pd
        df = pd.DataFrame([{
            "id": d.id,
            "page_content": d.page_content,
            "metadata": d.metadata
        } for d in docs])
        df.to_parquet(outfile, index=False)
    return docs
