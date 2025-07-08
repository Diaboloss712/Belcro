from __future__ import annotations

import asyncio
import functools
import json
import re
from collections import Counter
from typing import Dict, List
from urllib.parse import urlparse

import httpx
import requests
from bs4 import BeautifulSoup, Tag
from langchain.schema import Document
from tqdm.asyncio import tqdm_asyncio

import config as CFG

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
        if section not in mapping:
            mapping[section] = []
        if slug not in mapping[section]:
            mapping[section].append(slug)
    return {k: sorted(v) for k, v in mapping.items()}, live_ver

def extract_prefix_from_variables(soup) -> str | None:
    prefixes = []
    # Sass 변수 섹션에서 접두사 추출
    section = soup.find("h3", id="sass-variables")
    if not section:
        return None
    
    # 코드 블록에서 변수 추출
    code_block = section.find_next("pre", class_="language-scss")
    if not code_block:
        return None
    
    code = code_block.get_text()
    for line in code.splitlines():
        if line.strip().startswith("$"):
            var_name = line.strip().split(":")[0].split("-")[0]
            if len(var_name) > 1:
                prefixes.append(var_name)
    
    if not prefixes:
        return None
    return Counter(prefixes).most_common(1)[0][0]

def extract_variables_table(soup, header_text, slug):
    # 코드 블록에서 변수 추출
    section = soup.find(lambda tag: tag.name in ["h2", "h3"] and 
                       header_text.lower() in tag.get_text(strip=True).lower())
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
    horizontal = sorted(set(all_vars))
    return horizontal

def extract_structure(el: Tag, prefix: str, depth: int = 0, exclude_tags=("script", "style")) -> list[dict]:
    if el is None or el.name in exclude_tags:
        return []
    children_struct = []
    for child in el.find_all(recursive=False):
        classes = [c for c in (child.get("class") or []) if c.startswith(prefix)]
        if not classes:
            children = extract_structure(child, prefix, depth + 1, exclude_tags)
            if children:
                node = {
                    "classes": classes,
                    "depth": depth,
                    "children": children
                }
                children_struct.append(node)
            continue
        node = {
            "classes": classes,
            "depth": depth,
            "children": extract_structure(child, prefix, depth + 1, exclude_tags)
        }
        children_struct.append(node)
    return children_struct

def extract_keywords(soup, slug):
    keywords = {slug, slug + "s"}
    stopwords = {
        "about", "example", "examples", "content types", "sizing", "text alignment",
        "navigation", "images", "horizontal", "card styles", "card layout", "css"
    }
    main = soup.find("main")
    if main:
        for tag in main.find_all(["h1", "h2"]):
            text = tag.get_text(strip=True).lower()
            if slug in text and text not in stopwords:
                keywords.add(text)
    return sorted(keywords)

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

        # Bootstrap 5.3+에서 'bd-example' 대신 'example' 사용
        example_div = main.find("div", class_="example") or main.find("div", class_="bd-example")
        if not example_div:
            return None

        prefix = extract_prefix_from_variables(soup) or slug

        description = ""
        h1 = main.find("h1")
        if h1:
            desc = h1.find_next_sibling("p")
            if desc:
                description = desc.get_text(strip=True)
            else:
                description = h1.get_text(strip=True)
        if not description:
            description = re.sub(r"\s+", " ", main.get_text(strip=True))[:300]

        hierarchy = extract_structure(example_div, prefix)
        horizontal = extract_horizontal(soup, prefix)
        keywords = extract_keywords(soup, slug)

        return Document(
            page_content=description,
            id=slug,
            metadata={
                "url": url,
                "section": section,
                "slug": slug,
                "description": description,
                "example": str(example_div),
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
    tasks = []

    for section, slugs in mapping.items():
        for slug in slugs:
            tasks.append(crawl_one(section, slug, ver))

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