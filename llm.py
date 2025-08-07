import ast
import json
import asyncio
import re
import textwrap
from collections import defaultdict
from html import unescape
from pathlib import Path
from typing import Dict, List

from pyarrow import parquet as pq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

import config as CFG
from parsers import parse_code_with_lines
from crawling import crawl

_VEC: PineconeVectorStore | None = None
_LLM = ChatUpstage(api_key=CFG.UPSTAGE_API_KEY)
_PARSER = StrOutputParser()

_SYSTEM_TMPL = (
    "You are a senior front-end engineer.\n"
    "Generate a minimal Bootstrap HTML snippet inside a ```body``` block,\n"
    "then under a '## 설명' heading, write a one-paragraph explanation in Korean.\n"
    "Do NOT output any <html>, <head>, or <body> tags—only their inner contents."
)
_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM_TMPL), ("user", "{input}")])

_QSUM_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Summarize the user request into ONE short sentence, focusing on desired Bootstrap components and actions. Do NOT add explanations or markup."),
    ("user", "{query}")
])

def safe_list(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return []
    return []

def set_vectorstore(vs: PineconeVectorStore):
    global _VEC
    _VEC = vs

def _vec() -> PineconeVectorStore:
    if _VEC is None:
        raise RuntimeError("VectorStore not initialised.")
    return _VEC

def ensure_vectorstore_ready(namespace: str) -> None:
    pc = Pinecone(api_key=CFG.PINECONE_API_KEY)
    if "upstage-index" not in pc.list_indexes().names():
        pc.create_index(
            name="upstage-index",
            dimension=4096,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    vec = PineconeVectorStore.from_existing_index(
        index_name="upstage-index",
        embedding=UpstageEmbeddings(model="embedding-query", api_key=CFG.UPSTAGE_API_KEY),
        namespace=namespace,
    )
    set_vectorstore(vec)

def _variants(base: str) -> List[str]:
    forms = {base}
    if "-" in base: forms |= {base.replace("-", " "), base.replace("-", "")}
    if " " in base: forms.add(base.replace(" ", ""))
    if not base.endswith("s"): forms.add(f"{base}s")
    return list(forms)

_compiled_patterns_cache: Dict[tuple, Dict[str, re.Pattern]] = {}

def _compiled_patterns() -> Dict[str, re.Pattern]:
    key = tuple(CFG.COMPONENTS)
    if key in _compiled_patterns_cache:
        return _compiled_patterns_cache[key]
    pats = {
        c: re.compile(rf"\\b(?:{'|'.join(map(re.escape, _variants(str(c).lower())))})\\b(?!-)\b", re.I)
        for c in CFG.COMPONENTS if isinstance(c, str)
    }
    _compiled_patterns_cache[key] = pats
    return pats

def _extract_components(text: str) -> List[str]:
    if not isinstance(text, str):
        raise TypeError(f"Expected string in _extract_components, got {type(text)}")
    low = text.lower()
    return [c for c, p in _compiled_patterns().items() if p.search(low)]

def _ensure_parquet(pq_path: Path) -> None:
    if pq_path.exists(): return
    from crawling import crawl
    async def _job(): await crawl(outfile=str(pq_path))
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_job())
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as ex:
            ex.submit(lambda: asyncio.run(_job())).result()

def _load_bm25_docs(doc_version: str) -> List[Document]:
    path = Path(f"data/docs_{doc_version}.parquet")
    _ensure_parquet(path)
    table = pq.read_table(path)

    docs = []
    for row in table.to_pylist():
        metadata = {}
        for key in [
            "url", "section", "slug", "description", "example",
            "structure", "horizontal_groups", "keywords", "style_categories", "component"
        ]:
            val = row.get(key)
            if key in ["structure", "horizontal_groups", "keywords"]:
                try:
                    val = json.loads(val)
                except Exception:
                    val = []
            metadata[key] = val
        docs.append(Document(
            page_content=row.get("page_content", ""),
            metadata=metadata,
            id=row.get("id", "")
        ))
    return docs

_BM25_CACHE: dict[str, BM25Retriever] = {}

def get_retriever(prompt: str, doc_version: str) -> EnsembleRetriever:
    if doc_version not in _BM25_CACHE:
        _BM25_CACHE[doc_version] = BM25Retriever.from_documents(_load_bm25_docs(doc_version))

    base: List = [_BM25_CACHE[doc_version]]
    for comp in _extract_components(prompt):
        base.append(_vec().as_retriever(search_type="mmr", search_kwargs={"k": 2, "filter": {"slug": {"$eq": comp}}}))
    base.append(_vec().as_retriever(search_type="mmr", search_kwargs={"k": 3}))

    base_w = [1.0] if len(base) == 1 else [0.3] + [(0.7)/(len(base)-1)]*(len(base)-1)
    return EnsembleRetriever(retrievers=base, weights=base_w)

def _summarize_query(q: str) -> str:
    result = (_QSUM_PROMPT | _LLM | _PARSER).invoke({"query": q})
    if not isinstance(result, str):
        raise TypeError(f"LLM summary must return string, got {type(result)} → {result}")
    return result.strip()

def _make_prompt(summary: str, comps: List[str], structs: List[str],
                  horizontal: list[str] = None,
                  hierarchy: list[str] = None) -> str:
    prompt = f"""User request: {summary}
Components in context: {', '.join(sorted(comps))}
Available classes: {', '.join(sorted(structs))}"""

    if horizontal:
        prompt += f"\n\nAdd the following horizontal layout classes where appropriate: {', '.join(horizontal)}"

    if hierarchy:
        prompt += f"\n\nAdjust the component hierarchy according to: {', '.join(hierarchy)}"

    prompt += "\n\nPlease follow the system prompt: 먼저 HTML 코드 블록을, 그 아래에 '## 설명' 헤딩과 한국어 설명을 작성하세요."
    return prompt.strip()

def _select_docs_by_component(docs: List[Document], *, per_comp=1, limit=4) -> List[Document]:
    bucket: defaultdict[str, List[Document]] = defaultdict(list)
    for d in docs:
        if not hasattr(d, 'metadata') or not isinstance(d.metadata, dict):
            continue
        c = d.metadata.get("slug", "unknown")
        if not isinstance(c, str):
            continue
        if len(bucket[c]) < per_comp:
            bucket[c].append(d)
    picked: List[Document] = []
    for d in docs:
        c = d.metadata.get("slug")
        if not isinstance(c, str):
            continue
        while bucket[c]:
            picked.append(bucket[c].pop(0))
            if len(picked) >= limit:
                return picked
    return picked

def clean_html_snippet(raw: str):
    stripped = re.sub(r"```(?:html)?\s*|\s*```", "", raw.strip())
    stripped = stripped.replace("\\n", "\n")
    return unescape(stripped).strip()

def extract_html_from_llm_output(raw: str) -> str:
    match = re.search(r"```html\s*(.*?)\s*```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return clean_html_snippet(raw)

def build_class_tables_from_docs(docs: List[Document]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    h_table: dict[str, set[str]] = {}
    hier_table: dict[str, set[str]] = {}

    for doc in docs:
        if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
            continue

        slug = doc.metadata.get("slug", "")
        if not isinstance(slug, str):
            continue
        slug = slug.strip().lower()

        horizontal = safe_list(doc.metadata.get("horizontal_groups", []))
        hierarchy = safe_list(doc.metadata.get("hierarchy_groups", []))

        if horizontal:
            h_table.setdefault(slug, set()).update(horizontal)
        if hierarchy:
            hier_table.setdefault(slug, set()).update(hierarchy)

    return (
        {k: sorted(v) for k, v in h_table.items()},
        {k: sorted(v) for k, v in hier_table.items()},
    )

def extract_used_horizontal_classes(html: str, class_table: dict[str, list[str]]) -> list[str]:
    all_class_names = re.findall(r'class="([^"]+)"', html)
    used_classes = set()
    for class_attr in all_class_names:
        used_classes |= set(class_attr.strip().split())

    horizontal_candidates = set()
    for _, class_list in class_table.items():
        horizontal_candidates.update(class_list)

    return sorted(list(used_classes & horizontal_candidates))

def chat(query: str, doc_version: str):
    summary = _summarize_query(query)
    retriever = get_retriever(summary, doc_version)
    docs = list(retriever.invoke(summary, k=8))
    if not docs:
        raise ValueError("No relevant documents")

    picked = _select_docs_by_component(docs)
    
    comps = set()
    structs = set()
    for d in picked:
        if not isinstance(d.metadata, dict):
            continue
        
        slug_value = d.metadata.get("slug")
        if isinstance(slug_value, str):
            comps.add(slug_value)
            
        structure_list = safe_list(d.metadata.get("structure", []))
        if isinstance(structure_list, list):
            for cls in structure_list:
                if isinstance(cls, str):
                    structs.add(cls)

    horizontal_table, hierarchy_table = build_class_tables_from_docs(picked)

    prompt = _make_prompt(summary, list(comps), list(structs),
                          horizontal=[h for v in horizontal_table.values() for h in v],
                          hierarchy=[h for v in hierarchy_table.values() for h in v])

    raw = (_PROMPT | _LLM | _PARSER).invoke({"input": prompt}).strip()
    raw = raw.replace("\\n", "").replace("\\", "")

    html_code = extract_html_from_llm_output(raw)
    lines = parse_code_with_lines(
        html_code,
        horizontal_table,
        selected_components=list(comps)
    )

    horizontal_options = sorted({h for v in horizontal_table.values() for h in v})
    hierarchy_options = sorted({h for v in hierarchy_table.values() for h in v})
    return {
        "code": html_code,
        "lines": [line.dict() for line in lines],
        "selected_components": sorted(list(comps)),
        "horizontal_options": horizontal_options,
        "hierarchy_options": hierarchy_options,
        "structure": list(structs)
    }
