
import asyncio, concurrent.futures, re, textwrap
from collections import defaultdict
from html import unescape
from pathlib import Path
from typing import Dict, List

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_upstage import ChatUpstage
from langchain_pinecone import PineconeVectorStore

import config as CFG

# ───────────────────── 0. Parquet 보장 ──────────────────────
def _ensure_parquet(pq_path: Path) -> None:
    if pq_path.exists():
        return
    from crawling import crawl
    async def _job(): await crawl(outfile=str(pq_path))
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_job())
    else:
        with concurrent.futures.ThreadPoolExecutor() as ex:
            ex.submit(lambda: asyncio.run(_job())).result()

# ───────────────────── 1. 컴포넌트 키워드 패턴 ───────────────
def _variants(base: str) -> List[str]:
    forms = {base}
    if "-" in base: forms |= {base.replace("-", " "), base.replace("-", "")}
    if " " in base: forms.add(base.replace(" ", ""))
    if not base.endswith("s"): forms.add(f"{base}s")
    return list(forms)

def _compiled_patterns() -> Dict[str, re.Pattern]:
    key, cache = tuple(CFG.COMPONENTS), getattr(_compiled_patterns, "_cache", {})
    if key in cache: return cache[key]
    pats = {c: re.compile(rf"\b(?:{'|'.join(map(re.escape, _variants(c.lower())))})\b", re.I)
            for c in key}
    cache[key] = pats; _compiled_patterns._cache = cache
    return pats

def _extract_components(text: str) -> List[str]:
    low = text.lower()
    return [c for c, p in _compiled_patterns().items() if p.search(low)]

# ───────────────────── 2. VectorStore & BM25 ────────────────
_VEC: PineconeVectorStore | None = None
def set_vectorstore(vs: PineconeVectorStore): global _VEC; _VEC = vs
def _vec() -> PineconeVectorStore:
    if _VEC is None: raise RuntimeError("VectorStore not initialised.")
    return _VEC

def _dense_retriever():
    return _vec().as_retriever(search_type="mmr", search_kwargs={"k": 3})

def _load_bm25_docs() -> List[Document]:
    pq = Path(f"data/docs_{CFG.DOC_VERSION}.parquet"); _ensure_parquet(pq)
    df = pd.read_parquet(pq)
    return [Document(page_content=r.page_content, metadata=r.metadata) for _, r in df.iterrows()]

_BM25 = BM25Retriever.from_documents(_load_bm25_docs())

# ───────────────────── 3. Ensemble Retriever ───────────────
def get_retriever(prompt: str) -> EnsembleRetriever:
    base: List = [_BM25]
    for comp in _extract_components(prompt):
        base.append(_vec().as_retriever(search_type="mmr",
                   search_kwargs={"k": 2, "filter": {"component": {"$eq": comp}}}))
    base.append(_dense_retriever())
    base_w = [1.0] if len(base) == 1 else [0.3] + [(0.7)/(len(base)-1)]*(len(base)-1)
    return EnsembleRetriever(retrievers=base, weights=base_w)

# ───────────────────── 4. LLM 세팅 ─────────────────────────
_LLM, _PARSER = ChatUpstage(api_key=CFG.UPSTAGE_API_KEY), StrOutputParser()

_SYSTEM_TMPL = ("You are a senior front-end engineer. "
                f"Generate minimal Bootstrap {CFG.DOC_VERSION} HTML.")
_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM_TMPL),
                                            ("user", "{input}")])

# 쿼리 요약 프롬프트
_QSUM_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ("Summarize the user request into ONE short sentence, "
                "focusing on desired Bootstrap components and actions. "
                "Do NOT add explanations or markup.")),
    ("user", "{query}"),
])

def _summarize_query(q: str) -> str:
    return (_QSUM_PROMPT | _LLM | _PARSER).invoke({"query": q}).strip()

def _make_prompt(summary: str, comps: List[str], structs: List[str]) -> str:
    return textwrap.dedent(f"""
        User request: {summary}

        Components in context: {', '.join(sorted(comps))}
        Available classes: {', '.join(sorted(structs))}

        Respond **only** with the <html> snippet.
    """).strip()

def _select_docs_by_component(docs: List[Document], *, per_comp=1, limit=4) -> List[Document]:
    bucket: defaultdict[str, List[Document]] = defaultdict(list)
    for d in docs:
        c = d.metadata.get("component", "unknown")
        if len(bucket[c]) < per_comp:
            bucket[c].append(d)
    picked: List[Document] = []
    for d in docs:
        c = d.metadata["component"]
        while bucket[c]:
            picked.append(bucket[c].pop(0))
            if len(picked) >= limit: return picked
    return picked

# ───────────────────── 5. 응답 후처리 ───────────────────────
def clean_html_snippet(raw: str) -> str:
    stripped = re.sub(r"```(?:html)?\s*|\s*```", "", raw.strip())
    stripped = stripped.replace("\\n", "\n")
    return unescape(stripped).strip()

# ───────────────────── 6. 외부 API 함수 ────────────────────
def chat(query: str) -> dict:
    summary   = _summarize_query(query)
    retriever = get_retriever(summary)
    docs      = list(retriever.invoke(summary, k=8))
    if not docs: raise ValueError("No relevant documents")

    picked    = _select_docs_by_component(docs)
    structs   = {cls for d in picked for cls in d.metadata.get("structure", [])}
    comps     = {d.metadata["component"] for d in picked}
    prompt    = _make_prompt(summary, list(comps), list(structs))

    raw_html  = (_PROMPT | _LLM | _PARSER).invoke({"input": prompt})
    cleaned   = clean_html_snippet(raw_html)

    return {"answer": cleaned}