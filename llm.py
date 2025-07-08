# llm.py (리팩토링 버전)

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
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

import config as CFG

_VEC: PineconeVectorStore | None = None
_LLM = ChatUpstage(api_key=CFG.UPSTAGE_API_KEY)
_PARSER = StrOutputParser()

# ↓ DOC_VERSION 제거
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
    pats = {c: re.compile(rf"\\b(?:{'|'.join(map(re.escape, _variants(c.lower())))})(?!-)\\b", re.I)
            for c in CFG.COMPONENTS}
    _compiled_patterns_cache[key] = pats
    return pats

def _extract_components(text: str) -> List[str]:
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
        with concurrent.futures.ThreadPoolExecutor() as ex:
            ex.submit(lambda: asyncio.run(_job())).result()

def _dense_retriever():
    return _vec().as_retriever(search_type="mmr", search_kwargs={"k": 3})

def _load_bm25_docs(doc_version: str) -> List[Document]:
    pq = Path(f"data/docs_{doc_version}.parquet")
    _ensure_parquet(pq)
    df = pd.read_parquet(pq)
    return [Document(
        page_content=r.page_content,
        id=r.id,
        metadata=r.metadata) for _, r in df.iterrows()
        ]

# ↓ doc_version 인자로 전달
_BM25_CACHE: dict[str, BM25Retriever] = {}

def get_retriever(prompt: str, doc_version: str) -> EnsembleRetriever:
    if doc_version not in _BM25_CACHE:
        _BM25_CACHE[doc_version] = BM25Retriever.from_documents(_load_bm25_docs(doc_version))

    base: List = [_BM25_CACHE[doc_version]]
    for comp in _extract_components(prompt):
        base.append(_vec().as_retriever(search_type="mmr", search_kwargs={"k": 2, "filter": {"component": {"$eq": comp}}}))
    base.append(_dense_retriever())
    base_w = [1.0] if len(base) == 1 else [0.3] + [(0.7)/(len(base)-1)]*(len(base)-1)
    return EnsembleRetriever(retrievers=base, weights=base_w)

def _summarize_query(q: str) -> str:
    return (_QSUM_PROMPT | _LLM | _PARSER).invoke({"query": q}).strip()

def _make_prompt(summary: str, comps: List[str], structs: List[str]) -> str:
    return textwrap.dedent(f"""
        User request: {summary}

        Components in context: {', '.join(sorted(comps))}
        Available classes: {', '.join(sorted(structs))}
        Please follow the system prompt: 먼저 HTML 코드 블록을, 그 아래에 '＃＃ 설명' 헤딩과 한국어 설명을 작성하세요.
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

def clean_html_snippet(raw: str):
    stripped = re.sub(r"```(?:html)?\\s*|\\s*```", "", raw.strip())
    stripped = stripped.replace("\\n", "\n")
    return unescape(stripped).strip()

def chat(query: str, doc_version: str):
    summary = _summarize_query(query)
    retriever = get_retriever(summary, doc_version)
    docs = list(retriever.invoke(summary, k=8))
    if not docs:
        raise ValueError("No relevant documents")

    picked = _select_docs_by_component(docs)
    structs = {cls for d in picked for cls in d.metadata.get("structure", [])}
    comps = {d.metadata["component"] for d in picked}
    prompt = _make_prompt(summary, list(comps), list(structs))

    raw = (_PROMPT | _LLM | _PARSER).invoke({"input": prompt}).strip()
    raw = raw.replace("\\n", "").replace("\\", "")

    code_match = re.search(r"```html\\s*(.*?)\\s*```", raw, re.S)
    if code_match:
        raw_code = code_match.group(1).strip()
        cleaned_code = clean_html_snippet(f"```html\n{raw_code}\n```")
        explanation = raw[code_match.end():].strip()
    else:
        cleaned_code = clean_html_snippet(raw)
        explanation = ""

    return {
        "answer": cleaned_code + explanation,
    }