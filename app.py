from __future__ import annotations

from fastmcp import FastMCP
import mcp
import asyncio, json, pathlib, uvicorn
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

import config as CFG
from crawling import crawl
from embed import embvector
from llm import chat as llm_chat, ensure_vectorstore_ready, _compiled_patterns
from models import ChatRequest, ChatResponse

DATA_DIR   = pathlib.Path("data")
META_FILE  = DATA_DIR / "meta.json"

mcp  = FastMCP("belcro-v1")
app  = mcp.create_app()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

def _merge(ctx: str | None, q: str) -> str:
    return f"{ctx.strip()}\nUser: {q}" if ctx else q

from llm import get_retriever

def retrieve_docs(query: str, doc_version: str):
    retriever = get_retriever(query, doc_version)
    return retriever.invoke(query, k=8)

def structure_to_string(structure: list[dict], indent: int = 0) -> str:
    """계층 구조 트리를 읽기 좋은 문자열로 요약"""
    lines = []
    for node in structure:
        tag = node.get("tag", "div")
        classes = "." + ".".join(node.get("classes", [])) if node.get("classes") else ""
        prefix = "  " * indent
        lines.append(f"{prefix}{tag}{classes}")
        if node.get("children"):
            lines.append(structure_to_string(node["children"], indent + 1))
    return "\n".join(lines)

async def _verify_and_update():
    from crawling import prefix_map

    prev_ver = None
    if META_FILE.exists():
        meta = json.loads(META_FILE.read_text())
        prev_ver = meta.get("doc_version")

    mapping, live_ver = prefix_map()
    pq_path = DATA_DIR / f"docs_{live_ver}.parquet"

    if live_ver != prev_ver or not pq_path.exists():
        print(f"[update] Crawling triggered: meta={prev_ver} → live={live_ver}")
        docs = await crawl(outfile=str(pq_path))
        await asyncio.to_thread(embvector, docs, namespace=live_ver)

        CFG.COMPONENTS = sorted({d.metadata.get("component", d.metadata.get("section", "unknown")) for d in docs})
        META_FILE.write_text(json.dumps({
            "doc_version": live_ver,
            "components": CFG.COMPONENTS,
            "generated_at": datetime.now().isoformat()
        }, indent=2, ensure_ascii=False))
    else:
        print(f"[update] Skipped (already up to date): {live_ver}")

@app.on_event("startup")
async def bootstrap():
    if META_FILE.exists():
        meta = json.loads(META_FILE.read_text())
        CFG.COMPONENTS  = meta["components"]
        doc_version = meta["doc_version"]
    else:
        doc_version = "0.0"

    print(f"[app] DOC_VERSION: {doc_version}")
    await asyncio.to_thread(ensure_vectorstore_ready, namespace=doc_version)
    await _verify_and_update()
    _compiled_patterns._cache = {}

@mcp.tool()
def chat_tool(req: ChatRequest) -> ChatResponse:
    docs = retrieve_docs(req.question)
    doc = docs[0]
    
    structure = json.loads(doc.metadata["structure"])
    horizontal_groups = json.loads(doc.metadata["horizontal_groups"])
    struct_summary = structure_to_string(structure)

    prompt = f"""
    구조: {struct_summary}
    그룹: {horizontal_groups}
    예시: {doc.metadata['example']}
    설명: {doc.metadata['description']}
    """

    res = llm_chat(prompt, doc.metadata.get("doc_version", "0.0"))

    return ChatResponse(
        answer="LLM 응답 기반으로 코드 및 줄별 정보 반환",
        code=res["code"],
        lines=res["lines"]
    )

@app.post("/ask", response_model=ChatResponse)
async def chat(req: ChatRequest):
    docs = retrieve_docs(req.question)
    
    doc = docs[0]
    
    structure = json.loads(doc.metadata["structure"])
    horizontal_groups = json.loads(doc.metadata["horizontal_groups"])
    
    struct_summary = structure_to_string(structure)
    prompt = f"""
    구조: {struct_summary}
    그룹: {horizontal_groups}
    예시: {doc.metadata['example']}
    설명: {doc.metadata['description']}
    """
    
    res = llm_chat(prompt)
    
    return ChatResponse(answer=res)


@app.get("/health")
async def health():
    meta = json.loads(META_FILE.read_text()) if META_FILE.exists() else {}
    return {"doc_version": meta.get("doc_version", "unknown"), "status": "ok"}

@app.get("/check")
async def check_update():
    await _verify_and_update()
    meta = json.loads(META_FILE.read_text())
    return {"doc_version": meta["doc_version"], "components": len(meta.get("components", []))}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)