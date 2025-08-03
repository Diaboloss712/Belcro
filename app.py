from fastapi import FastAPI
from fastmcp import FastMCP
import mcp
import asyncio, json, pathlib, uvicorn
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import config as CFG
from crawling import crawl
from embed import embvector
from llm import chat as llm_chat, ensure_vectorstore_ready, _compiled_patterns
from models import ChatRequest, ChatResponse, CodeLine
from llm import get_retriever

DATA_DIR = pathlib.Path("data")
META_FILE = DATA_DIR / "meta.json"


mcp = FastMCP(name="belcro-v1")

def safe_json_load(metadata: dict, key: str) -> list:
    raw = metadata.get(key, "[]")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []
    return raw if isinstance(raw, list) else []

@mcp.tool()
def chat_tool(req: ChatRequest) -> ChatResponse:
    try:
        res = llm_chat(req.question, doc_version="0.0")
        return ChatResponse(
            answer="아래 HTML 코드를 참고하세요.",
            code=res["code"],
            lines=[CodeLine(**line) for line in res["lines"]],
            horizontal_options=res.get("horizontal_options", []),
            selected_components=res.get("selected_components", [])
        )
    except Exception as e:
        return ChatResponse(
            answer=f"오류가 발생했습니다: {str(e)}",
            code="",
            lines=[]
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    if META_FILE.exists():
        meta = json.loads(META_FILE.read_text())
        CFG.COMPONENTS = meta["components"]
        doc_version = meta["doc_version"]
    else:
        doc_version = "0.0"

    print(f"[app] DOC_VERSION: {doc_version}")
    await asyncio.to_thread(ensure_vectorstore_ready, namespace=doc_version)
    await _verify_and_update()
    _compiled_patterns._cache = {}

    yield

mcp_app = mcp.http_app()

@asynccontextmanager
async def merged_lifespan(app: FastAPI):
    async with mcp_app.lifespan(app):
        async with lifespan(app):
            yield

app = FastAPI(lifespan=merged_lifespan, root_path="/api")
app.mount("/mcp", mcp_app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

def _merge(ctx: str | None, q: str) -> str:
    return f"{ctx.strip()}\nUser: {q}" if ctx else q

def retrieve_docs(query: str, doc_version: str = "0.0"):
    retriever = get_retriever(query, doc_version)
    return retriever.invoke(query, k=8)

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

        CFG.COMPONENTS = sorted({
            d.metadata.get("component", d.metadata.get("section", "unknown"))
            for d in docs
        })
        META_FILE.write_text(json.dumps({
            "doc_version": live_ver,
            "components": CFG.COMPONENTS,
            "generated_at": datetime.now().isoformat()
        }, indent=2, ensure_ascii=False))
    else:
        print(f"[update] Skipped (already up to date): {live_ver}")

@app.post("/ask", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        res = llm_chat(req.question, doc_version="0.0")
        return ChatResponse(
            answer="아래 HTML 코드를 참고하세요.",
            code=res["code"],
            lines=[CodeLine(**line) for line in res["lines"]]
        )
    except Exception as e:
        return ChatResponse(
            answer=f"오류가 발생했습니다: {str(e)}",
            code="",
            lines=[]
        )

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
    # mcp.run(transport="http", host="0.0.0.0", port=8000)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, timeout_keep_alive=60)
