from __future__ import annotations

import asyncio, json, pathlib, uvicorn
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config as CFG
from crawling import crawl
from embed import embvector  # ✅ embedding 로직은 외부에서 호출

# ─────────── 설정 ─────────────────────────────
DATA_DIR   = pathlib.Path("data")
META_FILE  = DATA_DIR / "meta.json"
PINE_INDEX = "upstage-index"
llm_chat = None

# ─────────── 2. FastAPI 초기화 ──────────────────
app = FastAPI(title="Bootstrap-RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# DTO
class ChatRequest(BaseModel):
    question: str
    context: str | None = None

class ChatResponse(BaseModel):
    answer: str

def _merge(ctx: str | None, q: str) -> str:
    return f"{ctx.strip()}\nUser: {q}" if ctx else q

# ─────────── 3. 리소스 검사 & 갱신 ─────────────────────────────
async def _verify_and_update() -> None:
    latest = CFG.detect_latest_version()
    pq_path = DATA_DIR / f"docs_{latest}.parquet"

    if CFG.DOC_VERSION == latest and META_FILE.exists() and pq_path.exists():
        return

    print("목표 : " + CFG.DOC_VERSION, "latest : " + latest)
    print("META 존재 : ", META_FILE.exists())
    print("pq_path 존재 : ", pq_path.exists())

    # ✅ 크롤링
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    docs = await crawl(outfile=str(pq_path))

    # ✅ 외부 embedding 함수 호출
    await asyncio.to_thread(embvector, docs, namespace=latest)

    # ✅ CONFIG 및 메타 갱신
    CFG.DOC_VERSION = latest
    CFG.COMPONENTS = sorted({
        d.metadata["keywords"][0]
        for d in docs
        if d.metadata.get("is_component")
    })

    META_FILE.write_text(json.dumps({
        "doc_version": latest,
        "components": CFG.COMPONENTS,
        "generated_at": datetime.now().isoformat()
    }, indent=2, ensure_ascii=False))

# ─────────── 4. startup 이벤트 ───────────────────────
@app.on_event("startup")
async def bootstrap():
    # 1) meta.json → CFG 반영
    if META_FILE.exists():
        meta = json.loads(META_FILE.read_text())
        CFG.DOC_VERSION = meta["doc_version"]
        CFG.COMPONENTS  = meta["components"]

    # 2) crawling 모듈에 최신 버전 전파
    import crawling
    crawling.DOC_VERSION = CFG.DOC_VERSION

    # 3) 크롤링 및 벡터 업데이트
    await _verify_and_update()

    # 4) 벡터 스토어 연결
    from langchain_upstage import UpstageEmbeddings
    from langchain_pinecone import PineconeVectorStore

    vec = PineconeVectorStore.from_existing_index(
        index_name=PINE_INDEX,
        embedding=UpstageEmbeddings(
            model="embedding-query",
            api_key=CFG.UPSTAGE_API_KEY
        ),
        namespace=CFG.DOC_VERSION
    )

    global llm_chat
    from llm import set_vectorstore, chat as _chat, _compiled_patterns
    set_vectorstore(vec)
    _compiled_patterns._cache = {}
    llm_chat = _chat

# ─────────── 5. API 엔드포인트 ─────────────────────────────
@app.post("/ask", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if llm_chat is None:
        raise HTTPException(503, "Model not ready")
    prompt = _merge(req.context, req.question)
    try:
        res = llm_chat(prompt)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return ChatResponse(**res)

@app.get("/health")
async def health():
    return {"doc_version": CFG.DOC_VERSION, "status": "ok"}

@app.get("/check")
async def check_update():
    await _verify_and_update()
    return {
        "doc_version": CFG.DOC_VERSION,
        "components": len(CFG.COMPONENTS)
    }

# ─────────── 6. 로컬 실행 ─────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
