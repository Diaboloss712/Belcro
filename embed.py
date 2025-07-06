# embed.py

from typing import List
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
from langchain_upstage import UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore
from tqdm import tqdm
import config as CFG  # DOC_VERSION, API KEY

# ────────────── 1. 문서 → 청크 ─────────────────────────────
def _chunk_document(doc: Document,
                    chunk_size: int = 1800,
                    overlap: int = 300) -> List[Document]:
    base = doc.page_content.strip()
    struct = " ".join(doc.metadata.get("structure", []))
    html = doc.metadata.get("example", "") or doc.metadata.get("html", "")
    full = f"{base}\n\nStructure: {struct}\n\nExample:\n{html}"

    chunks, start = [], 0
    while start < len(full):
        end = start + chunk_size
        chunk = Document(
            page_content=full[start:end].strip(),
            metadata=doc.metadata
        )
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

# ────────────── 2. 임베딩 + 업서트 ─────────────────────────
def embvector(docs: List[Document],
              *,
              namespace: str,
              index_name: str = "upstage-index"):
    """
    docs       : crawl() 이 반환한 Document 리스트
    namespace  : Pinecone 네임스페이스 (권장: 새 DOC_VERSION)
    반환값      : dense Retriever (k=3, mmr)
    """
    # ① Chunking
    chunked_docs = [c for d in docs for c in _chunk_document(d)]

    # ② Pinecone 인덱스 준비
    pc = Pinecone(api_key=CFG.PINECONE_API_KEY)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=4096,  # embedding dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # ③ VectorStore 연결
    emb = UpstageEmbeddings(model="embedding-query",
                            api_key=CFG.UPSTAGE_API_KEY)
    store = PineconeVectorStore(index_name=index_name,
                                embedding=emb,
                                namespace=namespace)

    # ④ ID 매핑: slug 또는 component 값 사용
    batch = 32
    for i in tqdm(range(0, len(chunked_docs), batch), desc=f"Pinecone upsert ({namespace})"):
        batch_docs = chunked_docs[i:i+batch]
        ids = [
            d.metadata.get("slug")
            or d.metadata.get("component")
            or d.page_content  # fallback
            for d in batch_docs
        ]
        store.add_documents(batch_docs, ids=ids)

    # ⑤ Retriever 반환
    return store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3},
        namespace=namespace
    )
