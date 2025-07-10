from typing import List
from tqdm import tqdm
import json

from langchain.schema import Document
from langchain_upstage import UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore

import config as CFG

def _chunk_document(doc: Document, chunk_size: int = 1800, overlap: int = 300) -> List[Document]:
    base = doc.page_content.strip()
    struct = ""
    html   = doc.metadata.get("example", "")
    full   = f"{base}\n\nStructure: {struct}\n\nExample:\n{html}"

    chunks, start = [], 0
    while start < len(full):
        end = start + chunk_size
        chunks.append(Document(
            page_content=full[start:end].strip(),
            id=getattr(doc, "id", None),
            metadata=doc.metadata
        ))
        start += chunk_size - overlap
    return chunks

def _sanitize_metadata(md: dict) -> dict:
    result = {}
    for k, v in md.items():
        if isinstance(v, (str, int, float, bool)):
            result[k] = v
        elif isinstance(v, list) and all(isinstance(i, str) for i in v):
            result[k] = v
        else:
            result[k] = json.dumps(v, ensure_ascii=False)
    return result

def embvector(docs: List[Document], *, namespace: str, index_name: str = "upstage-index"):
    chunked_docs = [chunk for doc in docs for chunk in _chunk_document(doc)]

    emb = UpstageEmbeddings(
        model="embedding-query",
        api_key=CFG.UPSTAGE_API_KEY
    )
    store = PineconeVectorStore(
        index_name=index_name,
        embedding=emb,
        namespace=namespace
    )

    batch = 32
    for i in tqdm(range(0, len(chunked_docs), batch), desc=f"Pinecone upsert ({namespace})"):
        docs_batch = [
            Document(
                page_content=doc.page_content,
                id=getattr(doc, "id", None),
                metadata=_sanitize_metadata(doc.metadata)
            )
            for doc in chunked_docs[i:i + batch]
        ]
        store.add_documents(docs_batch)

    return store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3},
        namespace=namespace
    )
