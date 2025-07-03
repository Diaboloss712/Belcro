import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import Dataset
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    answer_relevancy,
    faithfulness,
)
import config as CFG
import numpy as np

UPSTAGE_API_KEY = CFG.UPSTAGE_API_KEY
# ───────────────────────────────────────────────
# 1. 기본 설정
llm = ChatUpstage(api_key=UPSTAGE_API_KEY)
parser = StrOutputParser()

bootstrap_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert on the Bootstrap framework. "
        "Answer the user's question based only on Bootstrap components. "
        "Avoid generic web development answers. Be concise and document-based."
    ),
    ("user", "{question}")
])
chain = bootstrap_prompt | llm | parser

questions = [
    'What are the required classes to create a Bootstrap card with a title and body?',
    'How can I add an image to the top of a Bootstrap card?',
    'What classes are used to add a brand logo and menu in Bootstrap navbar?',
    'How do I make a responsive card layout using Bootstrap grid system?',
    'Which classes control the card spacing and shadow in Bootstrap?',
    'How do I create a primary Bootstrap button with an icon?',
    'What are the available Bootstrap button variants and their classes?',
    'Which Bootstrap utility classes can I use to add margin and padding?',
    'How can I create a button group with toggle functionality?'
]

# ───────────────────────────────────────────────
# 2. Dataset 생성
def build_ragas_dataset(questions: list[str], retriever, chain) -> Dataset:
    records = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for q in tqdm(questions):
        answer = chain.invoke({"question": q})
        docs = retriever.invoke(q)
        context = [doc.page_content for doc in docs if doc.page_content]
        records["question"].append(q)
        records["answer"].append(answer)
        records["contexts"].append(context)
        records["ground_truth"].append("")
    return Dataset.from_dict(records)

# ───────────────────────────────────────────────
# 3. 평가 공통 함수
def evaluate_dataset(name: str, retriever, embedding) -> pd.Series:
    dataset = build_ragas_dataset(questions, retriever, chain)
    result = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,
            answer_relevancy,
            faithfulness,
        ],
        llm=llm,
        embeddings=embedding,
    )
    df_result = result.to_pandas()
    score = df_result[["context_precision", "context_recall", "answer_relevancy", "faithfulness"]].mean()   
    return score

# ───────────────────────────────────────────────
# 4. Embedding 모델 비교
def run_embedding_comparison(docs: list[Document]) -> pd.DataFrame:
    embeddings = {
        "e5-large": HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2"),
        "bge-large": HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5"),
        "upstage": UpstageEmbeddings(api_key=UPSTAGE_API_KEY, model="embedding-query"),
    }

    results = []
    for name, emb in embeddings.items():
        print(f"\n🔍 [Embedding 비교] {name}")
        vs = FAISS.from_documents(docs, embedding=emb)
        retriever = vs.as_retriever(search_kwargs={"k": 4})
        score = evaluate_dataset(name, retriever, emb)
        results.append(score)

    df = pd.DataFrame(results)
    df.to_csv("embedding_comparison.csv")
    return df

# ───────────────────────────────────────────────
# 5. Retriever 구조 비교
def run_retriever_comparison(docs: list[Document]) -> pd.DataFrame:
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    vs = FAISS.from_documents(docs, embedding=embedding)

    retrievers = {
        "dense": vs.as_retriever(search_kwargs={"k": 4}),
        "bm25": BM25Retriever.from_documents(docs),
        "ensemble": EnsembleRetriever(
            retrievers=[
                vs.as_retriever(search_kwargs={"k": 4}),
                BM25Retriever.from_documents(docs)
            ],
            weights=[0.5, 0.5],
        ),
    }

    results = []
    for name, retr in retrievers.items():
        print(f"\n🔍 [Retriever 비교] {name}")
        score = evaluate_dataset(name, retr, embedding)
        results.append(score)

    df = pd.DataFrame(results)
    df.to_csv("retriever_comparison.csv")
    return df

# ───────────────────────────────────────────────
# 6. 시각화 함수
def plot_results(df: pd.DataFrame, title: str, filename: str):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title(title)
    plt.savefig(filename)
    plt.close()

# ───────────────────────────────────────────────

# ───────────────────────────────
# 6. 시각화 함수
def plot_embedding_metrics_bar(df: pd.DataFrame, title: str, filename: str):
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ["context_precision", "context_recall", "answer_relevancy", "faithfulness"]
    method_labels = {
        "e5-large": "E5-Large",
        "bge-large": "BGE-Large",
        "upstage": "Upstage"
    }
    methods = [method_labels.get(m, m) for m in df.index.tolist()]
    bar_width = 0.2
    index = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        values = df[metric].values
        ax.bar(index + i * bar_width, values, bar_width, label=metric)

    ax.set_xlabel('Embedding Models')
    ax.set_ylabel('Scores')
    ax.set_title(title)
    ax.set_xticks(index + bar_width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(methods)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_retriever_precision_recall_bar(df: pd.DataFrame, title: str, filename: str):
    import matplotlib.pyplot as plt
    import numpy as np

    method_labels = {
        "dense": "Dense",
        "bm25": "Sparse",
        "ensemble": "Ensemble"
    }
    methods = [method_labels.get(m, m) for m in df.index.tolist()]
    precision = df["context_precision"].values
    recall = df["context_recall"].values

    bar_width = 0.35
    index = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(index, precision, bar_width, label='Precision')
    ax.bar(index + bar_width, recall, bar_width, label='Recall')

    ax.set_xlabel('Retrievers')
    ax.set_ylabel('Scores')
    ax.set_title(title)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(methods)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



# 7. 실행
if __name__ == "__main__":
    df = pd.read_parquet("data/docs_5.3.parquet")
    docs = [Document(page_content=row.page_content, metadata=row.metadata) for _, row in df.iterrows()]

    df_embed = run_embedding_comparison(docs)
    df_retriever = run_retriever_comparison(docs)

    print("\n📊 Embedding 비교:\n", df_embed)
    print("\n📊 Retriever 비교:\n", df_retriever)

    plot_embedding_metrics_bar(df_embed, "Embedding 모델 성능 비교 (RAGAS)", "embedding_bar.png")
    plot_retriever_precision_recall_bar(df_retriever, "Retriever 구조 비교 (RAGAS)", "retriever_bar.png")