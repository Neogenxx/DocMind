# src/rag_pipeline.py

import os
import fitz
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from src.config import *

load_dotenv()

# ── Load models ───────────────────────────────────────────────
def get_models():
    embed_model = SentenceTransformer(EMBED_MODEL)
    reranker    = CrossEncoder(RERANKER_MODEL)
    llm         = ChatGroq(
        model=LLM_MODEL,
        temperature=0.0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    return embed_model, reranker, llm

# ── Document ingestion ────────────────────────────────────────
def load_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page_num, page in enumerate(doc):
        text += f"\n[Page {page_num+1}]\n{page.get_text()}"
    doc.close()
    return text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)

def build_vectorstore(chunks, embed_model, collection_name="my_docs"):
    embeddings = embed_model.encode(chunks, show_progress_bar=False)
    client = chromadb.Client()
    try:
        client.delete_collection(collection_name)
    except:
        pass
    collection = client.create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection

# ── Retrieval ─────────────────────────────────────────────────
def vector_search(query, collection, embed_model, top_k=10):
    vec = embed_model.encode([query]).tolist()
    res = collection.query(query_embeddings=vec, n_results=top_k)
    return res["ids"][0], res["documents"][0]

def bm25_search(query, all_ids, all_chunks, bm25_index, top_k=10):
    scores  = bm25_index.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [all_ids[i] for i in top_idx], [all_chunks[i] for i in top_idx]

def rrf_fusion(v_ids, b_ids, k=60):
    scores = {}
    for rank, cid in enumerate(v_ids):
        scores[cid] = scores.get(cid, 0) + 1/(k+rank+1)
    for rank, cid in enumerate(b_ids):
        scores[cid] = scores.get(cid, 0) + 1/(k+rank+1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def hybrid_search(query, collection, embed_model, all_ids, all_chunks,
                  bm25_index, top_k=10):
    v_ids, v_chunks = vector_search(query, collection, embed_model, top_k)
    b_ids, b_chunks = bm25_search(query, all_ids, all_chunks, bm25_index, top_k)
    fused = rrf_fusion(v_ids, b_ids)
    id_to_chunk = {cid: c for cid, c in zip(v_ids+b_ids, v_chunks+b_chunks)}
    return [
        {"id": cid, "text": id_to_chunk[cid]}
        for cid, _ in fused[:top_k]
        if cid in id_to_chunk
    ]

def rerank_chunks(query, candidates, reranker, top_k=3):
    pairs  = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

# ── Full pipeline ─────────────────────────────────────────────
def rag_answer(query, collection, embed_model, reranker, llm,
               all_ids, all_chunks, bm25_index,
               top_k=TOP_K_FINAL, threshold=GUARD_THRESHOLD):

    candidates = hybrid_search(
        query, collection, embed_model,
        all_ids, all_chunks, bm25_index
    )
    top = rerank_chunks(query, candidates, reranker, top_k)

    best_score = top[0]["rerank_score"] if top else -999

    # Confidence level
    if best_score >= 5.0:   confidence = "high"
    elif best_score >= 0.0: confidence = "medium"
    elif best_score >= threshold: confidence = "low"
    else: confidence = "none"

    if confidence == "none":
        return (
            "I don't have enough information in the provided documents to answer this.",
            top, confidence
        )

    context  = "".join([f"[{r['id']}]\n{r['text']}\n\n" for r in top])
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n---\nQuestion: {query}\nAnswer:")
    ]
    response = llm.invoke(messages)
    return response.content, top, confidence