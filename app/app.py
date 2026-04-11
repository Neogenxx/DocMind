"""
DocMind — Production RAG System
Streamlit UI — Day 3

Run with: streamlit run app/app.py
"""

import streamlit as st
import chromadb
import numpy as np
import os
import tempfile
import fitz
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind",
    page_icon="◈",
    layout="wide"
)

st.title("◈ DocMind")
st.caption("Production RAG — Ask questions grounded in your documents")

# ── Load models (cached so they don't reload on every interaction) ───────────
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker    = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm         = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    return embed_model, reranker, llm

embed_model, reranker, llm = load_models()

SYSTEM_PROMPT = """You are DocMind, a precise document assistant.
Answer using ONLY the provided context chunks. Be concise and direct.
Rules:
1. Use only the provided context — never outside knowledge.
2. If context is insufficient, say exactly: "I don't have enough information in the provided documents to answer this."
3. End every answer with: Sources: [chunk_id]"""


# ── Sidebar — PDF upload ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file:
        st.success(f"Loaded: {uploaded_file.name}")

    st.divider()
    st.header("Settings")
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=5, value=3)
    show_chunks = st.toggle("Show retrieved chunks", value=True)
    show_scores = st.toggle("Show confidence scores", value=True)

    st.divider()
    st.caption("Built with LangChain · ChromaDB · Groq · Sentence Transformers · Ragas")


# ── Process uploaded PDF ──────────────────────────────────────────────────────
@st.cache_data
def process_pdf(file_bytes, filename):
    """Load, chunk, embed and store PDF. Returns collection name."""

    # Write to temp file for PyMuPDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Extract text
    doc = fitz.open(tmp_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    os.unlink(tmp_path)

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(full_text)

    # Embed
    embeddings = embed_model.encode(chunks, show_progress_bar=False)

    # Store in ChromaDB (in-memory for the app session)
    client = chromadb.Client()
    col_name = f"doc_{hash(filename) % 100000}"
    try:
        client.delete_collection(col_name)
    except:
        pass
    collection = client.create_collection(col_name, metadata={"hnsw:space": "cosine"})
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    return collection, chunks, len(chunks)


# ── Main area ─────────────────────────────────────────────────────────────────
if not uploaded_file:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

# Process PDF
with st.spinner("Processing PDF..."):
    collection, all_chunks, n_chunks = process_pdf(
        uploaded_file.read(), uploaded_file.name
    )

st.success(f"Ready — {n_chunks} chunks indexed from {uploaded_file.name}")

# Build BM25 index
all_data   = collection.get(include=["documents"])
doc_chunks = all_data["documents"]
doc_ids    = all_data["ids"]
tokenised  = [c.lower().split() for c in doc_chunks]
bm25_index = BM25Okapi(tokenised)


def hybrid_search_app(query, top_k=10):
    # Vector
    vec = embed_model.encode([query]).tolist()
    res = collection.query(query_embeddings=vec, n_results=top_k)
    v_ids, v_chunks = res["ids"][0], res["documents"][0]

    # BM25
    scores = bm25_index.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:top_k]
    b_ids   = [doc_ids[i] for i in top_idx]
    b_chunks = [doc_chunks[i] for i in top_idx]

    # RRF
    rrf = {}
    for rank, cid in enumerate(v_ids):
        rrf[cid] = rrf.get(cid, 0) + 1/(60+rank+1)
    for rank, cid in enumerate(b_ids):
        rrf[cid] = rrf.get(cid, 0) + 1/(60+rank+1)

    id_to_chunk = {cid: c for cid, c in zip(v_ids+b_ids, v_chunks+b_chunks)}
    fused = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    return [{\"id\": cid, \"text\": id_to_chunk[cid]} for cid, _ in fused[:top_k] if cid in id_to_chunk]


def answer_query(query, top_k=3):
    candidates = hybrid_search_app(query, top_k=10)
    pairs = [(query, c["text"]) for c in candidates]
    rerank_scores = reranker.predict(pairs)
    for i, c in enumerate(candidates):
        c["rerank_score"] = float(rerank_scores[i])
    top = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

    if top[0]["rerank_score"] < -5.0:
        return "I don't have enough information in the provided documents to answer this.", top, "none"

    context = "".join([f"[{r['id']}]\n{r['text']}\n\n" for r in top])
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n---\nQuestion: {query}\nAnswer:")
    ]
    response = llm.invoke(messages)
    confidence = "high" if top[0]["rerank_score"] > 5 else "medium" if top[0]["rerank_score"] > 0 else "low"
    return response.content, top, confidence


# ── Query interface ───────────────────────────────────────────────────────────
query = st.text_input(
    "Ask a question about your document",
    placeholder="What is the main topic of this document?"
)

if query:
    with st.spinner("Thinking..."):
        answer, retrieved, confidence = answer_query(query, top_k=top_k)

    # Confidence badge
    badge_color = {"high": "green", "medium": "orange", "low": "red", "none": "red"}
    st.markdown(
        f"**Confidence:** :{badge_color.get(confidence, 'gray')}[{confidence.upper()}]"
    )

    # Answer
    st.markdown("### Answer")
    st.markdown(answer)

    # Retrieved chunks
    if show_chunks:
        st.markdown("### Retrieved chunks")
        for i, r in enumerate(retrieved):
            score_display = f"{r['rerank_score']:.4f}" if show_scores else ""
            with st.expander(f"Chunk {i+1} — {r['id']}  {score_display}"):
                st.text(r["text"])

    st.divider()


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center; color: gray; font-size: 12px;'>"
    "DocMind · Hybrid RAG · LLaMA-3 via Groq · Built for ML/GenAI internship portfolio"
    "</div>",
    unsafe_allow_html=True
)
