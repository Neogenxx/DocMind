"""
DocMind — Production RAG System
Streamlit UI — Day 3

Run with:
    cd docmind
    streamlit run app/app.py
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind",
    page_icon="◈",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #888;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f8f8;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
    }
    .chunk-box {
        background: #f0f4ff;
        border-left: 3px solid #4a6cf7;
        padding: 0.8rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .guard-box {
        background: #fff3f0;
        border-left: 3px solid #ff4444;
        padding: 0.8rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
    }
    .answer-box {
        background: #f0fff4;
        border-left: 3px solid #00aa44;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
    .confidence-high  { color: #00aa44; font-weight: 600; }
    .confidence-med   { color: #ff8800; font-weight: 600; }
    .confidence-low   { color: #ff4444; font-weight: 600; }
    .confidence-none  { color: #cc0000; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Load models once ──────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker    = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm         = ChatGroq(
        model="llama-3.1-8b-instant",   # updated model name
        temperature=0.0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    return embed_model, reranker, llm

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are DocMind, a precise document assistant.
Answer using ONLY the provided context chunks. Be concise and direct.

Rules:
1. Use ONLY information from the provided context. Never use outside knowledge.
2. If context is insufficient, say exactly:
   "I don't have enough information in the provided documents to answer this."
3. End every answer with: Sources: [chunk_id_1, chunk_id_2]
   List each unique chunk ID only once.
4. No filler phrases. Go straight to the answer."""

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">◈ DocMind</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Production RAG — hybrid retrieval · cross-encoder reranking · hallucination guard</div>',
    unsafe_allow_html=True
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help="Upload any PDF — textbook, paper, notes"
    )

    st.divider()
    st.header("Settings")

    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1, max_value=5, value=3,
        help="How many text chunks to pass to the LLM"
    )

    confidence_threshold = st.slider(
        "Guard threshold",
        min_value=-10.0, max_value=10.0, value=-5.0, step=0.5,
        help="Reranker score below this → system declines to answer"
    )

    show_chunks = st.toggle("Show retrieved chunks", value=True)
    show_scores = st.toggle("Show relevance scores", value=True)

    st.divider()
    st.markdown("**Built with:**")
    st.markdown("- LangChain + Groq (LLaMA-3)")
    st.markdown("- ChromaDB + BM25 hybrid search")
    st.markdown("- Cross-encoder reranking")
    st.markdown("- Ragas evaluation")

# ── Stop if no file ───────────────────────────────────────────────────────────
if not uploaded_file:
    st.info("Upload a PDF in the sidebar to get started.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Hybrid retrieval")
        st.markdown("Combines BM25 keyword search with semantic vector search for best-of-both results.")
    with col2:
        st.markdown("### Cross-encoder reranking")
        st.markdown("Rescores retrieved chunks with a cross-encoder model, boosting precision significantly.")
    with col3:
        st.markdown("### Hallucination guard")
        st.markdown("Declines to answer when evidence is insufficient — no fabricated responses.")
    st.stop()

# ── Process PDF ───────────────────────────────────────────────────────────────
@st.cache_data
def process_pdf(file_bytes, filename):
    """Load, chunk, embed, store. Returns collection + metadata."""

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    doc = fitz.open(tmp_path)
    full_text = ""
    page_count = len(doc)
    for page_num, page in enumerate(doc):
        full_text += f"\n[Page {page_num + 1}]\n{page.get_text()}"
    doc.close()
    os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(full_text)

    embed_model, _, _ = load_models()
    embeddings = embed_model.encode(chunks, show_progress_bar=False)

    client = chromadb.Client()
    col_name = f"doc_{abs(hash(filename)) % 100000}"
    try:
        client.delete_collection(col_name)
    except:
        pass
    collection = client.create_collection(
        col_name,
        metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    return collection, chunks, len(chunks), page_count, len(full_text)


with st.spinner("Processing PDF — embedding chunks..."):
    embed_model, reranker, llm = load_models()
    collection, all_chunks, n_chunks, n_pages, n_chars = process_pdf(
        uploaded_file.read(), uploaded_file.name
    )

# ── Build BM25 ────────────────────────────────────────────────────────────────
all_data   = collection.get(include=["documents"])
doc_chunks = all_data["documents"]
doc_ids    = all_data["ids"]
tokenised  = [c.lower().split() for c in doc_chunks]
bm25_index = BM25Okapi(tokenised)

# ── Stats row ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Document", uploaded_file.name[:25] + "..." if len(uploaded_file.name) > 25 else uploaded_file.name)
col2.metric("Pages", n_pages)
col3.metric("Chunks indexed", n_chunks)
col4.metric("Characters", f"{n_chars:,}")

st.divider()

# ── Retrieval functions ───────────────────────────────────────────────────────
def hybrid_search(query, top_k=10):
    # Vector search
    vec = embed_model.encode([query]).tolist()
    res = collection.query(query_embeddings=vec, n_results=min(top_k, len(doc_chunks)))
    v_ids    = res["ids"][0]
    v_chunks = res["documents"][0]

    # BM25 search
    scores   = bm25_index.get_scores(query.lower().split())
    top_idx  = np.argsort(scores)[::-1][:top_k]
    b_ids    = [doc_ids[i] for i in top_idx]
    b_chunks = [doc_chunks[i] for i in top_idx]

    # RRF fusion
    rrf = {}
    for rank, cid in enumerate(v_ids):
        rrf[cid] = rrf.get(cid, 0) + 1 / (60 + rank + 1)
    for rank, cid in enumerate(b_ids):
        rrf[cid] = rrf.get(cid, 0) + 1 / (60 + rank + 1)

    id_to_chunk = {cid: c for cid, c in zip(v_ids + b_ids, v_chunks + b_chunks)}
    fused = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    return [
        {"id": cid, "text": id_to_chunk[cid]}
        for cid, _ in fused[:top_k]
        if cid in id_to_chunk
    ]


def answer_query(query, top_k=3, threshold=-5.0):
    candidates = hybrid_search(query, top_k=10)
    if not candidates:
        return "No relevant content found.", [], "none"

    pairs  = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])

    top = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

    best_score = top[0]["rerank_score"]

    # Confidence level
    if best_score >= 5.0:
        confidence = "high"
    elif best_score >= 0.0:
        confidence = "medium"
    elif best_score >= threshold:
        confidence = "low"
    else:
        confidence = "none"

    if confidence == "none":
        answer = "I don't have enough information in the provided documents to answer this."
        return answer, top, confidence

    context = "".join([f"[{r['id']}]\n{r['text']}\n\n" for r in top])
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n---\nQuestion: {query}\nAnswer:")
    ]
    response = llm.invoke(messages)
    answer = response.content

    if confidence == "low":
        answer += "\n\n⚠️ Low confidence — retrieved context has weak relevance. Verify manually."

    return answer, top, confidence


# ── Query interface ───────────────────────────────────────────────────────────
query = st.text_input(
    "Ask a question about your document",
    placeholder="e.g. What is the Banker's algorithm?",
    label_visibility="collapsed"
)

col_ask, col_clear = st.columns([6, 1])
with col_ask:
    ask_clicked = st.button("Ask DocMind ↗", type="primary", use_container_width=True)

# ── Suggested questions ───────────────────────────────────────────────────────
st.markdown("**Try asking:**")
example_cols = st.columns(4)
examples = [
    "What is the main topic?",
    "Explain a key concept",
    "What are the types discussed?",
    "What is X algorithm?"
]
for i, (col, ex) in enumerate(zip(example_cols, examples)):
    with col:
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            query = ex
            ask_clicked = True

# ── Run query ─────────────────────────────────────────────────────────────────
if (ask_clicked or query) and query.strip():
    with st.spinner("Retrieving · Reranking · Generating..."):
        answer, retrieved, confidence = answer_query(
            query.strip(),
            top_k=top_k,
            threshold=confidence_threshold
        )

    # Confidence badge
    badge_map = {
        "high":   ("confidence-high",  "HIGH confidence"),
        "medium": ("confidence-med",   "MEDIUM confidence"),
        "low":    ("confidence-low",   "LOW confidence"),
        "none":   ("confidence-none",  "INSUFFICIENT context"),
    }
    css_class, label = badge_map.get(confidence, ("confidence-none", "UNKNOWN"))
    st.markdown(
        f'<span class="{css_class}">● {label}</span>',
        unsafe_allow_html=True
    )

    # Answer
    st.markdown("#### Answer")
    if confidence == "none":
        st.markdown(f'<div class="guard-box">{answer}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    # Retrieved chunks
    if show_chunks and retrieved:
        st.markdown("#### Retrieved chunks")
        for i, r in enumerate(retrieved):
            score_str = f" — rerank score: `{r['rerank_score']:.3f}`" if show_scores else ""
            with st.expander(f"Chunk {i+1}  ·  {r['id']}{score_str}"):
                st.text(r["text"][:800] + ("..." if len(r["text"]) > 800 else ""))

    st.divider()

# ── History ───────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if (ask_clicked or query) and query.strip():
    st.session_state.history.append({
        "q": query.strip(),
        "a": answer,
        "conf": confidence
    })

if st.session_state.history:
    with st.expander(f"Query history ({len(st.session_state.history)} questions)"):
        for item in reversed(st.session_state.history[-10:]):
            badge = {"high": "🟢", "medium": "🟡", "low": "🔴", "none": "⛔"}.get(item["conf"], "⚪")
            st.markdown(f"{badge} **Q:** {item['q']}")
            st.markdown(f"**A:** {item['a'][:200]}{'...' if len(item['a']) > 200 else ''}")
            st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<br><center><small>DocMind · Production RAG · Built for ML/GenAI internship portfolio</small></center>",
    unsafe_allow_html=True
)