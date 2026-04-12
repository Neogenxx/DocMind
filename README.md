# DOCKNOWS

A production-grade Retrieval Augmented Generation (RAG) system built from scratch. By implementing hybrid retrieval, cross-encoder reranking, hallucination guards, and Ragas-based evaluation.

---

## What it does

Upload any PDF and ask questions in plain English. The system retrieves the most relevant chunks from your document, passes them to an LLM, and returns a grounded answer with source citations. If the retrieved context isn't strong enough to answer confidently, the system says so no hallucination.

---

## Architecture

```
PDF / Markdown
      │
      ▼
┌─────────────────────┐
│   Document Ingestion │  PyMuPDF + RecursiveCharacterTextSplitter
│   500-800 token      │  100 token overlap at boundaries
│   chunks             │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    Vector Store      │  sentence-transformers (all-MiniLM-L6-v2)
│    ChromaDB          │  384-dim embeddings, persisted to disk
└────────┬────────────┘
         │
    User Query
         │
         ▼
┌─────────────────────────────────────────┐
│           Hybrid Retrieval               │
│  BM25 keyword search  +  Vector search  │
│  (exact term match)   (semantic intent) │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Cross-Encoder       │  ms-marco-MiniLM-L-6-v2
│  Reranker            │  rescores top-10 → top-3
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Citation Enforcement│  declines if evidence score < threshold
│  + Hallucination     │  forces explicit source citation
│  Guard               │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Groq LLM           │  LLaMA-3 (free API)
│   (LLaMA-3)          │  grounded answer generation
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Streamlit UI       │  upload → ask → cited answer
└─────────────────────┘
```

---

## Evaluation Results

| Metric | Score |
|---|---|
| Faithfulness | — |
| Answer Relevancy | — |
| Context Precision | — |
| Golden Dataset Size | 20 QA pairs |
| Evaluation Framework | Ragas |


---

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| PDF parsing | PyMuPDF | Fast, handles complex PDFs |
| Chunking | LangChain text splitters | Recursive, context-aware |
| Embeddings | sentence-transformers | Free, runs locally, no API needed |
| Vector DB | ChromaDB | Local persistent storage |
| Keyword search | rank-bm25 | Exact term matching |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Boosts precision significantly |
| LLM | Groq API (LLaMA-3) | Free tier, fast inference |
| Evaluation | Ragas | Industry-standard RAG metrics |
| UI | Streamlit | Fast to build, easy to demo |

---

## Project Structure

```
production-rag/
│
├── notebooks/
│   ├── 01_document_ingestion.ipynb        # load, chunk, embed PDFs
│   ├── 02_vector_store_retrieval.ipynb    # ChromaDB storage + top-K search
│   ├── 03_groq_rag_pipeline.ipynb         # connect Groq LLM, generate answers
│   ├── 04_hybrid_retrieval.ipynb          # BM25 + vector hybrid search
│   ├── 05_reranker.ipynb                  # cross-encoder reranking
│   ├── 06_citation_hallucination_guard.ipynb  # grounding + refusal logic
│   └── 07_ragas_evaluation.ipynb          # faithfulness + relevancy scoring
│
├── docs/                                  # put your PDF files here
│   └── sample.pdf
│
├── data/
│   ├── golden_dataset.csv                 # 20 manually verified QA pairs
│   └── ragas_results.json                 # auto-generated evaluation scores
│
├── app/
│   └── app.py                             # Streamlit UI
│
├── requirements.txt
├── .env                                   # GROQ_API_KEY (never commit this)
├── .gitignore
└── README.md
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/production-rag.git
cd production-rag
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Get your free Groq API key**

Sign up at [console.groq.com](https://console.groq.com) → API Keys → Create Key. It's free.

**4. Add your key to `.env`**
```
GROQ_API_KEY=your_key_here
```

**5. Add a PDF to the `docs/` folder**

Any PDF works — research papers, textbooks, your college notes.

**6. Run the notebooks in order**
```bash
jupyter notebook
```
Open `notebooks/01_document_ingestion.ipynb` and run each notebook sequentially.

**7. Launch the Streamlit app**
```bash
streamlit run app/app.py
```

---

## Key Features Explained

### Hybrid Retrieval
Pure vector search misses exact keyword matches (like model names, author names, specific terms). Pure BM25 misses semantic meaning. This system combines both with a weighted merge, then reranks — getting the best of both worlds.

### Cross-Encoder Reranking
The initial retriever returns top-10 candidates cheaply. A cross-encoder then reads each chunk alongside the query and produces a precise relevancy score. Only the top-3 reranked chunks reach the LLM — significantly improving answer quality.

### Hallucination Guard
If no retrieved chunk scores above the relevancy threshold, the system responds with "I don't have enough information in the provided documents to answer this confidently" rather than fabricating an answer. This is what separates a production system from a demo.

### Ragas Evaluation
A golden dataset of 20 manually verified question-answer pairs is used to measure:
- **Faithfulness** — is every claim in the answer supported by the retrieved chunks?
- **Answer Relevancy** — does the answer actually address the question?
- **Context Precision** — were the retrieved chunks actually relevant?

---

## Running the Evaluation

```python
# Inside 07_ragas_evaluation.ipynb
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

results = evaluate(
    dataset=golden_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
print(results)
```

---

## What I Learned

Building this project beyond a basic demo required understanding:

- Why chunking strategy matters chunk too large and retrieval loses precision, too small and you lose context
- Why hybrid search outperforms pure vector search on technical documents
- Why cross-encoder reranking exists bi-encoders are fast but imprecise, cross-encoders are slow but accurate, so you combine them
- Why citation enforcement matters an LLM that says "I don't know" is more trustworthy than one that hallucinates confidently
- How to measure RAG quality quantitatively with Ragas rather than just eyeballing answers

---

## References

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Ragas Documentation](https://docs.ragas.io/)
- [Cohere Rerank Guide](https://docs.cohere.com/docs/rerank-guide)
- [sentence-transformers](https://www.sbert.net/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Groq API](https://console.groq.com/)

