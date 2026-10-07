# अनुच्छेद (Anuchhed) — Self-RAG Assistant for the Constitution of India

A retrieval-augmented, self-critiquing question-answering app over the
Constitution of India. Originally prototyped in a Jupyter notebook
(`notebook/langchain_rag_constitution.ipynb`), refactored here into a
deployable **FastAPI backend + static web frontend**.

> Ask a question → the pipeline retrieves the relevant Articles with FAISS,
> drafts a strict `Section: / Answer:` response with a local LLM, **critiques
> its own answer** on faithfulness / completeness / relevance / safety, and
> accepts, revises, or declines to answer based on that critique.

![status](https://img.shields.io/badge/status-active-brightgreen) ![python](https://img.shields.io/badge/python-3.11-blue) ![license](https://img.shields.io/badge/license-MIT-lightgrey)

---

## How it works

| Stage | What happens |
|---|---|
| 1. Load & chunk | `PyPDFLoader` loads your source PDF; `RecursiveCharacterTextSplitter` splits it into ~1500-char overlapping chunks |
| 2. Embed & index | `sentence-transformers/all-MiniLM-L6-v2` embeds each chunk; FAISS (cosine similarity) indexes them |
| 3. Retrieve | Top-k similarity search (k=4), with an automatic wider-k retry (k=8) if nothing is found |
| 4. Generate | A local instruction model (`Qwen/Qwen2.5-1.5B-Instruct`) drafts a two-line `Section:` / `Answer:` response, grounded only in the retrieved context |
| 5. Self-critique | The same model scores its own draft 0–5 on faithfulness, completeness, relevance, and safety |
| 6. Decide | Pure Python logic (not the LLM) turns those scores into **ACCEPT**, **REVISE** (ask the model to correct itself once), or **INSUFFICIENT** (decline rather than hallucinate) |

This logic is unchanged from the research notebook — it's simply been moved
into `backend/rag_pipeline.py` so a web server can call it.

---

## Project structure

```
constitution-rag-chatbot/
├── backend/
│   ├── app.py              # FastAPI app: /api/health, /api/ask, serves the frontend
│   ├── rag_pipeline.py      # Load/embed/index/retrieve/generate/critique logic
│   └── __init__.py
├── frontend/
│   ├── index.html           # Chat UI
│   ├── style.css
│   └── script.js
├── data/
│   ├── constitution.pdf      # <-- put your source PDF here (not included)
│   └── faiss_index/          # auto-generated on first run, then cached
├── notebook/
│   └── langchain_rag_constitution.ipynb   # original research notebook
├── requirements.txt
├── render.yaml               # one-click Render.com deploy config
├── Procfile                  # Railway/Heroku-style start command
└── README.md
```

---

## Running locally

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/constitution-rag-chatbot.git
cd constitution-rag-chatbot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your source PDF

Place your Constitution of India PDF at:

```
data/constitution.pdf
```

(Or point `RAG_PDF_PATH` at any other PDF — see **Configuration** below.)

### 3. Start the server

```bash
uvicorn backend.app:app --reload --port 8000
```

Open **http://localhost:8000** — the FastAPI app serves both the API and the
chat UI from a single process.

The first request after startup will download the embedding + LLM models
(a few GB) and build the FAISS index, then cache the index under
`data/faiss_index/` so subsequent restarts skip re-indexing. Watch the
terminal logs, or poll `GET /api/health` — the UI's status dot does this for
you automatically.

---

## API

### `POST /api/ask`

```json
{ "question": "What are the Fundamental Duties?", "verbose": false }
```

**Response**

```json
{
  "answer": "Section: Article 51A\nAnswer: The Fundamental Duties are moral obligations of citizens, including respecting the Constitution and protecting the environment.",
  "decision": "ACCEPT",
  "scores": { "faithfulness": 5, "completeness": 4, "relevance": 5, "safety": 5, "raw": "..." },
  "sources": ["page 12", "page 13"]
}
```

`decision` is one of `ACCEPT`, `REVISE`, `REVISE_FAILED`, `INSUFFICIENT`,
`NO_CONTEXT`, `UNGRADED`, or `ERROR`.

### `GET /api/health`

```json
{ "ready": true, "error": null }
```

---

## Configuration

All optional, set as environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `RAG_PDF_PATH` | `data/constitution.pdf` | Source PDF to index |
| `RAG_INDEX_DIR` | `data/faiss_index` | Where the FAISS index is cached |
| `RAG_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `RAG_LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Generation/critique model |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `1500` / `200` | Text splitting |
| `RAG_RETRIEVER_K` / `RAG_RETRIEVER_K_WIDE` | `4` / `8` | Retrieval breadth |
| `RAG_MAX_NEW_TOKENS` | `120` | LLM generation length |

---

## Deploying

This is a single FastAPI process (API + static frontend together), so it
deploys anywhere that runs a Python web service.

**Important:** `Qwen2.5-1.5B-Instruct` on CPU comfortably needs **4+ GB of
RAM** once PyTorch and the model weights are loaded. Free tiers on some
platforms (e.g. Render's free 512 MB plan) will not have enough memory —
pick a plan/platform with more RAM, or swap in a smaller model via
`RAG_LLM_MODEL`.

**Recommended: Hugging Face Spaces (Docker/FastAPI SDK)** — generous free
CPU RAM, and it's the same ecosystem the models already live in.

**Also works:** Render (paid instance or higher-RAM plan) using the included
`render.yaml`, Railway using the included `Procfile`, or any VM (`uvicorn
backend.app:app --host 0.0.0.0 --port $PORT`).

Either way:
1. Commit your PDF (or fetch it at build time) so `data/constitution.pdf`
   exists on the deployed instance — the `.gitignore` currently excludes
   PDFs and the generated index by default; remove that line if you want to
   version-control your source PDF directly.
2. Set any environment variables from the table above.
3. Deploy — the server builds the FAISS index once and caches it.

---

## Notes & limitations

- Answers are generated **only** from the indexed PDF's text — the model is
  explicitly instructed not to use outside knowledge, and declines when the
  retrieved context doesn't cover the question.
- This is **not legal advice**. It's a document-grounded Q&A tool.
- The self-critique step uses the *same* small local model that generated
  the answer, which is a useful hallucination check but not as reliable as
  a larger or independent judge model.

## License

MIT — see `LICENSE`.
