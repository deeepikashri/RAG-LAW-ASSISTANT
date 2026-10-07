"""
rag_pipeline.py
----------------
Self-RAG pipeline for the Constitution of India, extracted from the original
research notebook (langchain_rag_cons.ipynb) and refactored into an
importable module that a web backend can call.

Pipeline stages (unchanged from the notebook):
1. Load a PDF and split it into overlapping chunks.
2. Embed chunks with a sentence-transformers model and index them in FAISS.
3. Retrieve top-k chunks for a question (with a wider-k retry if empty).
4. Generate a strict "Section / Answer" formatted response with a local LLM.
5. Critique the answer (faithfulness / completeness / relevance / safety).
6. Accept, revise, or reject the answer based on the critique scores.

The heavy objects (embeddings, FAISS index, LLM pipeline) are built once and
cached as module-level singletons so a web server can reuse them across
requests instead of reloading models on every call.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("self_rag")

# ---------------------------------------------------------------------------
# Configuration (overridable via environment variables so this works the same
# on a laptop and on a deployed server)
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", BASE_DIR / "data"))
PDF_PATH = Path(os.environ.get("RAG_PDF_PATH", DATA_DIR / "constitution.pdf"))
INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR", DATA_DIR / "faiss_index"))

EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", 1500))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", 200))
RETRIEVER_K = int(os.environ.get("RAG_RETRIEVER_K", 4))
RETRIEVER_K_WIDE = int(os.environ.get("RAG_RETRIEVER_K_WIDE", 8))
MAX_NEW_TOKENS = int(os.environ.get("RAG_MAX_NEW_TOKENS", 120))

ANSWER_FORMAT_RE = re.compile(r"Section\s*:\s*.+\n\s*Answer\s*:\s*.+", re.IGNORECASE)

INSUFFICIENT_CONTEXT_MSG = (
    "The provided Constitution of India documents do not contain "
    "sufficient information to answer this question."
)
GENERIC_ERROR_MSG = "Something went wrong while answering this question. Please try again."

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_db = None
_retriever = None
_pipe = None
_ready = False
_init_error: Optional[str] = None


def is_ready() -> bool:
    return _ready


def init_error() -> Optional[str]:
    return _init_error


def _build_or_load_index(embeddings):
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.faiss import DistanceStrategy

    if INDEX_DIR.exists() and any(INDEX_DIR.iterdir()):
        logger.info("Loading cached FAISS index from %s", INDEX_DIR)
        return FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"No source PDF found at {PDF_PATH}. Place your Constitution PDF "
            f"there (or set RAG_PDF_PATH) before starting the server."
        )

    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    logger.info("Building FAISS index from %s", PDF_PATH)
    loader = PyPDFLoader(str(PDF_PATH))
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)
    logger.info("Loaded %d pages -> %d chunks", len(docs), len(chunks))

    db = FAISS.from_documents(
        chunks, embeddings, distance_strategy=DistanceStrategy.COSINE, normalize_L2=True
    )
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    db.save_local(str(INDEX_DIR))
    return db


def init_pipeline() -> None:
    """Build (or load cached) embeddings/index and the local LLM pipeline.
    Safe to call multiple times; only runs once."""
    global _db, _retriever, _pipe, _ready, _init_error

    with _lock:
        if _ready or _init_error:
            return
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

            _db = _build_or_load_index(embeddings)
            _retriever = _db.as_retriever(search_kwargs={"k": RETRIEVER_K})

            from transformers import pipeline as hf_pipeline

            _pipe = hf_pipeline(
                "text-generation",
                model=LLM_MODEL,
                device=-1,  # CPU; set device=0 in the env if a GPU is available
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.2,
                do_sample=False,
                repetition_penalty=1.15,
            )
            _pipe.tokenizer.pad_token_id = _pipe.tokenizer.eos_token_id

            _ready = True
            logger.info("RAG pipeline ready.")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to initialize RAG pipeline")
            _init_error = str(exc)
            raise


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def ask_llm(system_msg: str, user_msg: str) -> str:
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    output = _pipe(messages)
    return output[0]["generated_text"][-1]["content"].strip()


def extract_score(label: str, text: str, default: int = 3) -> int:
    match = re.search(rf"{label}\s*:\s*(\d+)", text, re.IGNORECASE)
    if not match:
        logger.warning(
            "extract_score: could not find '%s' in critique output -- defaulting to %d. Raw: %r",
            label, default, text,
        )
        return default
    return int(match.group(1))


def generate_answer(question: str, context: str) -> str:
    system_msg = (
        "You are a Constitutional Law citation assistant for the Constitution "
        "of India. Answer using the provided context retrieved from the "
        "Constitution of India. The question may be a general/definitional "
        "question (e.g. 'what is a High Court') -- if the context describes "
        "the relevant institution, power, or provision, answer from it even "
        "if the question doesn't quote the Constitution's exact wording.\n\n"
        "Rules:\n"
        "1. Use only the provided context. Do not use outside knowledge.\n"
        "2. Do not invent Articles, clauses, or legal interpretations.\n"
        "3. If the context genuinely does not cover the question, reply exactly:\n"
        "'The provided Constitution of India documents do not contain sufficient information to answer this question.'\n"
        "4. Output format is EXACTLY two lines and nothing else:\n"
        "Section: <Article/clause number as it appears in the context>\n"
        "Answer: <one or two short, plain sentences -- no preamble, no elaboration>\n"
        "5. Do not provide legal advice. Do not speculate."
    )
    user_msg = (
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Respond in the exact two-line 'Section:' / 'Answer:' format described above. Nothing else."
    )
    return ask_llm(system_msg, user_msg)


def enforce_answer_format(question: str, context: str, answer: str) -> str:
    if ANSWER_FORMAT_RE.search(answer):
        return answer

    logger.warning("Output did not match 'Section:'/'Answer:' format, retrying once. Raw: %r", answer)

    retry_system_msg = (
        "Reformat the text below into EXACTLY two lines and nothing else:\n"
        "Section: <Article/clause number>\n"
        "Answer: <one or two short sentences>\n"
        "Do not add commentary. Do not add a preamble. Only output the two lines."
    )
    retry_user_msg = f"Text to reformat:\n{answer}"
    retried = ask_llm(retry_system_msg, retry_user_msg)

    if ANSWER_FORMAT_RE.search(retried):
        return retried

    logger.warning("Retry also failed to match format. Falling back to templated wrapper. Raw: %r", retried)
    return f"Section: Unspecified\nAnswer: {answer.strip()}"


def critique_answer(question: str, context: str, answer: str) -> dict:
    system_msg = (
        "You are a strict evaluator of legal RAG answers. You output ONLY "
        "numeric scores in the exact format requested -- no explanations, "
        "no extra commentary, no repeated questions."
    )
    user_msg = (
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:\n{answer}\n\n"
        "Score the answer from 0 to 5 on each criterion below. Respond in EXACTLY this format, nothing else:\n\n"
        "Faithfulness: <number>\nCompleteness: <number>\nRelevance: <number>\nSafety: <number>"
    )
    critique_text = ask_llm(system_msg, user_msg)
    return {
        "faithfulness": extract_score("Faithfulness", critique_text),
        "completeness": extract_score("Completeness", critique_text),
        "relevance": extract_score("Relevance", critique_text),
        "safety": extract_score("Safety", critique_text),
        "raw": critique_text,
    }


def revise_answer(question: str, context: str, answer: str) -> str:
    system_msg = (
        "You are revising a draft legal citation answer. Correct any "
        "inaccuracies, remove anything not supported by the context, and "
        "keep the EXACT two-line format:\n"
        "Section: <Article/clause number>\n"
        "Answer: <one or two short sentences>\n"
        "Do not add new information beyond the context. Output only the "
        "corrected two lines -- no preamble, no explanation."
    )
    user_msg = (
        f"Context:\n{context}\n\nQuestion: {question}\n\nDraft answer:\n{answer}\n\n"
        "Provide the corrected answer in the exact 'Section:' / 'Answer:' format."
    )
    return ask_llm(system_msg, user_msg)


def retrieve_with_retry(question: str, verbose: bool = False):
    docs = _retriever.invoke(question)
    if docs:
        return docs

    if verbose:
        print("--- No results on first retrieval, retrying with wider k ---")

    try:
        wider_retriever = _db.as_retriever(search_kwargs={"k": RETRIEVER_K_WIDE})
        docs = wider_retriever.invoke(question)
    except Exception:
        logger.exception("retrieve_with_retry: widened retrieval attempt failed")
        docs = []
    return docs


def self_rag_part(question: str, verbose: bool = False) -> dict:
    """Run the full self-RAG loop and return a structured result dict:
    {"answer": str, "decision": str, "scores": dict | None, "sources": list[str]}
    """
    if not _ready:
        raise RuntimeError("Pipeline not initialized. Call init_pipeline() first.")

    try:
        docs = retrieve_with_retry(question, verbose=verbose)
    except Exception:
        logger.exception("self_rag_part: retrieval failed")
        return {"answer": GENERIC_ERROR_MSG, "decision": "ERROR", "scores": None, "sources": []}

    if not docs:
        return {"answer": INSUFFICIENT_CONTEXT_MSG, "decision": "NO_CONTEXT", "scores": None, "sources": []}

    context = "\n\n".join(doc.page_content for doc in docs)
    sources = sorted({f"page {doc.metadata.get('page', '?')}" for doc in docs})

    try:
        answer = generate_answer(question, context)
        answer = enforce_answer_format(question, context, answer)
    except Exception:
        logger.exception("self_rag_part: answer generation failed")
        return {"answer": GENERIC_ERROR_MSG, "decision": "ERROR", "scores": None, "sources": sources}

    try:
        scores = critique_answer(question, context, answer)
    except Exception:
        logger.exception("self_rag_part: critique failed -- returning ungraded answer")
        return {"answer": answer, "decision": "UNGRADED", "scores": None, "sources": sources}

    if verbose:
        print("--- Critique scores ---")
        print(scores)

    if (
        scores["faithfulness"] >= 4
        and scores["completeness"] >= 4
        and scores["relevance"] >= 4
        and scores["safety"] >= 4
    ):
        decision = "ACCEPT"
    elif scores["relevance"] <= 2 or scores["completeness"] <= 2:
        decision = "INSUFFICIENT"
    else:
        decision = "REVISE"

    if decision == "ACCEPT":
        return {"answer": answer, "decision": decision, "scores": scores, "sources": sources}

    if decision == "REVISE":
        try:
            revised = revise_answer(question, context, answer)
            revised = enforce_answer_format(question, context, revised)
            return {"answer": revised, "decision": decision, "scores": scores, "sources": sources}
        except Exception:
            logger.exception("self_rag_part: revision failed -- returning original answer")
            return {"answer": answer, "decision": "REVISE_FAILED", "scores": scores, "sources": sources}

    return {
        "answer": (
            "The retrieved context does not sufficiently answer this "
            "question. Please rephrase or ask about a more specific "
            "provision of the Constitution."
        ),
        "decision": decision,
        "scores": scores,
        "sources": sources,
    }
