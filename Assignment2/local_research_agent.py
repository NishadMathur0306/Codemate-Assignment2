"""
local_research_agent.py

A self-contained Python research agent that:
- ingests local documents (txt, md, pdf)
- generates embeddings locally using sentence-transformers
- stores embeddings in a FAISS index and metadata in SQLite
- retrieves relevant passages for a query
- performs simple multi-step reasoning by decomposing queries
- summarizes retrieved sources into a research report (Markdown)
- exports report to Markdown and PDF (reportlab)

Requirements (install with pip):
    pip install sentence-transformers faiss-cpu sqlalchemy nltk PyPDF2 transformers torch reportlab tqdm

Notes:
- All models run locally. For summarization/advanced reasoning you can point `Reasoner` at a local transformers model (e.g., 'google/flan-t5-small') if you have it downloaded; otherwise the agent will use a lightweight heuristic summarizer.
- This script avoids external web APIs and focuses on offline operation.

Usage example (at bottom of file) demonstrates typical workflow.

"""

import os
import json
import math
import sqlite3
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from dataclasses import dataclass, asdict

# Embedding and NLP
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# PDF and text reading
import PyPDF2

# Simple summarization / local model (optional)
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# For PDF export
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# For progress
from tqdm import tqdm

# For simple tokenization and chunking
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# -----------------------------
# Utilities
# -----------------------------

def read_text_file(path: Path) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def read_pdf_file(path: Path) -> str:
    text_chunks = []
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for p in range(len(reader.pages)):
            try:
                text = reader.pages[p].extract_text() or ""
            except Exception:
                text = ""
            text_chunks.append(text)
    return "\n".join(text_chunks)


def chunk_text(text: str, max_tokens: int = 250) -> List[str]:
    # Simple chunking based on sentences; `max_tokens` is rough words target
    sents = sent_tokenize(text)
    chunks = []
    cur = []
    cur_len = 0
    for s in sents:
        words = s.split()
        if cur_len + len(words) > max_tokens and cur:
            chunks.append(' '.join(cur))
            cur = []
            cur_len = 0
        cur.append(s)
        cur_len += len(words)
    if cur:
        chunks.append(' '.join(cur))
    return chunks

# -----------------------------
# Data classes
# -----------------------------

@dataclass
class DocChunk:
    doc_id: str
    chunk_id: str
    text: str
    source_path: str
    meta: Dict

# -----------------------------
# Embedding store (FAISS + SQLite for metadata)
# -----------------------------

class EmbeddingStore:
    def __init__(self, index_path: str = 'faiss.index', db_path: str = 'embeddings.db', model_name: str = 'all-MiniLM-L6-v2'):
        self.index_path = index_path
        self.db_path = db_path
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._index = None
        self._conn = None
        self._ensure_db()
        self._load_or_create_index()

    # ---- SQLite metadata ----
    def _ensure_db(self):
        self._conn = sqlite3.connect(self.db_path)
        cur = self._conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS chunks (
                        id TEXT PRIMARY KEY,
                        doc_id TEXT,
                        text TEXT,
                        source_path TEXT,
                        meta TEXT
                    )''')
        self._conn.commit()

    def _load_or_create_index(self):
        if os.path.exists(self.index_path):
            try:
                self._index = faiss.read_index(self.index_path)
            except Exception:
                print('Could not read existing FAISS index, creating new one')
                self._index = faiss.IndexFlatIP(self.dim)
        else:
            # use inner product index with normalized vectors for cosine similarity
            self._index = faiss.IndexFlatIP(self.dim)

    def _save_index(self):
        faiss.write_index(self._index, self.index_path)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def add_chunks(self, chunks: List[DocChunk], batch_size: int = 128):
        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metas = [json.dumps(c.meta) for c in chunks]
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        vectors = self._normalize(vectors).astype('float32')
        # add to index
        if self._index.is_trained or isinstance(self._index, faiss.IndexFlat):
            self._index.add(vectors)
        else:
            raise RuntimeError('FAISS index not ready')
        # store metadata in sqlite
        cur = self._conn.cursor()
        for cid, doc_id, text, source, meta in zip(ids, [c.doc_id for c in chunks], texts, [c.source_path for c in chunks], metas):
            cur.execute('INSERT OR REPLACE INTO chunks (id, doc_id, text, source_path, meta) VALUES (?,?,?,?,?)',
                        (cid, doc_id, text, source, meta))
        self._conn.commit()
        self._save_index()

    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[DocChunk, float]]:
        qvec = self.model.encode([query_text], convert_to_numpy=True)
        qvec = self._normalize(qvec).astype('float32')
        D, I = self._index.search(qvec, top_k)
        results = []
        cur = self._conn.cursor()
        for score, idx in zip(D[0], I[0]):
            # FAISS IndexFlat returns index positions; we need to fetch the corresponding id in sqlite
            # sqlite rows are keyed by chunk id, but we didn't store row numbers. To map, we can store the chunk ids
            # in insertion order by reading all ids from DB into a list once.
            pass
        # To avoid expensive mapping each query, maintain an in-memory id list
        ids = self._get_all_chunk_ids()
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(ids):
                continue
            cid = ids[idx]
            cur.execute('SELECT doc_id, text, source_path, meta FROM chunks WHERE id=?', (cid,))
            row = cur.fetchone()
            if not row:
                continue
            doc_id, text, source_path, meta_json = row
            meta = json.loads(meta_json)
            chunk = DocChunk(doc_id=doc_id, chunk_id=cid, text=text, source_path=source_path, meta=meta)
            results.append((chunk, float(score)))
        return results

    def _get_all_chunk_ids(self) -> List[str]:
        cur = self._conn.cursor()
        cur.execute('SELECT id FROM chunks ORDER BY rowid')
        rows = cur.fetchall()
        return [r[0] for r in rows]

# -----------------------------
# Research Agent: ingestion, retrieval, reasoning, summarization
# -----------------------------

class ResearchAgent:
    def __init__(self, store: EmbeddingStore):
        self.store = store
        self.reasoner = Reasoner()

    def ingest_directory(self, dirpath: str, file_types: Optional[List[str]] = None, chunk_size_words: int = 250):
        file_types = file_types or ['.txt', '.md', '.pdf']
        dirp = Path(dirpath)
        files = [p for p in dirp.rglob('*') if p.suffix.lower() in file_types]
        all_chunks = []
        for p in tqdm(files, desc='Ingesting files'):
            text = ''
            try:
                if p.suffix.lower() == '.pdf':
                    text = read_pdf_file(p)
                else:
                    text = read_text_file(p)
            except Exception as e:
                print(f'Failed to read {p}: {e}')
                continue
            chunks = chunk_text(text, max_tokens=chunk_size_words)
            for i, ch in enumerate(chunks):
                chunk = DocChunk(doc_id=str(p.resolve()), chunk_id=f'{p.stem}__{i}', text=ch, source_path=str(p.resolve()), meta={'chunk_index': i})
                all_chunks.append(chunk)
        if all_chunks:
            self.store.add_chunks(all_chunks)
        return len(all_chunks)

    def answer_query(self, query: str, top_k: int = 8, summarize: bool = True) -> Dict:
        # Multi-step: decompose, retrieve per subquery, synthesize
        subqueries = self.reasoner.decompose(query)
        gathered = []
        for sq in subqueries:
            hits = self.store.query(sq, top_k=top_k)
            gathered.append({'subquery': sq, 'hits': hits})
        # Combine unique chunks
        unique = {}
        for g in gathered:
            for chunk, score in g['hits']:
                unique[chunk.chunk_id] = (chunk, score)
        # Rank by score
        ranked = sorted(unique.values(), key=lambda x: x[1], reverse=True)
        context = '\n\n'.join([f"Source: {c.source_path}\nScore: {s:.4f}\n{c.text}" for c, s in ranked[:top_k]])
        if summarize:
            summary = self.reasoner.summarize(query, context)
        else:
            summary = context
        report = {
            'query': query,
            'subqueries': subqueries,
            'top_hits': [(c.chunk_id, c.source_path, s) for c, s in ranked[:top_k]],
            'summary': summary
        }
        return report

    def export_report_markdown(self, report: Dict, out_path: str):
        md_lines = [f"# Research Report\n", f"**Query:** {report['query']}\n\n", "## Subqueries\n"]
        for i, sq in enumerate(report['subqueries']):
            md_lines.append(f"{i+1}. {sq}\n")
        md_lines.append('\n## Top Hits\n')
        for cid, src, score in report['top_hits']:
            md_lines.append(f"- `{cid}` — {src} (score {score:.4f})\n")
        md_lines.append('\n## Summary\n')
        md_lines.append(report['summary'] + '\n')
        md_text = '\n'.join(md_lines)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(md_text)
        return out_path

    def export_report_pdf(self, report: Dict, out_path: str):
        # Simple PDF via reportlab
        md_path = out_path.replace('.pdf', '.md')
        self.export_report_markdown(report, md_path)
        c = canvas.Canvas(out_path, pagesize=A4)
        width, height = A4
        textobject = c.beginText(50, height - 50)
        textobject.setFont('Helvetica', 10)
        with open(md_path, 'r', encoding='utf-8') as f:
            for line in f:
                # naive wrapping: reportlab will wrap but keep simple
                textobject.textLine(line.rstrip())
        c.drawText(textobject)
        c.save()
        return out_path

# -----------------------------
# Reasoner: decomposition and summarization
# -----------------------------

class Reasoner:
    def __init__(self, llm_model: Optional[str] = None):
        # If user supplied a local transformers seq2seq model, we can use it for decomposition/summarization
        self.llm_model = llm_model
        if TRANSFORMERS_AVAILABLE and llm_model:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(llm_model)
                self.pipe = pipeline('text2text-generation', model=self.model, tokenizer=self.tokenizer)
            except Exception as e:
                print('Could not load local transformers model, falling back to heuristics:', e)
                self.pipe = None
        else:
            self.pipe = None

    def decompose(self, query: str) -> List[str]:
        # If a local LLM pipeline exists, use it to decompose into sub-questions
        if self.pipe:
            prompt = f"Decompose the following research query into 3-6 concise sub-questions or search queries, each on its own line:\n\n{query}"
            try:
                out = self.pipe(prompt, max_length=256)[0]['generated_text']
                subs = [line.strip() for line in out.splitlines() if line.strip()]
                if len(subs) >= 1:
                    return subs
            except Exception:
                pass
        # Heuristic decomposition: split by punctuation and important nouns (simple)
        sents = sent_tokenize(query)
        # If query is short, create variations
        if len(sents) == 1:
            q = sents[0]
            subs = [q, f"background on {q}", f"recent findings about {q}", f"methods for {q}"]
            return subs
        return sents

    def summarize(self, query: str, context: str, max_length: int = 300) -> str:
        if self.pipe:
            prompt = f"You are an expert researcher. Given the research query:\n{query}\nand the following gathered context:\n{context}\nCompose a concise, structured summary (short background, findings, and suggested next steps). Keep it under {max_length} words."
            try:
                out = self.pipe(prompt, max_length=512)[0]['generated_text']
                return out.strip()
            except Exception:
                pass
        # Simple extractive summarizer: pick top N sentences by TF-IDF-like scoring
        sentences = sent_tokenize(context)
        # score by occurrence of query words
        q_words = set([w.lower() for w in query.split() if len(w) > 2])
        scored = []
        for s in sentences:
            words = [w.lower() for w in s.split()]
            match = sum(1 for w in words if w in q_words)
            scored.append((match, len(s), s))
        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        top_sents = [s for _, _, s in scored[:6]]
        return '\n'.join(top_sents)

# -----------------------------
# CLI / Example usage
# -----------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Local Research Agent')
    parser.add_argument('--ingest', type=str, help='Directory to ingest')
    parser.add_argument('--query', type=str, help='Query to run')
    parser.add_argument('--index', type=str, default='faiss.index', help='FAISS index path')
    parser.add_argument('--db', type=str, default='embeddings.db', help='SQLite DB path')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model')
    parser.add_argument('--export-md', type=str, help='Export report to markdown')
    parser.add_argument('--export-pdf', type=str, help='Export report to pdf')
    args = parser.parse_args()

    store = EmbeddingStore(index_path=args.index, db_path=args.db, model_name=args.model)
    agent = ResearchAgent(store)

    if args.ingest:
        n = agent.ingest_directory(args.ingest)
        print(f'Ingested {n} chunks')

    if args.query:
        report = agent.answer_query(args.query)
        print('\n---- SUMMARY ----\n')
        print(report['summary'][:2000])
        if args.export_md:
            agent.export_report_markdown(report, args.export_md)
            print('Exported markdown to', args.export_md)
        if args.export_pdf:
            agent.export_report_pdf(report, args.export_pdf)
            print('Exported pdf to', args.export_pdf)

    if not args.ingest and not args.query:
        print('Run with --ingest <dir> to ingest documents, and --query "your question" to run a research query')
