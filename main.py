""" RAG chatbot (minimal, open-source components)

Converts a text file into chunked passages

Encodes passages with sentence-transformers (all-MiniLM-L6-v2)

Simple cosine-similarity semantic search (numpy)

Retrieves ~500 tokens (approx) of context and calls Ollama (1-1.5B model) via local HTTP API


Requirements:

python 3.10+

pip install sentence-transformers numpy requests tqdm

Ollama installed and running locally (https://ollama.com) with a ~1-1.5B model pulled, eg ollama pull llama2-mini or any small open-source model you prefer.


Usage:

Edit MODEL_NAME below to your local ollama model name.

Run: python rag_ollama_chatbot.py --build-db data.txt

Then: python rag_ollama_chatbot.py --chat


This is intentionally short and dependency-light. """

from future import annotations import argparse import json import os import re from dataclasses import dataclass from typing import List, Tuple

import numpy as np from sentence_transformers import SentenceTransformer import requests from tqdm import tqdm

=== CONFIG ===

MODEL_NAME = os.getenv('OLLAMA_MODEL', 'llama2-mini')  # change to your local ollama model name (1-1.5B) OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434/api/generate') EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'  # small, open-source CHUNK_SIZE = 800  # characters per chunk (tune) CHUNK_OVERLAP = 200 CONTEXT_TOKEN_LIMIT = 500  # tokens for retrieval used to prompt LLM TOKEN_CHAR_RATIO = 4  # heuristic: ~4 chars/token DB_PATH = 'rag_db.json'

=== Helpers ===

def clean_text(s: str) -> str: s = s.replace('\r', ' ') s = re.sub(r"\s+", " ", s).strip() return s

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]: text = clean_text(text) chunks = [] start = 0 n = len(text) while start < n: end = min(start + chunk_size, n) chunk = text[start:end] chunks.append(chunk) if end == n: break start = end - overlap return chunks

def approximate_truncate_by_tokens(text: str, token_limit: int) -> str: # heuristic: token_limit * TOKEN_CHAR_RATIO characters max_chars = token_limit * TOKEN_CHAR_RATIO if len(text) <= max_chars: return text # try to cut at sentence boundary cut = text[:max_chars] last_period = cut.rfind('. ') if last_period > int(max_chars * 0.6): return cut[:last_period+1] return cut

@dataclass class Passage: id: int text: str embedding: List[float]

class SimpleRAGIndex: def init(self, model_name: str = EMBED_MODEL_NAME): self.encoder = SentenceTransformer(model_name) self.passages: List[Passage] = [] self.emb_matrix: np.ndarray | None = None

def build_from_text(self, text: str):
    chunks = chunk_text(text)
    embeddings = self.encoder.encode(chunks, show_progress_bar=True)
    self.passages = [Passage(i, chunks[i], embeddings[i].tolist()) for i in range(len(chunks))]
    self.emb_matrix = np.vstack([p.embedding for p in self.passages])

def save(self, path: str = DB_PATH):
    serial = [{
        'id': p.id,
        'text': p.text,
        'embedding': p.embedding
    } for p in self.passages]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'model': EMBED_MODEL_NAME, 'passages': serial}, f)

def load(self, path: str = DB_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    self.passages = [Passage(p['id'], p['text'], p['embedding']) for p in data['passages']]
    self.emb_matrix = np.vstack([p.embedding for p in self.passages])
    # re-init encoder lazily
    self.encoder = SentenceTransformer(EMBED_MODEL_NAME)

def query(self, q: str, top_k: int = 5) -> List[Tuple[Passage, float]]:
    q_emb = self.encoder.encode([q])[0]
    emb = np.array(q_emb)
    # cosine similarity
    M = self.emb_matrix
    dots = M @ emb
    m_norm = np.linalg.norm(M, axis=1)
    q_norm = np.linalg.norm(emb) + 1e-12
    sims = dots / (m_norm * q_norm + 1e-12)
    idx = np.argsort(-sims)[:top_k]
    return [(self.passages[i], float(sims[i])) for i in idx]

=== Ollama call ===

def call_ollama_system(system_prompt: str, user_prompt: str, model: str = MODEL_NAME, max_tokens: int = 512): # Ollama HTTP generate API expects JSON like {"model": "...", "prompt": "..."} # We'll craft a prompt that includes system + user prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n" payload = { 'model': model, 'prompt': prompt, 'max_tokens': max_tokens, # you can add temperature/top_p as desired } try: resp = requests.post(OLLAMA_URL, json=payload, timeout=60) resp.raise_for_status() j = resp.json() # Ollama's response structure may vary; many endpoints return {'choices': [{'message': {'content': '...'}}]} # We'll try to find text in the JSON if isinstance(j, dict): # try common keys if 'choices' in j and len(j['choices']) > 0: c = j['choices'][0] if 'message' in c and 'content' in c['message']: return c['message']['content'] if 'text' in c: return c['text'] if 'text' in j: return j['text'] return json.dumps(j) except Exception as e: return f"[OLLAMA ERROR] {e}"

=== High-level chat logic ===

def build_database_from_file(path: str, db_out: str = DB_PATH): if not os.path.exists(path): raise FileNotFoundError(path) with open(path, 'r', encoding='utf-8') as f: text = f.read() rag = SimpleRAGIndex() rag.build_from_text(text) rag.save(db_out) print(f"Saved database to {db_out} with {len(rag.passages)} passages.")

def chat_loop(db_path: str = DB_PATH): if not os.path.exists(db_path): print(f"DB not found: {db_path}. Run with --build-db first.") return rag = SimpleRAGIndex() rag.load(db_path) print("Loaded RAG DB. Ready to chat. Type 'exit' to quit.") while True: q = input('\nYou: ') if not q or q.strip().lower() in ('exit', 'quit'): break hits = rag.query(q, top_k=8) # assemble context until approx CONTEXT_TOKEN_LIMIT tokens accumulated = '' chars_limit = CONTEXT_TOKEN_LIMIT * TOKEN_CHAR_RATIO for p, s in hits: if len(accumulated) + len(p.text) > chars_limit: need = chars_limit - len(accumulated) if need <= 0: break accumulated += '\n' + approximate_truncate_by_tokens(p.text, int(need / TOKEN_CHAR_RATIO)) break accumulated += '\n' + p.text context = clean_text(accumulated) system_prompt = ( "You are a helpful assistant. Use the provided CONTEXT to answer the USER question. " "If the answer is not in the context, be honest and say you don't know. Keep answers concise." ) user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{q}\n\nProvide an answer based only on the CONTEXT when possible." print('\n[Calling model...]') out = call_ollama_system(system_prompt, user_prompt, model=MODEL_NAME, max_tokens=512) print('\nAssistant:', out)

=== CLI ===

def main(): parser = argparse.ArgumentParser() parser.add_argument('--build-db', type=str, help='Path to text file to build RAG DB') parser.add_argument('--db-path', type=str, default=DB_PATH) parser.add_argument('--chat', action='store_true', help='Run interactive chat (requires DB)') parser.add_argument('--model', type=str, help='Ollama model name (overrides default)') args = parser.parse_args() global MODEL_NAME if args.model: MODEL_NAME = args.model if args.build_db: build_database_from_file(args.build_db, db_out=args.db_path) if args.chat: chat_loop(db_path=args
.db_path)

if name == 'main': main()
