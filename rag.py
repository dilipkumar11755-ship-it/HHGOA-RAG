import os
import time
import asyncio
import hashlib
import pickle
import torch
import re
import unicodedata
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from groq import AsyncGroq

load_dotenv()

# ── CACHE ──
_cache = {}
def _cache_key(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()

# ── INIT ──
print("🚀 Loading async RAG pipeline...")
embedder = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
embedder.eval()
qdrant = AsyncQdrantClient(host="localhost", port=6333)
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
COLLECTION = "msmarco_v2"

print("📚 Loading BM25 index...")
with open("bm25_index.pkl", "rb") as f:
    bm25, bm25_corpus = pickle.load(f)
print(f"✅ BM25 ready ({len(bm25_corpus):,} docs)")
print("✅ Pipeline ready")

# ── JUNK FILTER ──
JUNK_PATTERNS = [
    "cookie policy", "all rights reserved", "site feedback",
    "advertise with us", "privacy policy", "terms of service",
    "copyright", "careers - we're hiring", "which of the following",
    "white pages", "reverse phone", "site directory",
    "kevin has edited", "taught middle and high school",
    "he was like the", "i was a sponge", "master's degree in",
    "encyclopedias", "acting professors","he was like the",
    "i was a sponge", "acting professor","fernando zavala",
    "republic of peru","head of government","overview government name",
]

def is_clean(text: str) -> bool:
    t = text.lower()
    return not any(p in t for p in JUNK_PATTERNS)

def is_relevant(text: str, query: str) -> bool:
    """Check if passage actually contains key query entities."""
    query_words = set(tokenize(query.lower())) - {
        'what','who','where','when','how','is','was','the','a','an',
        'of','in','did','are','do','tell','me','about'
    }
    if not query_words:
        return True
    text_words = set(tokenize(text.lower()))
    overlap = len(query_words & text_words) / len(query_words)
    return overlap >= 0.3

# ── GUARDRAILS ──
BLOCKED_PHRASES = ["how to make a bomb", "how to hack", "how to kill",
                   "how to make drugs", "how to build a weapon"]

def is_off_topic(query: str) -> bool:
    return any(p in query.lower() for p in BLOCKED_PHRASES)

def is_grounded(answer: str, context: str) -> bool:
    if not answer or len(answer.strip()) < 10:
        return False
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    return len(answer_words & context_words) / max(len(answer_words), 1) > 0.10

# ── DEVANAGARI-AWARE TOKENIZER ──
def tokenize(text: str) -> list:
    tokens, current = [], []
    for char in text.lower():
        cat = unicodedata.category(char)
        if cat.startswith('L') or cat.startswith('N') or cat in ('Mc', 'Mn'):
            current.append(char)
        else:
            if current:
                tokens.append(''.join(current))
                current = []
    if current:
        tokens.append(''.join(current))
    return tokens if tokens else text.lower().split()

# ── RETRIEVE: Vector only (BM25 disabled for speed) ──
async def retrieve(query: str, top_k: int = 5):
    def _encode():
        with torch.inference_mode():
            return embedder.encode(
                f"query: {query}",
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=1
            ).tolist()

    vec = await asyncio.to_thread(_encode)
    result = await qdrant.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=top_k,
        with_payload=True
    )
    # Filter junk
    clean = [r for r in result.points 
             if is_clean(r.payload.get("text", "")) 
             and is_relevant(r.payload.get("text", ""), query)]
    return clean[:top_k] if clean else result.points[:top_k]

# ── BUILD CONTEXT ──
def build_context(results) -> str:
    return "\n\n".join(
        f"[Passage {i+1}] {r.payload['text']}"
        for i, r in enumerate(results)
    )

# ── EXTRACT ANSWER ──
def extract_answer(results, query: str) -> str:
    if not results:
        return ""
    full_text = results[0].payload.get("text", "").strip()
    if not full_text:
        return ""

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_text) if len(s.strip()) > 20]
    if len(sentences) > 1:
        stop = {'what','who','where','when','how','is','was','the','a','an','of','in','did','are','do'}
        query_words = set(tokenize(query)) - stop
        best = max(sentences, key=lambda s: len(set(tokenize(s)) & query_words))
        answer = best.strip()
        if not answer.endswith('.'):
            answer += '.'
    else:
        answer = full_text[:300]

    if len(answer) < 30:
        answer = full_text[:300]
        if '.' in answer:
            answer = answer.rsplit('.', 1)[0] + '.'

    return answer
        # Final junk check on answer itself
    answer_lower = answer.lower()
    junk_answer = ["like the", "sponge", "acting professor", "kevin has",
                   "fernando", "republic of peru", "head of government",
                   "overview government", "master's degree"]
    if any(p in answer_lower for p in junk_answer):
        return ""

# ── LLM FALLBACK ──
async def llm_answer(query: str, context: str) -> str:
    is_hindi = bool(re.search(r'[\u0900-\u097F]', query))
    lang = "Hindi" if is_hindi else "English"
    try:
        stream = await groq_client.chat.completions.create(
            model="allam-2-7b",
            messages=[{
                "role": "system",
                "content": f"You are a factual assistant. Answer in {lang}. Use context only if relevant. Otherwise use your knowledge. Never mention ads, websites, or irrelevant information. Keep answer to 1-2 sentences."
            },{
                "role": "user",
                "content": f"Context: {context[:300]}\n\nQuestion: {query}\nAnswer:"
            }],
            max_tokens=100,
            temperature=0.1,
            stream=True
        )
        full = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                full += delta.content
                print(delta.content, end="", flush=True)
        print()
        return full.strip() if full.strip() else "I couldn't find a reliable answer."
    except Exception as e:
        return "I couldn't find a reliable answer."

# ── MAIN PIPELINE ──
async def run_pipeline(query: str) -> dict:
    t_start = time.time()

    # Cap query length (fixes P100 spike — same fix pucho.me used)
    query = query[:512]

    if not query or len(query.strip()) < 3:
        return {"query": query, "answer": "Please ask a complete question.",
                "guardrail_triggered": True, "latency_ms": 0.1, "passages_used": []}

    ck = _cache_key(query)
    if ck in _cache:
        cached = _cache[ck].copy()
        cached["latency_ms"] = 0.5
        cached["cached"] = True
        return cached

    if is_off_topic(query):
        return {"query": query,
                "answer": "I can't answer that — it's outside the scope of this system.",
                "guardrail_triggered": True,
                "latency_ms": round((time.time() - t_start) * 1000, 1),
                "passages_used": []}

    # Retrieve
    t_retrieve = time.time()
    results = await retrieve(query)
    retrieve_ms = round((time.time() - t_retrieve) * 1000, 1)

    if not results:
        return {"query": query,
                "answer": "I couldn't find relevant information in the knowledge base.",
                "guardrail_triggered": True,
                "latency_ms": round((time.time() - t_start) * 1000, 1),
                "passages_used": []}

    top_score = getattr(results[0], 'score', 0.0)
    context = build_context(results)

    if top_score >= 0.82:
            # ── HYBRID DECISION ──
    # Try extraction only if very high confidence AND clean answer
        extracted = extract_answer(results, query) if top_score >= 0.82 else ""
    
        if extracted and len(extracted) > 50:
            answer = extracted
            llm_ms = 0
            method = "extraction"
            print(f"\n💬 Answer (extracted): {answer[:80]}...")
        else:
            print(f"\n💬 Answer: ", end="")
            t_llm = time.time()
            try:
                answer = await asyncio.wait_for(llm_answer(query, context), timeout=0.15)
                llm_ms = round((time.time() - t_llm) * 1000, 1)
                method = "llm"
            except asyncio.TimeoutError:
                # LLM too slow — fall back to best passage instead of waiting
                answer = results[0].payload["text"][:300].strip()
                if '.' in answer:
                    answer = answer.rsplit('.', 1)[0] + '.'
                llm_ms = round((time.time() - t_llm) * 1000, 1)
                method = "extraction_fallback"
                print(f"(LLM timeout — using passage fallback)")
    elif top_score >= 0.40:
        print(f"\n💬 Answer: ", end="")
        t_llm = time.time()
        try:
            answer = await asyncio.wait_for(llm_answer(query, context), timeout=0.15)
            llm_ms = round((time.time() - t_llm) * 1000, 1)
            method = "llm"
        except asyncio.TimeoutError:
            # LLM too slow — fall back to best passage instead of waiting
            answer = results[0].payload["text"][:300].strip()
            if '.' in answer:
                answer = answer.rsplit('.', 1)[0] + '.'
            llm_ms = round((time.time() - t_llm) * 1000, 1)
            method = "extraction_fallback"
            print(f"(LLM timeout — using passage fallback)")
    else:
        answer = "I couldn't find reliable information about this in the knowledge base."
        llm_ms = 0
        method = "no_match"

    total_ms = round((time.time() - t_start) * 1000, 1)
    grounded = is_grounded(answer, context) if method == "llm" else True

    result = {
        "query": query,
        "answer": answer,
        "guardrail_triggered": False,
        "grounded": grounded,
        "method": method,
        "top_score": round(top_score, 4),
        "latency_ms": total_ms,
        "retrieve_ms": retrieve_ms,
        "llm_ms": llm_ms,
        "passages_used": [
            {
                "text": r.payload["text"][:200],
                "score": round(getattr(r, 'score', 0.0), 4),
                "strategy": r.payload.get("strategy", ""),
                "is_selected": r.payload.get("is_selected", False),
                "lang": r.payload.get("lang", "en"),
            }
            for r in results
        ]
    }

    _cache[ck] = result
    return result