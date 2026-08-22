# HHGOA-Rag
# VoiceRAG — Multilingual Voice-Powered RAG Pipeline
### HH Goa 2026 Shortlisting Task 2

A real-time voice-enabled Retrieval-Augmented Generation system built on the **ai4bharat/MSMARCO-XI** dataset, supporting Hindi, Bengali, Tamil, and Urdu queries with sub-20ms retrieval latency.

---

## Live Demo
🔗 [Live Link — paste link after deploying]

---

## Architecture

```
🎤 Voice Input
    ↓
📝 ElevenLabs STT (Speech-to-Text)
    ↓
🔍 Qdrant Vector Search — multilingual-e5-small embeddings
    ↓
⚡ Hybrid Decision Engine
    → High confidence (score ≥ 0.67) → Direct Extraction (~16ms)
    → Medium confidence → Groq LLM Generation (~480ms)
    → Low confidence → "Not in knowledge base"
    ↓
✅ Grounded Answer with Citations
```

---

## Benchmark Results

| Metric | Retrieval Only | Full Pipeline |
|--------|---------------|---------------|
| P50 | 15.9ms | 16.1ms |
| P70 | 16.5ms | 16.7ms |
| P100 | 18.0ms | 18.2ms |

> 19/20 benchmark queries resolved via direct extraction under 20ms total.  
> All queries meet the sub-200ms target at the retrieval layer.

---

## Stack

| Layer | Technology |
|-------|-----------|
| STT | ElevenLabs Scribe v1 |
| Embeddings | intfloat/multilingual-e5-small |
| Vector DB | Qdrant (Docker) |
| LLM Fallback | Groq — allam-2-7b |
| Backend | FastAPI + Python |
| Frontend | Vanilla HTML/CSS/JS |
| Dataset | ai4bharat/MSMARCO-XI |

---

## Chunking Strategy

Four strategies applied during offline indexing:

- **Full passage** — complete passage as single chunk
- **Sentence-level** — individual sentences for pinpoint retrieval
- **Sliding window** — 2-sentence windows with 1-sentence overlap
- **Translated** — native language passages for multilingual queries

**695,000+ vectors indexed** across Hindi, Bengali, Tamil, and Urdu.

---

## Guardrails

- Off-topic detection — blocks harmful or out-of-scope queries
- Confidence threshold — refuses low-confidence retrievals honestly
- Grounding check — verifies answer is supported by retrieved context
- Short answer fallback — uses full passage when extracted sentence is too brief

---

## Languages Supported

| Language | Script | Source |
|----------|--------|--------|
| Hindi | Devanagari | MSMARCO-XI train |
| Bengali | Bengali | MSMARCO-XI validation |
| Tamil | Tamil | MSMARCO-XI validation |
| Urdu | Nastaliq | MSMARCO-XI validation |

---

## Run Locally

```bash
# Prerequisites: Python 3.11+, Docker

# 1. Start Qdrant
docker start qdrant

# 2. Clone and install
git clone https://github.com/dilipkumar11755-ship-it/HHGOA-RAG
cd HHGOA-RAG
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Set up environment
# Create .env with:
# GROQ_API_KEY=your_key
# ELEVENLABS_API_KEY=your_key

# 4. Run
python app.py
# Open http://localhost:8000
```

---

## Features

- 🎤 Voice input via browser microphone
- ⚡ Sub-20ms response for most queries via direct extraction
- 🌍 4 Indic languages supported
- 📊 Live latency benchmark button in UI
- 🛡️ Multi-layer guardrails
- 💬 Chat-style interface with retrieval citations
- 📈 Real-time pipeline visualization sidebar

---

Built for **Hacker House Goa 2026** — Shortlisting Task 2  
*"Less Noise. More Signal."*
