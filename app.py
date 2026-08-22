import os
import asyncio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from rag import run_pipeline, embedder, qdrant, groq_client
from stt import transcribe_bytes_async

load_dotenv()

app = FastAPI(title="HH Goa RAG — Voice Q&A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Step 1: Voice → Text"""
    audio_bytes = await audio.read()
    
    # Using the fast async function!
    result = await transcribe_bytes_async(audio_bytes, audio.filename or "audio.wav")
    
    if not result["success"]:
        return JSONResponse({"error": result["error"]}, status_code=500)
    return {"transcript": result["transcript"]}

@app.post("/ask")
async def ask(query: str = Form(...)):
    """Step 2: Text → RAG Answer"""
    result = await run_pipeline(query)
    return result

@app.post("/benchmark")
async def benchmark_endpoint(query: str = Form(...)):
    """Benchmark endpoint — bypasses cache for fresh measurements"""
    import hashlib
    from rag import _cache
    ck = hashlib.md5(query.lower().strip().encode()).hexdigest()
    _cache.pop(ck, None)
    result = await run_pipeline(query)
    # Also remove after so next benchmark call is fresh too
    _cache.pop(ck, None)
    return result

@app.post("/clear-cache")
async def clear_cache():
    from rag import _cache
    _cache.clear()
    return {"cleared": True}

@app.post("/voice-ask")
async def voice_ask(audio: UploadFile = File(...)):
    """Full pipeline: Voice → Text → RAG Answer"""
    import time
    t_start = time.time()

    # STT
    audio_bytes = await audio.read()
    
    # Using the fast async function!
    stt_result = await transcribe_bytes_async(audio_bytes, audio.filename or "audio.wav")

    if not stt_result["success"]:
        return JSONResponse({"error": stt_result["error"]}, status_code=500)

    transcript = stt_result["transcript"]
    if not transcript.strip():
        return JSONResponse({"error": "Could not transcribe audio. Please speak clearly."}, status_code=400)

    # RAG
    rag_result = await run_pipeline(transcript)
    rag_result["transcript"] = transcript
    rag_result["stt_ms"] = round((time.time() - t_start) * 1000 - rag_result["latency_ms"], 1)

    return rag_result

@app.on_event("startup")
async def startup():
    from rag import _cache
    _cache.clear()
    print("✅ Cache cleared on startup")


@app.get("/health")
async def health():
    count = await qdrant.count(collection_name="msmarco_hi")
    return {
        "status": "ok",
        "vectors_indexed": count.count,
        "models_loaded": True
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

@app.get("/clear")
async def clear_cache():
    from rag import _cache
    _cache.clear()
    return {"cleared": len(_cache)}