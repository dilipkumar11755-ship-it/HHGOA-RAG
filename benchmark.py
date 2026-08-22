import asyncio
import time
from rag import run_pipeline

QUERIES = [
    "what was the impact of the manhattan project?",
    "who led the manhattan project?",
    "when was the first atomic bomb tested?",
    "what is the capital of France?",
    "who was Robert Oppenheimer?",
    "what is nuclear fission?",
    "where is Los Alamos located?",
    "what happened in World War 2?",
    "who was General Leslie Groves?",
    "what is the Columbia River?",
    "what are modal verbs in English?",
    "how does DNA replication work?",
    "what is photosynthesis?",
    "who invented the telephone?",
    "what is the speed of light?",
    "what causes earthquakes?",
    "how do vaccines work?",
    "what is the water cycle?",
    "who was Albert Einstein?",
    "what is gravity?",
]

async def benchmark():
    print("🔥 Warming up...")
    await run_pipeline("test warmup query")
    print("✅ Warm up done\n")

    retrieve_times = []
    llm_times = []
    total_times = []

    print(f"⏱ Running {len(QUERIES)} queries...\n")

    for i, q in enumerate(QUERIES):
        r = await run_pipeline(q)
        retrieve_times.append(r.get("retrieve_ms") or 0)
        llm_times.append(r.get("llm_ms") or 0)
        total_times.append(r.get("latency_ms") or 0)
        print(f"[{i+1:02d}] {q[:45]:<45} | R:{r.get('retrieve_ms')}ms | L:{r.get('llm_ms')}ms | T:{r.get('latency_ms')}ms | Score:{r.get('top_score')} | Method:{r.get('method')}")

    def percentile(data, p):
        s = sorted(data)
        i = int(len(s) * p / 100)
        return round(s[min(i, len(s)-1)], 1)

    print(f"\n{'='*60}")
    print(f"📊 BENCHMARK RESULTS ({len(QUERIES)} queries)")
    print(f"{'='*60}")
    print(f"\n🔍 Retrieval latency:")
    print(f"   P50: {percentile(retrieve_times, 50)}ms")
    print(f"   P70: {percentile(retrieve_times, 70)}ms")
    print(f"   P100: {percentile(retrieve_times, 100)}ms")
    print(f"\n🤖 LLM latency:")
    print(f"   P50: {percentile(llm_times, 50)}ms")
    print(f"   P70: {percentile(llm_times, 70)}ms")
    print(f"   P100: {percentile(llm_times, 100)}ms")
    print(f"\n⚡ Total pipeline latency:")
    print(f"   P50: {percentile(total_times, 50)}ms")
    print(f"   P70: {percentile(total_times, 70)}ms")
    print(f"   P100: {percentile(total_times, 100)}ms")
    print(f"\n✅ These are your official submission numbers!")

asyncio.run(benchmark())