from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
import json
import time
import pickle
from pathlib import Path

COLLECTION_NAME = "msmarco_v2"  # new collection
BATCH_SIZE = 1024

print("📂 Loading chunks...")
with open("chunks.json", "r", encoding="utf-8") as f:
    hindi_chunks = json.load(f)

with open("chunks_multilingual.json", "r", encoding="utf-8") as f:
    all_multi = json.load(f)

# Filter multilingual to 3 languages
from collections import defaultdict
lang_count = defaultdict(int)
filtered_multi = []
for c in all_multi:
    if c.get("lang") in ["ben", "tam", "urd"] and lang_count[c["lang"]] < 50000:
        filtered_multi.append(c)
        lang_count[c["lang"]] += 1

# Keep only full_passage strategy — cuts volume by 75%
hindi_chunks = [c for c in hindi_chunks if c["strategy"] == "full_passage"]
filtered_multi = [c for c in filtered_multi if c["strategy"] == "full_passage"]
all_chunks = hindi_chunks + filtered_multi
print(f"✅ Total chunks: {len(all_chunks):,} (Hindi: {len(hindi_chunks):,} + Multilingual: {len(filtered_multi):,})")

print("\n🧠 Loading multilingual-e5-small...")
model = SentenceTransformer("intfloat/multilingual-e5-small")
print("✅ Model ready (384 dims)")

print("\n🗄️ Setting up Qdrant v2 collection...")
client = QdrantClient(host="localhost", port=6333)

existing = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME in existing:
    client.delete_collection(COLLECTION_NAME)
    print("  Deleted existing v2 collection")

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=384,
        distance=models.Distance.COSINE
    )
)
print(f"✅ Collection '{COLLECTION_NAME}' created")

print(f"\n⚡ Embedding and indexing {len(all_chunks):,} chunks...")
start_time = time.time()
total_batches = (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in range(total_batches):
    batch_start = batch_idx * BATCH_SIZE
    batch_end = min(batch_start + BATCH_SIZE, len(all_chunks))
    batch = all_chunks[batch_start:batch_end]

    # multilingual-e5 needs "query: " or "passage: " prefix
    texts = [f"passage: {c['text']}" for c in batch]
    vectors = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).tolist()

    payloads = [{
        "text": c["text"],
        "lang": c.get("lang", "en"),
        "lang_name": c.get("lang_name", "English"),
        "strategy": c["strategy"],
        "query_id": c["query_id"],
        "passage_idx": c["passage_idx"],
        "is_selected": c["is_selected"],
        "query_type": c["query_type"],
        "source_query": c["source_query"],
    } for c in batch]

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=models.Batch(
            ids=list(range(batch_start, batch_end)),
            vectors=vectors,
            payloads=payloads,
        )
    )

    elapsed = time.time() - start_time
    done = batch_end
    rate = done / elapsed
    eta = (len(all_chunks) - done) / rate if rate > 0 else 0
    print(f"  [{batch_idx+1}/{total_batches}] {done:,}/{len(all_chunks):,} | "
          f"{rate:.0f} chunks/sec | ETA: {eta/60:.1f} min")

total_time = time.time() - start_time
print(f"\n✅ Indexing done in {total_time/60:.1f} minutes")

# Build BM25 index
print("\n📚 Building BM25 index...")
corpus = [c["text"].lower().split() for c in all_chunks]
bm25 = BM25Okapi(corpus)
with open("bm25_index.pkl", "wb") as f:
    pickle.dump((bm25, [c["text"] for c in all_chunks]), f)
print(f"✅ BM25 index saved ({Path('bm25_index.pkl').stat().st_size/1024/1024:.1f}MB)")

# Test retrieval
print("\n🧪 Testing new retrieval...")
test_query = "what was the impact of the manhattan project"
test_vec = model.encode(f"query: {test_query}", normalize_embeddings=True).tolist()

times = []
for _ in range(20):
    t0 = time.time()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=test_vec,
        limit=5,
        with_payload=True
    ).points
    times.append((time.time() - t0) * 1000)

times.sort()
print(f"⚡ Retrieval P50: {times[10]:.1f}ms | P70: {times[14]:.1f}ms | P100: {times[19]:.1f}ms")
print(f"Top result (score {results[0].score:.4f}): {results[0].payload['text'][:150]}")