import pyarrow.parquet as pq
import json
import re
from pathlib import Path

PARQUET_PATH = r"C:\Users\itzsa\.cache\huggingface\hub\datasets--ai4bharat--MSMARCO-XI\snapshots\bf5cdc1f26e581e519018e434db14edd1b77602b\train\hintrain.parquet"
OUTPUT_PATH = "chunks.json"
MAX_ROWS = 5000  # ~50k passages, plenty for a strong submission

def split_sentences(text):
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]

def sliding_window(sentences, window=2, overlap=1):
    """Generate overlapping windows of sentences."""
    chunks = []
    step = window - overlap
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i+window])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def chunk_row(row_idx, query_id, query_type, eng_query, passages):
    chunks = []
    eng_passages = passages.get("English_passages") or []
    hin_passages = passages.get("Translated_passages") or []
    is_selected = passages.get("is_selected") or []

    for p_idx, passage in enumerate(eng_passages):
        if not passage or len(passage) < 30:
            continue

        selected = bool(is_selected[p_idx]) if p_idx < len(is_selected) else False
        base_meta = {
            "query_id": query_id,
            "passage_idx": p_idx,
            "is_selected": selected,
            "query_type": query_type,
            "source_query": eng_query,
        }

        # Strategy 1: Full passage (English)
        chunks.append({
            "text": passage,
            "lang": "en",
            "strategy": "full_passage",
            **base_meta
        })

        # Strategy 2: Sentence-level (English)
        sentences = split_sentences(passage)
        for s_idx, sent in enumerate(sentences):
            chunks.append({
                "text": sent,
                "lang": "en",
                "strategy": "sentence",
                "sentence_idx": s_idx,
                **base_meta
            })

        # Strategy 3: Sliding window (English)
        if len(sentences) >= 2:
            for window_chunk in sliding_window(sentences):
                chunks.append({
                    "text": window_chunk,
                    "lang": "en",
                    "strategy": "sliding_window",
                    **base_meta
                })

        # Strategy 4: Bilingual pair (English + Hindi together)
        if p_idx < len(hin_passages) and hin_passages[p_idx]:
            bilingual = f"{passage} | {hin_passages[p_idx]}"
            chunks.append({
                "text": bilingual,
                "lang": "bilingual",
                "strategy": "bilingual_pair",
                **base_meta
            })

    return chunks

# ── MAIN ──
print("⏳ Reading parquet in batches...")
pf = pq.ParquetFile(PARQUET_PATH)

all_chunks = []
row_count = 0

for batch in pf.iter_batches(batch_size=100):
    d = batch.to_pydict()
    batch_len = len(d["query_id"])

    for i in range(batch_len):
        if row_count >= MAX_ROWS:
            break

        passages = {
            "English_passages": d["passages"][i].get("English_passages", []),
            "Translated_passages": d["passages"][i].get("Translated_passages", []),
            "is_selected": d["passages"][i].get("is_selected", []),
        }

        row_chunks = chunk_row(
            row_idx=row_count,
            query_id=d["query_id"][i],
            query_type=d["query_type"][i],
            eng_query=d["Eng_Query"][i],
            passages=passages,
        )
        all_chunks.extend(row_chunks)
        row_count += 1

        if row_count % 500 == 0:
            print(f"  Processed {row_count} rows → {len(all_chunks)} chunks so far")

    if row_count >= MAX_ROWS:
        break

print(f"\n✅ Done! {row_count} rows → {len(all_chunks)} total chunks")

# Save
print(f"💾 Saving to {OUTPUT_PATH}...")
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print(f"✅ Saved! File size: {Path(OUTPUT_PATH).stat().st_size / 1024 / 1024:.1f} MB")

# Stats breakdown
from collections import Counter
strategy_counts = Counter(c["strategy"] for c in all_chunks)
print("\n📊 Chunks by strategy:")
for strategy, count in strategy_counts.items():
    print(f"  {strategy}: {count:,}")