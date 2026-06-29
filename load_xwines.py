"""
load_xwines.py - load the X-Wines dataset into the local wine RAG index.

Builds one retrieval document per wine (name, type, region, winery, grapes, body,
acidity, ABV, food pairings) and folds in an average star rating computed from the
21M-row ratings file. The resulting index also keeps the curated kb_wine/ knowledge,
so the web chat answers from both the reference base and 100K real wines.

X-Wines: de Azambuja, R.X.; Morais, A.J.; Filipe, V. "X-Wines: A Wine Dataset for
Recommender Systems and Machine Learning." Big Data Cogn. Comput. 2023, 7, 20.
Open license; see https://github.com/rogerioxavier/X-Wines

Usage:
    uv run load_xwines.py --limit 4000 --order ratings        # popular subset
    uv run load_xwines.py --limit 0                           # all 100K (slow)
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PRIVATE_LLM_INDEX", "wine_index.npz")
import rag

csv.field_size_limit(10**7)  # the Vintages column holds long lists

DATA = Path("xwines_data/last")
WINES_CSV = DATA / "XWines_Full_100K_wines.csv"
RATINGS_CSV = DATA / "XWines_Full_21M_ratings.csv"
AGG_CACHE = Path("xwines_agg.json")
KB_WINE = Path("kb_wine")


def parse_list(s: str) -> list[str]:
    try:
        v = ast.literal_eval(s)
        return [str(x) for x in v] if isinstance(v, list) else [str(v)]
    except Exception:
        return []


def rating_aggregates() -> dict[str, list]:
    """Average rating + count per WineID. Cached so we stream the 1 GB file once."""
    if AGG_CACHE.exists():
        print(f"using cached rating aggregates ({AGG_CACHE})")
        return json.loads(AGG_CACHE.read_text())
    if not RATINGS_CSV.exists():
        print("no ratings file; skipping average ratings")
        return {}
    print("streaming 21M ratings to compute averages (one-time)...")
    agg: dict[str, list] = {}
    t0 = time.time()
    with open(RATINGS_CSV, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for n, row in enumerate(reader, 1):
            try:
                wid, val = row[2], float(row[4])
            except (IndexError, ValueError):
                continue
            slot = agg.get(wid)
            if slot:
                slot[0] += val
                slot[1] += 1
            else:
                agg[wid] = [val, 1]
            if n % 5_000_000 == 0:
                print(f"  ...{n:,} ratings ({time.time()-t0:.0f}s)")
    AGG_CACHE.write_text(json.dumps(agg))
    print(f"aggregated {len(agg):,} wines in {time.time()-t0:.0f}s -> {AGG_CACHE}")
    return agg


def wine_doc(row: dict, agg: dict) -> str:
    grapes = ", ".join(parse_list(row["Grapes"]))
    pairings = ", ".join(parse_list(row["Harmonize"]))
    region = ", ".join(x for x in (row.get("RegionName"), row.get("Country")) if x)
    doc = f'{row["WineName"]} - {row["Type"]} wine'
    if region:
        doc += f" from {region}"
    doc += "."
    if row.get("WineryName"):
        doc += f' Winery: {row["WineryName"]}.'
    if grapes:
        doc += f" Grapes: {grapes}."
    sensory = [s for s in (row.get("Body"), f'{row.get("Acidity")} acidity' if row.get("Acidity") else None,
                           f'{row["ABV"]}% ABV' if row.get("ABV") else None) if s]
    if sensory:
        doc += " " + ", ".join(sensory) + "."
    if pairings:
        doc += f" Pairs with: {pairings}."
    slot = agg.get(str(row["WineID"]))
    if slot and slot[1]:
        doc += f" Average rating {slot[0]/slot[1]:.2f}/5 from {slot[1]} ratings."
    return doc


def embed_batch(texts: list[str], tries: int = 5) -> np.ndarray:
    """Embed one batch, retrying transient Ollama timeouts with backoff."""
    for attempt in range(1, tries + 1):
        try:
            return rag.embed(texts)
        except Exception as exc:
            if attempt == tries:
                raise
            wait = 2 * attempt
            print(f"\n  embed retry {attempt}/{tries-1} after error ({type(exc).__name__}); "
                  f"waiting {wait}s", end="")
            time.sleep(wait)


def embed_docs(docs: list[str], batch: int = 64) -> np.ndarray:
    out = []
    t0 = time.time()
    for i in range(0, len(docs), batch):
        out.append(embed_batch(docs[i : i + batch]))
        done = min(i + batch, len(docs))
        print(f"  embedded {done:,}/{len(docs):,} ({time.time()-t0:.0f}s)      ", end="\r")
    print()
    return np.vstack(out)


def curated_chunks() -> tuple[list[str], list[str]]:
    """Re-chunk the curated kb_wine/ files so the index keeps the reference knowledge."""
    texts, sources = [], []
    for path in sorted(KB_WINE.rglob("*")):
        if path.is_file() and path.suffix.lower() in rag.TEXT_EXTS:
            for chunk in rag._chunk(path.read_text(encoding="utf-8", errors="replace")):
                texts.append(chunk)
                sources.append(f"kb_wine/{path.name}")
    return texts, sources


def main() -> None:
    ap = argparse.ArgumentParser(description="Load X-Wines into the wine RAG index.")
    ap.add_argument("--limit", type=int, default=4000, help="Max wines (0 = all 100K).")
    ap.add_argument("--order", choices=["file", "ratings"], default="ratings",
                    help="ratings = most-reviewed wines first (best for a capped run).")
    ap.add_argument("--no-ratings", action="store_true", help="Skip the 21M ratings averages.")
    args = ap.parse_args()

    agg = {} if args.no_ratings else rating_aggregates()

    with open(WINES_CSV, newline="") as f:
        wines = list(csv.DictReader(f))
    print(f"read {len(wines):,} wines")

    if args.order == "ratings" and agg:
        wines.sort(key=lambda r: agg.get(str(r["WineID"]), [0, 0])[1], reverse=True)
    if args.limit:
        wines = wines[: args.limit]

    docs = [wine_doc(w, agg) for w in wines]
    sources = [f'x-wines: {w["WineName"]}' for w in wines]
    print(f"embedding {len(docs):,} X-Wines documents...")
    wine_vecs = embed_docs(docs)

    c_texts, c_sources = curated_chunks()
    print(f"embedding {len(c_texts)} curated chunks...")
    c_vecs = embed_docs(c_texts) if c_texts else np.empty((0, wine_vecs.shape[1]), np.float32)

    vectors = np.vstack([wine_vecs, c_vecs])
    texts = np.array(docs + c_texts, dtype=object)
    src = np.array(sources + c_sources, dtype=object)
    np.savez(rag.INDEX_PATH, vectors=vectors, texts=texts, sources=src)
    print(f"\nsaved {len(texts):,} documents -> {rag.INDEX_PATH}  "
          f"({len(docs):,} X-Wines + {len(c_texts)} curated)")


if __name__ == "__main__":
    main()
