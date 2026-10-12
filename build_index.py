from __future__ import annotations

import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from config import (
    JSONL_PATH,
    DATABASE_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
)


# ============================================================
# BATCH READER
# ============================================================

def read_batches(
    jsonl_path: Path,
    batch_size: int = 64,
):

    batch = []

    with jsonl_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            batch.append(
                json.loads(line)
            )

            if len(batch) >= batch_size:

                yield batch

                batch = []

    if batch:
        yield batch


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING LOCAL CHROMADB")
    print("=" * 70)

    # --------------------------------------------------------
    # Check JSONL
    # --------------------------------------------------------

    if not JSONL_PATH.exists():

        print()
        print(
            "ERROR: manual_chunks.jsonl not found."
        )

        print()
        print(
            "Run:"
        )

        print(
            "python scripts\\convert_manual.py"
        )

        return 1

    DATABASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Count records
    # --------------------------------------------------------

    print()
    print("Counting chunks...")

    total_records = 0

    with JSONL_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if line.strip():
                total_records += 1

    print(
        f"Total chunks: "
        f"{total_records:,}"
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print()
    print(
        f"Loading embedding model:"
    )

    print(
        EMBED_MODEL
    )

    model = SentenceTransformer(
        EMBED_MODEL
    )

    # --------------------------------------------------------
    # Chroma
    # --------------------------------------------------------

    client = chromadb.PersistentClient(
        path=str(
            DATABASE_DIR
        )
    )

    # Delete old collection if it exists
    try:

        client.delete_collection(
            COLLECTION_NAME
        )

        print(
            "Old Chroma collection deleted."
        )

    except Exception:

        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        },
    )

    # --------------------------------------------------------
    # Embed and insert
    # --------------------------------------------------------

    batch_size = 64

    processed = 0

    for batch in tqdm(
        read_batches(
            JSONL_PATH,
            batch_size,
        ),
        total=(
            total_records
            + batch_size
            - 1
        )
        // batch_size,
        desc="Embedding",
        unit="batch",
    ):

        documents = [
            item["text"]
            for item in batch
        ]

        ids = [
            item["id"]
            for item in batch
        ]

        embeddings = model.encode(
            documents,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        metadatas = []

        for item in batch:

            metadatas.append(
                {
                    "vehicle": str(
                        item.get(
                            "vehicle",
                            "",
                        )
                    ),

                    "year": str(
                        item.get(
                            "year",
                            "",
                        )
                    ),

                    "model": str(
                        item.get(
                            "model",
                            "",
                        )
                    ),

                    "manual": str(
                        item.get(
                            "manual",
                            "",
                        )
                    ),

                    "title": str(
                        item.get(
                            "title",
                            "",
                        )
                    ),

                    "section": str(
                        item.get(
                            "section",
                            "",
                        )
                    ),

                    "source_file": str(
                        item.get(
                            "source_file",
                            "",
                        )
                    ),

                    "dtc_codes": ", ".join(
                        item.get(
                            "dtc_codes",
                            [],
                        )
                    ),

                    "images": " | ".join(
                        item.get(
                            "images",
                            [],
                        )
                    ),
                }
            )

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        processed += len(batch)

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INDEX COMPLETE")
    print("=" * 70)

    print(
        f"Chunks indexed: "
        f"{collection.count():,}"
    )

    print()
    print(
        "Database:"
    )

    print(
        DATABASE_DIR
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )