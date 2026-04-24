"""Run gender detection on TFBench test split and print statistics."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

from scb.gender import batch_detect  # noqa: E402

CACHE = Path("data/cache/genders.jsonl")
MODEL = "openrouter/openai/gpt-4o-mini"


def main() -> None:
    ds = load_dataset("AWuhrmann/TFBench")["test"]
    records = [{"id": row["prompt_id"], "text": row["text"]} for row in ds]

    print(f"Running gender detection on {len(records)} samples (model: {MODEL})")
    print(f"Cache: {CACHE}")
    results = batch_detect(records, model=MODEL, cache_path=CACHE)

    # --- statistics ---
    no_names = sum(1 for r in results if not r.has_names)
    total_placeholders = sum(len(r.placeholders) for r in results)
    gender_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    pattern_counts: Counter[str] = Counter()  # male-only, female-only, mixed, unknown-only

    for r in results:
        if not r.has_names:
            continue
        genders = {p.gender for p in r.placeholders}
        for p in r.placeholders:
            gender_counts[p.gender] += 1
            confidence_counts[p.confidence] += 1

        if genders == {"unknown"}:
            pattern_counts["unknown-only"] += 1
        elif genders == {"male"}:
            pattern_counts["male-only"] += 1
        elif genders == {"female"}:
            pattern_counts["female-only"] += 1
        elif "unknown" in genders and len(genders) == 2:
            resolved = (genders - {"unknown"}).pop()
            pattern_counts[f"{resolved}+unknown"] += 1
        else:
            pattern_counts["mixed (male+female)"] += 1

    print(f"\n{'='*50}")
    print(f"Total docs:              {len(results)}")
    print(f"Docs with no names:      {no_names}")
    print(f"Docs with names:         {len(results) - no_names}")
    print(f"Total placeholder slots: {total_placeholders}")
    print(f"\nGender distribution (across all placeholders):")
    for gender, count in gender_counts.most_common():
        print(f"  {gender:10s}: {count}")
    print(f"\nConfidence:")
    for conf, count in confidence_counts.most_common():
        print(f"  {conf:10s}: {count}")
    print(f"\nDoc-level pattern (docs with names only):")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"  {pattern:25s}: {count}")


if __name__ == "__main__":
    main()
