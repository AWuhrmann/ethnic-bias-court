"""Gender detection for anonymized placeholders in Swiss court documents.

Uses LLM-as-a-judge to infer grammatical gender from contextual signals
(articles, gendered nouns, pronouns) in DE/FR/IT legal text.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Literal

import litellm
from pydantic import BaseModel
from tqdm import tqdm

# Swiss BGer anonymization: A.________, A.A.________, B.X.________ etc.
PLACEHOLDER_RE = re.compile(r"\b([A-Z](?:\.[A-Z])*)\._+")

_SYSTEM = """\
You are analyzing anonymized Swiss Federal Supreme Court decisions (BGer).
Parties are redacted as "A.________", "B.________", "A.A.________" etc.

For each unique placeholder, determine grammatical gender from linguistic cues:
- German: articles (der/die), gendered nouns (Beschuldigter/Beschuldigte,
  Kläger/Klägerin, Beschwerdegegner/Beschwerdegegnerin), pronouns (er/sie)
- French: articles (le/la), past participle agreement, gendered nouns
- Italian: articles (il/la), gendered nouns and adjectives

Return only valid JSON matching the schema exactly. No prose.\
"""

_USER = """\
Unique placeholders found: {placeholders}

Text:
{text}

Return JSON:
{{
  "placeholders": [
    {{
      "placeholder": "<e.g. A. or A.A.>",
      "gender": "male" | "female" | "unknown",
      "confidence": "high" | "low",
      "evidence": "<brief quote or grammatical reason>"
    }}
  ]
}}
"""


class PlaceholderGender(BaseModel):
    placeholder: str
    gender: Literal["male", "female", "unknown"]
    confidence: Literal["high", "low"]
    evidence: str


class GenderAnalysis(BaseModel):
    doc_id: str
    placeholders: list[PlaceholderGender]

    @property
    def has_names(self) -> bool:
        return len(self.placeholders) > 0

    @property
    def genders(self) -> set[str]:
        return {p.gender for p in self.placeholders}


def _extract_placeholders(text: str) -> list[str]:
    """Return unique placeholder tokens in order of first appearance."""
    seen: dict[str, None] = {}
    for match in PLACEHOLDER_RE.finditer(text):
        token = match.group(1) + "."  # restore trailing dot: A → A., A.A → A.A.
        seen[token] = None
    return list(seen)


def detect_genders(
    doc_id: str,
    text: str,
    *,
    model: str = "openrouter/openai/gpt-4o-mini",
) -> GenderAnalysis:
    placeholders = _extract_placeholders(text)

    if not placeholders:
        return GenderAnalysis(doc_id=doc_id, placeholders=[])

    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _USER.format(
                    placeholders=", ".join(placeholders),
                    text=text,
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    raw = json.loads(response.choices[0].message.content)
    # LLM sometimes echoes full form "C.________" instead of short "C." — normalize.
    # LLM may also return duplicate entries for the same placeholder — keep the best
    # (high confidence beats low; known gender beats unknown).
    best: dict[str, PlaceholderGender] = {}
    for p in raw.get("placeholders", []):
        p["placeholder"] = p["placeholder"].rstrip("_")
        entry = PlaceholderGender(**p)
        key = entry.placeholder
        prev = best.get(key)
        if prev is None:
            best[key] = entry
            continue
        prev_score = (prev.confidence == "high", prev.gender != "unknown")
        curr_score = (entry.confidence == "high", entry.gender != "unknown")
        if curr_score > prev_score:
            best[key] = entry
    entries = list(best.values())
    # Ensure every detected placeholder has an entry; fill unknown for any missed
    returned = {e.placeholder for e in entries}
    for ph in placeholders:
        if ph not in returned:
            entries.append(
                PlaceholderGender(
                    placeholder=ph, gender="unknown", confidence="low", evidence="no signal found"
                )
            )
    return GenderAnalysis(doc_id=doc_id, placeholders=entries)


def batch_detect(
    records: list[dict],
    *,
    model: str = "openrouter/openai/gpt-4o-mini",
    cache_path: Path | None = None,
    max_workers: int = 8,
) -> list[GenderAnalysis]:
    """Run gender detection on a list of {'id': ..., 'text': ...} dicts.

    Results are cached to cache_path as JSONL so reruns are free.
    Uncached docs are processed in parallel with up to max_workers threads.
    """
    cache: dict[str, GenderAnalysis] = {}
    if cache_path and cache_path.exists():
        for line in cache_path.read_text().splitlines():
            ga = GenderAnalysis.model_validate_json(line)
            cache[ga.doc_id] = ga

    todo = [rec for rec in records if str(rec["id"]) not in cache]
    results_map: dict[str, GenderAnalysis] = dict(cache)
    write_lock = Lock()

    def _process(rec: dict) -> GenderAnalysis:
        ga = detect_genders(str(rec["id"]), rec["text"], model=model)
        if cache_path:
            with write_lock:
                with cache_path.open("a") as f:
                    f.write(ga.model_dump_json() + "\n")
        return ga

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process, rec): rec for rec in todo}
        with tqdm(total=len(todo), desc="gender detection", unit="doc") as bar:
            for fut in as_completed(futures):
                ga = fut.result()
                results_map[ga.doc_id] = ga
                bar.update()

    return [results_map[str(rec["id"])] for rec in records]
