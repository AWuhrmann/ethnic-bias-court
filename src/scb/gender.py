"""Gender detection for anonymized placeholders in Swiss court documents.

Uses LLM-as-a-judge to infer grammatical gender from contextual signals
(articles, gendered nouns, pronouns) in DE/FR/IT legal text.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import litellm
from pydantic import BaseModel

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
    entries = [PlaceholderGender(**p) for p in raw.get("placeholders", [])]
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
) -> list[GenderAnalysis]:
    """Run gender detection on a list of {'id': ..., 'text': ...} dicts.

    Results are cached to cache_path as JSONL so reruns are free.
    """
    cache: dict[str, GenderAnalysis] = {}
    if cache_path and cache_path.exists():
        for line in cache_path.read_text().splitlines():
            ga = GenderAnalysis.model_validate_json(line)
            cache[ga.doc_id] = ga

    results: list[GenderAnalysis] = []
    for rec in records:
        doc_id = str(rec["id"])
        if doc_id in cache:
            results.append(cache[doc_id])
            continue
        ga = detect_genders(doc_id, rec["text"], model=model)
        results.append(ga)
        if cache_path:
            with cache_path.open("a") as f:
                f.write(ga.model_dump_json() + "\n")

    return results
