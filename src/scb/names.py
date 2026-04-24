"""De-anonymization utilities for the old Document-based API (kept for compatibility)."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from pathlib import Path

import yaml

from scb.models import DeanonymizedDoc, Document, NameGroup, Substitution

_NAMES_PATH = Path(__file__).parent.parent.parent / "data" / "names" / "groups.yaml"


def load_name_groups(path: Path = _NAMES_PATH) -> dict[str, NameGroup]:
    raw = yaml.safe_load(path.read_text())
    return {origin: NameGroup(origin=origin, **data) for origin, data in raw.items()}


def deanonymize(
    doc: Document,
    name_group: NameGroup,
    *,
    placeholder_pattern: str = r"\b([A-Z]\.(?:\s[A-Z]\.)*)",
    seed: int | None = None,
) -> DeanonymizedDoc:
    rng = random.Random(seed)
    placeholders: list[str] = sorted(
        set(re.findall(placeholder_pattern, doc.text)),
        key=lambda p: doc.text.index(p),
    )

    substitutions: list[Substitution] = []
    mapping: dict[str, str] = {}

    for placeholder in placeholders:
        first = rng.choice(name_group.first_names)
        last = rng.choice(name_group.last_names)
        substitutions.append(
            Substitution(placeholder=placeholder, first_name=first, last_name=last)
        )
        mapping[placeholder] = f"{first} {last}"

    text = doc.text
    for placeholder, full_name in mapping.items():
        text = re.sub(re.escape(placeholder), full_name, text)

    return DeanonymizedDoc(
        original=doc,
        deanonymized_text=text,
        substitutions=substitutions,
        name_origin=name_group.origin,
    )


def deanonymize_all_origins(
    doc: Document,
    name_groups: dict[str, NameGroup] | None = None,
    *,
    origins: Sequence[str] | None = None,
    seed: int | None = None,
) -> list[DeanonymizedDoc]:
    if name_groups is None:
        name_groups = load_name_groups()
    selected = {k: v for k, v in name_groups.items() if origins is None or k in origins}
    return [deanonymize(doc, group, seed=seed) for group in selected.values()]
