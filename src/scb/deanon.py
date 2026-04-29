"""Name substitution and re-anonymization for the bias eval pipeline.

Flow per sample:
  1. sample_substitutions()  — pick gender-appropriate names using the gender cache
  2. apply_substitutions()   — replace A.________ with full names in the prompt
  3. reanonymize()           — reverse step 2 in model output, detecting form changes
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from scb.gender import GenderAnalysis
from scb.models import NameGroup

_NAMES_PATH = Path(__file__).parent.parent.parent / "data" / "names" / "groups.yaml"

# Matches A.________ or A.A.________ — captures the letter token and the full underscore run
_FULL_PLACEHOLDER_RE = re.compile(r"\b([A-Z](?:\.[A-Z])*)(\._+)")

# Honorifics across DE / FR / IT / EN
_TITLE_PAT = (
    r"(?:Mr\.|Mrs\.|Ms\.|Dr\.|M\.|Mme\.?|Mlle\.?|"
    r"Herr|Frau|Sig\.|Sig\.ra|Prof\.|Dott\.)\s+"
)


class NameSubstitution(BaseModel):
    placeholder: str  # e.g. "A."
    placeholder_full: str  # e.g. "A.________" (preserves original underscore count)
    first_name: str
    last_name: str
    gender: Literal["male", "female", "unknown"]

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


class FormChange(BaseModel):
    placeholder: str
    inserted_name: str  # what we put in (e.g. "Moussa Diallo")
    found_form: str  # what the model produced (e.g. "M. Diallo")
    form_type: Literal["title_full", "title_last", "last_only", "first_only"]


def load_name_groups(path: Path = _NAMES_PATH) -> dict[str, NameGroup]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        # Try resolving relative to the project root (parent of src/)
        project_root = Path(__file__).parent.parent.parent
        candidate = project_root / path
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(
            f"Name groups file not found: {path} (cwd={Path.cwd()})"
        )
    raw = yaml.safe_load(path.read_text())
    return {origin: NameGroup(origin=origin, **data) for origin, data in raw.items()}


def _first_names_for_gender(group: NameGroup, gender: str) -> list[str]:
    if gender == "male":
        return group.male_first_names
    if gender == "female":
        return group.female_first_names
    return group.first_names  # unknown: draw from full pool


def sample_substitutions(
    doc_id: str,
    text: str,
    gender_cache: dict[str, GenderAnalysis],
    name_group: NameGroup,
    *,
    seed: int = 42,
) -> list[NameSubstitution]:
    """Sample one name per placeholder, respecting detected gender."""
    rng = random.Random(seed)
    ga = gender_cache.get(doc_id)
    gender_map: dict[str, str] = {p.placeholder: p.gender for p in ga.placeholders} if ga else {}

    seen: dict[str, NameSubstitution] = {}
    for m in _FULL_PLACEHOLDER_RE.finditer(text):
        token = m.group(1) + "."  # e.g. "A." or "A.A."
        full = m.group(1) + m.group(2)  # e.g. "A.________"
        if token in seen:
            continue
        gender = gender_map.get(token, "unknown")
        firsts = _first_names_for_gender(name_group, gender)
        seen[token] = NameSubstitution(
            placeholder=token,
            placeholder_full=full,
            first_name=rng.choice(firsts),
            last_name=rng.choice(name_group.last_names),
            gender=gender,
        )

    return list(seen.values())


def apply_substitutions(text: str, subs: list[NameSubstitution]) -> str:
    """Replace A.________ → First Last in text."""
    result = text
    for sub in subs:
        result = result.replace(sub.placeholder_full, sub.full_name)
    return result


def reanonymize(text: str, subs: list[NameSubstitution]) -> tuple[str, list[FormChange]]:
    """Replace inserted names back to their original placeholders.

    Tries patterns from most to least specific. Logs any match that isn't
    an exact full-name match as a FormChange (the model altered the name form).
    """
    result = text
    form_changes: list[FormChange] = []

    for sub in subs:
        f = re.escape(sub.first_name)
        la = re.escape(sub.last_name)
        target = sub.placeholder_full

        # Ordered from most to least specific — each replacement removes the token
        # so later (weaker) patterns won't double-match.
        patterns: list[tuple[str, str | None]] = [
            (rf"{f}\s+{la}", None),  # full name  (no log)
            (rf"{_TITLE_PAT}{f}\s+{la}", "title_full"),  # Mr. First Last
            (rf"{_TITLE_PAT}{la}", "title_last"),  # Mr. Last
            (rf"\b{la}\b", "last_only"),  # Last alone
            (rf"\b{f}\b", "first_only"),  # First alone
        ]

        for pat, form_type in patterns:

            def _replace(
                m: re.Match,
                ft: str | None = form_type,
                sub: NameSubstitution = sub,
                tgt: str = target,
            ) -> str:
                if ft is not None:
                    form_changes.append(
                        FormChange(
                            placeholder=sub.placeholder,
                            inserted_name=sub.full_name,
                            found_form=m.group(),
                            form_type=ft,  # type: ignore[arg-type]
                        )
                    )
                return tgt

            result = re.sub(pat, _replace, result)

    return result, form_changes
