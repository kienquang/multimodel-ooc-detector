"""
schemas.py — Pydantic models for S-V-O structured output

Critical fix: relation field now has a real validator (not just a description hint).
Without this, LLMs can return any string ("IS_IN", "HAPPENED_AT", etc.) and
Pydantic silently accepts it — breaking RetrievalAgent's key-based conflict detection.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List

RELATIONS = ["PERFORMS", "LOCATED_IN", "OCCURRED_ON", "TARGETS", "HAS_STATE", "SAME_AS"]

# Aliases the LLM might use — map them to the canonical relation
# This is more robust than strict rejection (LLMs paraphrase relations)
_RELATION_ALIASES: dict[str, str] = {
    # LOCATED_IN aliases
    "located_in":    "LOCATED_IN",
    "located at":    "LOCATED_IN",
    "is in":         "LOCATED_IN",
    "is at":         "LOCATED_IN",
    "held in":       "LOCATED_IN",
    "took place in": "LOCATED_IN",
    "happened in":   "LOCATED_IN",
    # OCCURRED_ON aliases
    "occurred_on":   "OCCURRED_ON",
    "occurred on":   "OCCURRED_ON",
    "happened on":   "OCCURRED_ON",
    "took place on": "OCCURRED_ON",
    "dated":         "OCCURRED_ON",
    "on":            "OCCURRED_ON",
    # PERFORMS aliases
    "performs":      "PERFORMS",
    "did":           "PERFORMS",
    "conducted":     "PERFORMS",
    "participated":  "PERFORMS",
    "attended":      "PERFORMS",
    "gave":          "PERFORMS",
    # TARGETS aliases
    "targets":       "TARGETS",
    "aimed at":      "TARGETS",
    "directed at":   "TARGETS",
    "against":       "TARGETS",
    "for":           "TARGETS",
    # HAS_STATE aliases
    "has_state":     "HAS_STATE",
    "has state":     "HAS_STATE",
    "has":           "HAS_STATE",
    "contains":      "HAS_STATE",
    "shows":         "HAS_STATE",
    "depicts":       "HAS_STATE",
    # SAME_AS aliases
    "same_as":       "SAME_AS",
    "same as":       "SAME_AS",
    "also known as": "SAME_AS",
    "aka":           "SAME_AS",
    "is":            "SAME_AS",
}


class SVOTriplet(BaseModel):
    subject:  str = Field(..., description="Main entity (person, organization, or scene)")
    relation: str = Field(..., description=f"Must be one of: {RELATIONS}")
    object:   str = Field(..., description="Value, target, or attribute")

    @field_validator("relation", mode="before")
    @classmethod
    def normalize_relation(cls, v: str) -> str:
        """
        Normalize relation to one of the 6 canonical EGMMG relations.

        Strategy:
          1. Exact match (case-insensitive) → return canonical
          2. Alias map → return canonical
          3. Substring match → return first containing relation
          4. Unknown → default to HAS_STATE (weakest claim, least harmful)
             and log a warning so we can improve the alias map
        """
        if not isinstance(v, str):
            return "HAS_STATE"

        normalized = v.strip().upper()

        # Exact match
        if normalized in RELATIONS:
            return normalized

        # Alias map (lowercase lookup)
        lower = v.strip().lower()
        if lower in _RELATION_ALIASES:
            return _RELATION_ALIASES[lower]

        # Substring match (e.g. "IS_LOCATED_IN" → "LOCATED_IN")
        for canonical in RELATIONS:
            if canonical in normalized:
                return canonical

        # Last resort: log and default to HAS_STATE
        print(
            f"[Schema] WARNING: Unknown relation '{v}'. "
            f"Defaulting to HAS_STATE. Add to _RELATION_ALIASES if common."
        )
        return "HAS_STATE"

    @field_validator("subject", "object", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else str(v)


class SVOList(BaseModel):
    triplets: List[SVOTriplet] = Field(
        default_factory=list,
        description="List of S-V-O triplets extracted from text"
    )

    def to_display(self) -> str:
        """Human-readable format for logging and debugging."""
        if not self.triplets:
            return "(empty)"
        return "\n".join(
            f"  {t.subject}  --[{t.relation}]-->  {t.object}"
            for t in self.triplets
        )