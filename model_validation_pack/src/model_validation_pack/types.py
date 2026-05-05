"""Core types for the extractable validation pack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ScoredObservation:
    observation_id: str
    score: float
    outcome: int
    period: str
    segment: str = "all"
    features: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationBundle:
    observations: list[ScoredObservation]
    reference_observations: list[ScoredObservation]
    benchmark_scores: list[float] | None = None
    sensitivity_samples: list[dict[str, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
