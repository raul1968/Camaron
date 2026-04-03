"""question_database.py – Question Learning System

Stores clarification questions together with user helpfulness ratings
(1-5 scale), computes quality analytics, and surfaces the best-performing
questions for similar future queries.

Public API
----------
QuestionDatabase(db_path=None)
    .log_question(question, query_type, query_text="")                     -> str  (question_id)
    .record_feedback(question_id, rating)                                  -> None
    .get_best_questions(query_type, top_k=3)                               -> list[QuestionRecord]
    .analytics()                                                           -> dict
    .all_records()                                                         -> list[QuestionRecord]
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QuestionRecord:
    question_id: str
    question_text: str
    query_type: str              # e.g. "art_style", "technical", "general"
    query_text: str              # the original user query that triggered this question
    created_at: float
    ratings: List[int] = field(default_factory=list)   # individual 1-5 scores

    # Derived – computed on access
    @property
    def rating_count(self) -> int:
        return len(self.ratings)

    @property
    def average_rating(self) -> float:
        if not self.ratings:
            return 0.0
        return round(sum(self.ratings) / len(self.ratings), 4)

    @property
    def quality_score(self) -> float:
        """Bayesian-style quality score that down-weights questions with very
        few ratings so that a single 5-star rating doesn't dominate."""
        prior_mean = 3.0    # neutral prior
        prior_strength = 2  # equivalent to 2 prior ratings
        n = self.rating_count
        total = sum(self.ratings)
        return round(
            (prior_mean * prior_strength + total) / (prior_strength + n), 4
        )


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

class QuestionDatabase:
    """Persistent, file-backed store for question feedback and analytics."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._path: Optional[Path] = Path(db_path) if db_path else None
        self._records: Dict[str, QuestionRecord] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_question(
        self,
        question: str,
        query_type: str,
        query_text: str = "",
    ) -> str:
        """Register a new clarification question and return its ID."""
        qid = str(uuid.uuid4())
        record = QuestionRecord(
            question_id=qid,
            question_text=question,
            query_type=query_type,
            query_text=query_text,
            created_at=time.time(),
        )
        self._records[qid] = record
        self._save()
        return qid

    def record_feedback(self, question_id: str, rating: int) -> None:
        """Attach a 1-5 helpfulness rating to a previously logged question.

        Raises
        ------
        KeyError  – if *question_id* is unknown.
        ValueError – if *rating* is outside [1, 5].
        """
        if question_id not in self._records:
            raise KeyError(f"Unknown question_id: {question_id!r}")
        if not (1 <= rating <= 5):
            raise ValueError(f"Rating must be between 1 and 5, got {rating}")
        self._records[question_id].ratings.append(int(rating))
        self._save()

    def get_best_questions(
        self, query_type: str, top_k: int = 3
    ) -> List[QuestionRecord]:
        """Return the top-*k* highest-quality questions for *query_type*."""
        matching = [
            r for r in self._records.values() if r.query_type == query_type
        ]
        matching.sort(key=lambda r: r.quality_score, reverse=True)
        return matching[:top_k]

    def analytics(self) -> Dict:
        """Return aggregate analytics across all stored questions."""
        if not self._records:
            return {
                "total_questions": 0,
                "total_ratings": 0,
                "overall_average_rating": 0.0,
                "by_category": {},
            }

        total_ratings = sum(r.rating_count for r in self._records.values())
        all_ratings = [v for r in self._records.values() for v in r.ratings]
        overall_avg = (
            round(sum(all_ratings) / len(all_ratings), 4)
            if all_ratings
            else 0.0
        )

        by_category: Dict[str, Dict] = {}
        for record in self._records.values():
            cat = record.query_type
            entry = by_category.setdefault(
                cat,
                {"question_count": 0, "rated_count": 0, "average_rating": 0.0, "best_question": ""},
            )
            entry["question_count"] += 1
            if record.rating_count > 0:
                entry["rated_count"] += 1

        for cat, entry in by_category.items():
            best = self.get_best_questions(cat, top_k=1)
            if best:
                entry["best_question"] = best[0].question_text
                rated_in_cat = [
                    r for r in self._records.values()
                    if r.query_type == cat and r.rating_count > 0
                ]
                if rated_in_cat:
                    entry["average_rating"] = round(
                        sum(r.average_rating for r in rated_in_cat) / len(rated_in_cat), 4
                    )

        return {
            "total_questions": len(self._records),
            "total_ratings": total_ratings,
            "overall_average_rating": overall_avg,
            "by_category": by_category,
        }

    def all_records(self) -> List[QuestionRecord]:
        """Return all stored question records."""
        return list(self._records.values())

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": time.time(),
            "records": [
                {**asdict(r), "ratings": r.ratings}
                for r in self._records.values()
            ],
        }
        self._path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for item in payload.get("records", []):
            try:
                ratings = item.pop("ratings", [])
                record = QuestionRecord(**item)
                record.ratings = [int(r) for r in ratings if 1 <= int(r) <= 5]
                self._records[record.question_id] = record
            except (TypeError, KeyError):
                continue
