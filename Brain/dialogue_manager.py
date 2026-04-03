"""dialogue_manager.py – Confidence-Based Clarification System

Estimates response confidence for a user query and, when confidence falls
below the configured threshold, generates context-aware clarification
questions.  All dialogue state is tracked in an inner-monologue log so the
session history is always transparent.

Public API
----------
DialogueManager(config=None)
    .estimate_confidence(query, context="") -> float          (0.0 – 1.0)
    .needs_clarification(query, context="")  -> bool
    .generate_clarifications(query, context="") -> list[str]
    .record_clarification(query, clarification, refined_query) -> None
    .get_state()  -> dict
    .reset()      -> None
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: Dict = {
    "confidence_threshold": 0.65,
    "max_clarification_questions": 3,
    "state_log_path": None,          # set to a file path to persist the log
    "vague_word_penalty": 0.20,
    "short_query_threshold": 5,      # words – queries shorter than this are penalised
    "short_query_penalty": 0.20,
}

# Words that are known to indicate a vague or under-specified query.
_VAGUE_WORDS = frozenset(
    [
        "thing", "stuff", "it", "that", "this", "something", "anything",
        "whatever", "somehow", "somewhere", "someone", "somebody", "etc",
        "idk", "dunno", "maybe", "perhaps", "possibly", "kind", "sort",
        "type", "way", "how", "what", "why", "when", "who", "where",
    ]
)

# Clarification question templates keyed by detected query category.
_CLARIFICATION_TEMPLATES: Dict[str, List[str]] = {
    "art_style": [
        "Which art style are you referring to – realistic, cartoon, abstract, or another?",
        "Are you asking about a specific artist, era, or movement?",
        "Could you describe a reference work or name a style you have in mind?",
    ],
    "character": [
        "Which character are you asking about – a specific name or role?",
        "Are you interested in the character's backstory, abilities, or design?",
        "Is this character from a particular series, comic, or story?",
    ],
    "story": [
        "Which part of the story are you asking about – plot, theme, or structure?",
        "Are you looking for a summary, analysis, or suggestions for development?",
        "Is this about a published work or something you are creating yourself?",
    ],
    "technical": [
        "Which specific technical aspect would you like help with?",
        "Are you working in a particular programming language or framework?",
        "Can you share the relevant code or error message?",
    ],
    "general": [
        "Could you give me a bit more context about what you are looking for?",
        "What is the end goal you are trying to achieve?",
        "Are you asking for an explanation, an example, or step-by-step guidance?",
    ],
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DialogueTurn:
    timestamp: float
    role: str            # "user" | "system" | "clarification"
    content: str
    confidence: float = 1.0
    category: str = "general"


@dataclass
class DialogueState:
    turns: List[DialogueTurn] = field(default_factory=list)
    pending_clarification: bool = False
    original_query: str = ""
    refined_query: str = ""
    clarification_count: int = 0


# ---------------------------------------------------------------------------
# Core manager
# ---------------------------------------------------------------------------

class DialogueManager:
    """Confidence-based clarification manager for ROCA."""

    def __init__(self, config: Optional[Dict] = None) -> None:
        self._cfg = {**_DEFAULT_CONFIG, **(config or {})}
        self._state = DialogueState()
        self._log_path: Optional[Path] = (
            Path(self._cfg["state_log_path"])
            if self._cfg.get("state_log_path")
            else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_confidence(self, query: str, context: str = "") -> float:
        """Return a confidence score in [0.0, 1.0] for the given query.

        Higher values mean the query is well-specified and the system can
        answer with reasonable confidence.
        """
        score = 1.0
        words = query.strip().split()

        if len(words) < self._cfg["short_query_threshold"]:
            score -= self._cfg["short_query_penalty"]

        lower_words = set(w.lower().strip("?.,!") for w in words)
        vague_hits = lower_words & _VAGUE_WORDS
        score -= len(vague_hits) * self._cfg["vague_word_penalty"]

        # Reward queries that contain specific nouns or proper names.
        if re.search(r"[A-Z][a-z]{2,}", query):
            score += 0.05

        # Reward queries accompanied by context.
        if context.strip():
            score += 0.10

        # Penalise queries that are pure question words with no noun.
        if re.fullmatch(r"(what|how|why|who|where|when)[?.\s]*", query.strip(), re.I):
            score -= 0.25

        return max(0.0, min(1.0, round(score, 4)))

    def needs_clarification(self, query: str, context: str = "") -> bool:
        """Return True when the estimated confidence is below the threshold."""
        return self.estimate_confidence(query, context) < self._cfg["confidence_threshold"]

    def generate_clarifications(
        self, query: str, context: str = ""
    ) -> List[str]:
        """Generate up to *max_clarification_questions* clarification prompts."""
        category = self._detect_category(query, context)
        templates = _CLARIFICATION_TEMPLATES.get(category, _CLARIFICATION_TEMPLATES["general"])
        n = self._cfg["max_clarification_questions"]
        questions = templates[:n]

        self._append_turn(
            role="clarification",
            content="; ".join(questions),
            confidence=self.estimate_confidence(query, context),
            category=category,
        )
        self._state.pending_clarification = True
        self._state.original_query = query
        self._persist_log()
        return questions

    def record_clarification(
        self,
        query: str,
        clarification: str,
        refined_query: str,
    ) -> None:
        """Record a clarification exchange and update the dialogue state."""
        self._append_turn(role="user", content=query)
        self._append_turn(role="user", content=f"[clarification] {clarification}")
        self._append_turn(role="system", content=f"[refined] {refined_query}")

        self._state.pending_clarification = False
        self._state.refined_query = refined_query
        self._state.clarification_count += 1
        self._persist_log()

    def get_state(self) -> Dict:
        """Return the current dialogue state as a plain dict."""
        return {
            "pending_clarification": self._state.pending_clarification,
            "original_query": self._state.original_query,
            "refined_query": self._state.refined_query,
            "clarification_count": self._state.clarification_count,
            "turns": [asdict(t) for t in self._state.turns],
        }

    def reset(self) -> None:
        """Clear the dialogue state (start a new session)."""
        self._state = DialogueState()
        self._persist_log()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_category(self, query: str, context: str = "") -> str:
        combined = (query + " " + context).lower()
        if any(w in combined for w in ("art", "style", "draw", "paint", "sketch", "colour", "color")):
            return "art_style"
        if any(w in combined for w in ("character", "hero", "villain", "protagonist", "antagonist")):
            return "character"
        if any(w in combined for w in ("story", "plot", "narrative", "chapter", "scene", "arc")):
            return "story"
        if any(w in combined for w in ("code", "bug", "error", "function", "class", "import", "module")):
            return "technical"
        return "general"

    def _append_turn(
        self,
        role: str,
        content: str,
        confidence: float = 1.0,
        category: str = "general",
    ) -> None:
        self._state.turns.append(
            DialogueTurn(
                timestamp=time.time(),
                role=role,
                content=content,
                confidence=confidence,
                category=category,
            )
        )

    def _persist_log(self) -> None:
        if self._log_path is None:
            return
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as fh:
            record = {
                "ts": time.time(),
                "state": self.get_state(),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
