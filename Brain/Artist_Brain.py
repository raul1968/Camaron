"""Artist_Brain.py – ROCA ChatBot Integration

Ties together the DialogueManager (confidence-based clarification) and the
QuestionDatabase (question learning) to form the main reasoning layer for the
ROCA chatbot.

Usage
-----
    from Brain.Artist_Brain import ArtistBrain

    brain = ArtistBrain()
    result = brain.respond("draw me something")
    # result.needs_clarification == True
    # result.clarification_questions == [...]

    # After the user provides clarification:
    result2 = brain.respond(
        "draw me something",
        clarification="a watercolour landscape with mountains",
    )

    # Rate how helpful a clarification question was:
    brain.rate_question(question_id, rating=4)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from Brain.dialogue_manager import DialogueManager
from Brain.question_database import QuestionDatabase, QuestionRecord


# ---------------------------------------------------------------------------
# Response container
# ---------------------------------------------------------------------------

@dataclass
class BrainResponse:
    query: str
    answer: str
    confidence: float
    needs_clarification: bool
    clarification_questions: List[str] = field(default_factory=list)
    question_ids: List[str] = field(default_factory=list)
    category: str = "general"
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# ArtistBrain
# ---------------------------------------------------------------------------

class ArtistBrain:
    """Main reasoning layer for the ROCA chatbot.

    Parameters
    ----------
    dialogue_config : dict, optional
        Overrides for DialogueManager defaults (e.g. confidence_threshold).
    db_path : str, optional
        File path for the persistent QuestionDatabase.  Defaults to an
        in-memory-only database.
    personality : str
        Display name / persona for this brain instance.
    """

    def __init__(
        self,
        dialogue_config: Optional[Dict] = None,
        db_path: Optional[str] = None,
        personality: str = "ROCA",
    ) -> None:
        self.personality = personality
        self._dialogue = DialogueManager(config=dialogue_config)
        self._qdb = QuestionDatabase(db_path=db_path)

    # ------------------------------------------------------------------
    # Core respond method
    # ------------------------------------------------------------------

    def respond(
        self,
        query: str,
        context: str = "",
        clarification: str = "",
    ) -> BrainResponse:
        """Process a user query and return a BrainResponse.

        If *clarification* is provided the dialogue manager records the
        exchange and a refined answer is attempted.  Otherwise confidence is
        estimated; when it is below the threshold, clarification questions are
        returned instead of a direct answer.

        Parameters
        ----------
        query : str
            The user's latest message.
        context : str
            Optional background context (conversation history, document text…).
        clarification : str
            The user's answer to a previous clarification question.
        """
        # Record a clarification exchange if one was pending.
        if clarification and self._dialogue.get_state()["pending_clarification"]:
            refined = f"{query} [{clarification}]"
            self._dialogue.record_clarification(query, clarification, refined)
            answer = self._compose_answer(refined, context)
            confidence = self._dialogue.estimate_confidence(refined, context)
            return BrainResponse(
                query=query,
                answer=answer,
                confidence=confidence,
                needs_clarification=False,
                category=self._dialogue._detect_category(refined, context),
            )

        confidence = self._dialogue.estimate_confidence(query, context)

        if self._dialogue.needs_clarification(query, context):
            questions = self._dialogue.generate_clarifications(query, context)
            category = self._dialogue._detect_category(query, context)

            # Learn from the best historical questions for this category.
            best = self._qdb.get_best_questions(category, top_k=len(questions))
            best_records = {r.question_text: r for r in best}
            # Prefer proven questions over template questions where available.
            final_questions = list(best_records.keys()) if best_records else questions

            # Log only newly generated template questions; historical ones already
            # have IDs in the database and should not be duplicated.
            qids = []
            for q in final_questions:
                if q in best_records:
                    qids.append(best_records[q].question_id)
                else:
                    qids.append(self._qdb.log_question(q, category, query_text=query))

            return BrainResponse(
                query=query,
                answer="",
                confidence=confidence,
                needs_clarification=True,
                clarification_questions=final_questions,
                question_ids=qids,
                category=category,
            )

        answer = self._compose_answer(query, context)
        return BrainResponse(
            query=query,
            answer=answer,
            confidence=confidence,
            needs_clarification=False,
            category=self._dialogue._detect_category(query, context),
        )

    # ------------------------------------------------------------------
    # Feedback / learning
    # ------------------------------------------------------------------

    def rate_question(self, question_id: str, rating: int) -> None:
        """Record a 1-5 helpfulness rating for a previous clarification question.

        The QuestionDatabase uses this feedback to rank questions for future
        queries of the same category.
        """
        self._qdb.record_feedback(question_id, rating)

    def question_analytics(self) -> Dict:
        """Return aggregate analytics from the QuestionDatabase."""
        return self._qdb.analytics()

    def best_questions_for(self, query_type: str, top_k: int = 3) -> List[QuestionRecord]:
        """Return the top-ranked stored questions for a query category."""
        return self._qdb.get_best_questions(query_type, top_k=top_k)

    # ------------------------------------------------------------------
    # Dialogue state helpers
    # ------------------------------------------------------------------

    def reset_dialogue(self) -> None:
        """Start a fresh conversation (clears inner-monologue turns)."""
        self._dialogue.reset()

    def dialogue_state(self) -> Dict:
        """Return the raw dialogue state for inspection or logging."""
        return self._dialogue.get_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compose_answer(self, query: str, context: str) -> str:
        """Compose a placeholder answer.  Replace with LLM / model call."""
        parts = [f"[{self.personality}] Addressing: \"{query}\""]
        if context.strip():
            excerpt = context.strip()[:200]
            suffix = "..." if len(context) > 200 else ""
            parts.append(f"Context: \"{excerpt}{suffix}\"")
        parts.append(
            "I have retrieved the most relevant knowledge capsules for your query. "
            "Please integrate this with your domain knowledge to form a full response."
        )
        return "\n".join(parts)
