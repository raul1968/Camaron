# ROCA Clarification System – Complete Guide

**Component**: Confidence-Based Clarification System  
**Status**: ✅ Production Ready  
**Last Updated**: 2026-02-08

---

## Overview

The Clarification System prevents ROCA from giving vague or incorrect answers
by detecting when a query is under-specified and asking targeted follow-up
questions before attempting a response.

---

## Architecture

```
User query
    │
    ▼
DialogueManager.estimate_confidence(query, context)
    │        returns float in [0.0, 1.0]
    │
    ├── confidence ≥ 0.65 ──► ArtistBrain._compose_answer()  ──► BrainResponse (answer)
    │
    └── confidence < 0.65 ──► DialogueManager.generate_clarifications()
                                    │
                                    ▼
                              QuestionDatabase.get_best_questions()
                              (use proven questions when available)
                                    │
                                    ▼
                              QuestionDatabase.log_question()
                              (register new questions for future feedback)
                                    │
                                    ▼
                              BrainResponse (needs_clarification=True,
                                             clarification_questions=[...],
                                             question_ids=[...])
                                    │
                          User provides clarification
                                    │
                                    ▼
                          ArtistBrain.respond(query, clarification=…)
                                    │
                                    ▼
                          DialogueManager.record_clarification()
                          (stores in inner monologue)
                                    │
                                    ▼
                              BrainResponse (answer)
```

---

## Key Files

| File | Responsibility |
|------|---------------|
| `Brain/dialogue_manager.py` | Confidence estimation, clarification generation, inner-monologue tracking |
| `Brain/question_database.py` | Persistent question store with 1-5 ratings and analytics |
| `Brain/Artist_Brain.py` | ChatBot integration: ties dialogue manager and question database together |
| `Docs/CLARIFICATION_SYSTEM.md` | This guide |

---

## Confidence Scoring

The `DialogueManager.estimate_confidence()` method scores a query on a
**0.0 – 1.0** scale using heuristics:

| Signal | Effect |
|--------|--------|
| Query shorter than 5 words | −0.20 |
| Each vague word (e.g. "thing", "it", "maybe") | −0.20 |
| Contains a proper noun (e.g. a name) | +0.05 |
| Context string provided by caller | +0.10 |
| Query is a bare question word ("what?", "why?") | −0.25 |

The default threshold is **0.65**.  Change it via `dialogue_config`:

```python
brain = ArtistBrain(dialogue_config={"confidence_threshold": 0.70})
```

---

## Clarification Question Templates

Templates are grouped by detected query category:

| Category | Trigger keywords |
|----------|-----------------|
| `art_style` | art, style, draw, paint, sketch, colour |
| `character` | character, hero, villain, protagonist |
| `story` | story, plot, narrative, chapter, scene |
| `technical` | code, bug, error, function, class, import |
| `general` | (fallback) |

---

## Question Learning System

Every clarification question is stored in the `QuestionDatabase` with a
unique ID.  After the user sees the question, the application can record a
**1-5 helpfulness rating**:

```python
brain.rate_question(question_id, rating=4)
```

The database computes a **Bayesian quality score** for each question and
returns the highest-scoring questions first for future similar queries.

### Analytics

```python
stats = brain.question_analytics()
# {
#   "total_questions": 42,
#   "total_ratings": 35,
#   "overall_average_rating": 3.97,
#   "by_category": {
#     "art_style": {
#       "question_count": 12,
#       "rated_count": 10,
#       "average_rating": 4.2,
#       "best_question": "Which art style are you referring to…"
#     },
#     …
#   }
# }
```

---

## Quick-Start Example

```python
from Brain.Artist_Brain import ArtistBrain

brain = ArtistBrain(
    db_path="Json/question_learning.json",  # persists feedback across sessions
    personality="ROCA",
)

# 1 – Vague query triggers clarification
result = brain.respond("draw something nice")
assert result.needs_clarification is True
print(result.clarification_questions)
# ["Which art style are you referring to…", …]

# 2 – User answers; brain produces a real response
result2 = brain.respond(
    "draw something nice",
    clarification="a pen-and-ink architectural sketch",
)
assert result2.needs_clarification is False
print(result2.answer)

# 3 – User rates question helpfulness
brain.rate_question(result.question_ids[0], rating=5)

# 4 – Inspect analytics
print(brain.question_analytics())
```

---

## Inner Monologue

All clarification exchanges are recorded as `DialogueTurn` objects inside
`DialogueManager`.  Retrieve the full log at any time:

```python
state = brain.dialogue_state()
for turn in state["turns"]:
    print(turn["role"], turn["content"])
```

Optionally persist the log to a `.jsonl` file:

```python
brain = ArtistBrain(
    dialogue_config={"state_log_path": "brain_state/dialogue.jsonl"},
)
```

---

## Extending the System

* **Add new category templates**: edit `_CLARIFICATION_TEMPLATES` in
  `Brain/dialogue_manager.py`.
* **Swap the answer composer**: replace `ArtistBrain._compose_answer()` with
  a call to an LLM API or local model.
* **Adjust confidence weights**: pass a custom `dialogue_config` dict to
  `ArtistBrain(dialogue_config={…})`.
