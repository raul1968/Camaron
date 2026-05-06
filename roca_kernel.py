#!/usr/bin/env python3
"""
ROCA (Routed Orbital Capsule Architecture) — Unified Kernel v2.0
================================================================
Deterministic capsule-based cognitive engine built around:
  • Semantic Capsules (core data units)
  • Routing-by-Agreement (consensus computation)
  • Orbital hierarchy (0 = core cognition, 3 = raw input)
  • Gravity-based relevance weighting
  • Temporal graph memory (frame-based persistence)
  • Hardware-aware adaptive execution

This file merges the ROCAEnhancedAIAssistant architecture from
enhanced_ai_assistant.py with the ROCA Codex Build Guide v1.3 spec,
producing a single-file kernel that can boot, route, and persist
without any neural-network dependency.

Author: ROCA Codex Migration
Date:   2026-05-05
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union,
)

import numpy as np

# ---------------------------------------------------------------------------
# 0.  Project root & JSON directory helpers
# ---------------------------------------------------------------------------

def _resolve_project_root() -> Path:
    """Return the parent directory of the *roca/* package root."""
    return Path(__file__).resolve().parent.parent

PROJECT_ROOT: Path = _resolve_project_root()
JSON_DIR: Path = PROJECT_ROOT / "Json"
JSON_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_KNOWLEDGE_BASE_PATH: Path = JSON_DIR / "roca_knowledge_base.json"
DEFAULT_TEMPORAL_STORE_PATH: Path = JSON_DIR / "roca_timeline.jsonl"

# ---------------------------------------------------------------------------
# 1.  Core Enums (aligned with Build Guide §8 and enhanced_assistant lanes)
# ---------------------------------------------------------------------------

class CapsuleKind(str, Enum):
    """Kind of capsule — shapes routing behaviour and orbit decay."""
    NUCLEUS      = "nucleus"
    CHARACTER    = "character"
    STYLE        = "style"
    SKILL        = "skill"
    MEMORY       = "memory"
    WORKFLOW     = "workflow"
    TOPIC        = "topic"
    EXPERIMENTAL = "experimental"

class LanePosition(str, Enum):
    """Radial orbit lane — maps to orbit_level 0-3 per Build Guide §8."""
    INNER        = "inner"        # orbit_level 0
    MIDDLE       = "middle"       # orbit_level 1
    OUTER        = "outer"        # orbit_level 2
    EXPERIMENTAL = "experimental" # orbit_level 3

# Orbit-level ↔ LanePosition map
ORBIT_LEVEL_TO_LANE: Dict[int, LanePosition] = {
    0: LanePosition.INNER,
    1: LanePosition.MIDDLE,
    2: LanePosition.OUTER,
    3: LanePosition.EXPERIMENTAL,
}
LANE_TO_ORBIT_LEVEL: Dict[LanePosition, int] = {v: k for k, v in ORBIT_LEVEL_TO_LANE.items()}

# Lane radius bounds (for visualisation / orbit_radius calculations)
LANE_BOUNDARIES: Dict[LanePosition, Tuple[float, float]] = {
    LanePosition.INNER:        (0.0,  0.25),
    LanePosition.MIDDLE:       (0.25, 0.50),
    LanePosition.OUTER:        (0.50, 0.75),
    LanePosition.EXPERIMENTAL: (0.75, 1.00),
}

# ---------------------------------------------------------------------------
# 2.  Capsule (Build Guide §5 + enhanced_assistant Capsule dataclass)
# ---------------------------------------------------------------------------

@dataclass
class Capsule:
    """Single ROCA capsule — the fundamental unit of cognition.

    Build Guide §5 – Every capsule MUST implement:
      • id, type, state, confidence, timestamp
      • orbit_level, gravity, lane, hibernating
      • evaluate_agreement(other) → float
      • merge(other) → Capsule
    """
    id: str
    kind: CapsuleKind
    name: str

    # --- Core identity fields ---
    state: Dict[str, Any] = field(default_factory=dict)          # arbitrary payload
    confidence: float = 0.5
    usage_count: int = 0

    # --- Temporal fields ---
    created_at: datetime = field(default_factory=datetime.now)
    last_used_at: datetime = field(default_factory=datetime.now)

    # --- Orbital / gravity fields ---
    orbit_level: int = 2                # 0-3 (Build Guide §8)
    lane: LanePosition = LanePosition.OUTER
    orbit_radius: float = 0.75          # normalised 0-1
    orbit_angle: float = 0.0            # radians 0-2π
    gravity_score: float = 0.1          # salience (Build Guide §7)
    hibernating: bool = False

    # --- Merge / shadow identity ---
    agreement_score: float = 0.5        # rolling agreement metric
    lineage: List[str] = field(default_factory=list)
    shadows: List[str] = field(default_factory=list)
    merged_into: Optional[str] = None
    merge_confidence: float = 0.0

    # --- Vector embedding (used by routing) ---
    embedding: Optional[np.ndarray] = None

    # --- Auxiliary metadata ---
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    #  Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Replace enums with their string values
        d["kind"] = self.kind.value
        d["lane"] = self.lane.value
        d["created_at"] = self.created_at.isoformat()
        d["last_used_at"] = self.last_used_at.isoformat()
        # Handle numpy array
        if self.embedding is not None:
            d["embedding"] = self.embedding.tolist()
        else:
            d["embedding"] = None
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Capsule":
        # Restore enums
        data = dict(data)
        data["kind"] = CapsuleKind(data["kind"])
        data["lane"] = LanePosition(data["lane"])
        # Restore datetimes
        for key in ("created_at", "last_used_at"):
            if key in data and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key])
        # Restore numpy embedding
        if data.get("embedding") is not None:
            data["embedding"] = np.array(data["embedding"], dtype=np.float32)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    # ------------------------------------------------------------------
    #  Agreement & merge  (Build Guide §5 methods)
    # ------------------------------------------------------------------

    def evaluate_agreement(self, other: "Capsule") -> float:
        """Compute pairwise agreement score between two capsules.

        Uses cosine similarity of embeddings when available, falling back
        to keyword-set overlap.  Deterministic — no gradient involved.

        Build Guide §6: threshold default is 0.63.
        """
        # Route 1: structural embedding similarity
        if self.embedding is not None and other.embedding is not None:
            norm_a = float(np.linalg.norm(self.embedding))
            norm_b = float(np.linalg.norm(other.embedding))
            if norm_a > 1e-9 and norm_b > 1e-9:
                return float(np.dot(self.embedding, other.embedding) / (norm_a * norm_b))

        # Route 2: keyword overlap
        kw_a = set(self.content.get("keywords", []))
        kw_b = set(other.content.get("keywords", []))
        if kw_a or kw_b:
            union = kw_a | kw_b
            if not union:
                return 0.0
            return len(kw_a & kw_b) / len(union)

        # Route 3: kind-match bonus
        return 0.3 if self.kind == other.kind else 0.1

    def merge(self, other: "Capsule") -> "Capsule":
        """Deterministic merge of two capsules (Build Guide §5).

        Merged confidence is a weighted blend.  Randomness is allowed ONLY
        in the blending of state dicts when both keys collide (controlled
        stochastic blending — Build Guide §12), but the default is
        deterministic key overwrite.
        """
        # ID: deterministic hash of both IDs
        merged_id = f"merged_{hashlib.sha256(f'{self.id}+{other.id}'.encode()).hexdigest()[:12]}"

        # Confidence: usage-weighted average
        total_usage = self.usage_count + other.usage_count
        if total_usage > 0:
            w_self = self.usage_count / total_usage
            w_other = other.usage_count / total_usage
            merged_conf = w_self * self.confidence + w_other * other.confidence
        else:
            merged_conf = (self.confidence + other.confidence) / 2.0

        # Embedding: weighted blend
        if self.embedding is not None and other.embedding is not None:
            emb = (self.embedding * self.usage_count + other.embedding * other.usage_count) / max(total_usage, 1)
        else:
            emb = self.embedding or other.embedding

        # State merge: controlled stochastic on collision (Build Guide §12)
        merged_state = dict(self.state)
        for k, v in other.state.items():
            if k not in merged_state:
                merged_state[k] = v
            elif merged_state[k] != v:
                # Stochastic blend when values differ (allowed randomness)
                merged_state[k] = [merged_state[k], v]

        # Content merge
        merged_content = dict(self.content)
        merged_content.setdefault("keywords", [])
        merged_content["keywords"] = list(set(merged_content["keywords"]) | set(other.content.get("keywords", [])))
        merged_content["original_contents"] = [self.content, other.content]

        # Determine orbit: use the more-salient capsule's orbit
        if self.gravity_score >= other.gravity_score:
            orbit_level = self.orbit_level
            lane = self.lane
        else:
            orbit_level = other.orbit_level
            lane = other.lane

        merged = Capsule(
            id=merged_id,
            kind=self.kind,
            name=f"{self.name}+{other.name}",
            state=merged_state,
            confidence=merged_conf,
            usage_count=self.usage_count + other.usage_count,
            orbit_level=orbit_level,
            lane=lane,
            gravity_score=max(self.gravity_score, other.gravity_score),
            agreement_score=(self.agreement_score + other.agreement_score) / 2.0,
            lineage=sorted(set(self.lineage + other.lineage + [self.id, other.id])),
            shadows=[self.id, other.id],
            merge_confidence=merged_conf,
            embedding=emb,
            content=merged_content,
            metadata={"merged_from": [self.id, other.id], "merged_at": datetime.now().isoformat()},
            created_at=min(self.created_at, other.created_at),
            last_used_at=max(self.last_used_at, other.last_used_at),
        )
        return merged

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def touch(self) -> None:
        """Mark capsule as recently used."""
        self.last_used_at = datetime.now()
        self.usage_count += 1

    def to_dict_safe(self) -> Dict[str, Any]:
        """Serialize with numpy-safe defaults for JSON persistence."""
        d = self.to_dict()
        return d

# ---------------------------------------------------------------------------
# 3.  Routing Engine (Build Guide §6)
# ---------------------------------------------------------------------------

class RoutingEngine:
    """Implements deterministic routing-by-agreement.

    Build Guide §6:
      • Pairwise capsule comparison
      • Threshold-based merging (default 0.63)
      • Fixed routing rounds (default 4)
      • No gradient updates
    """

    def __init__(
        self,
        agreement_threshold: float = 0.63,
        routing_rounds: int = 4,
    ) -> None:
        self.agreement_threshold = agreement_threshold
        self.routing_rounds = routing_rounds
        self.agreement_graph: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.routing_stats: Dict[str, Any] = {
            "cycles_run": 0,
            "merges_performed": 0,
            "total_agreements_computed": 0,
        }

    def route(
        self,
        capsules: Dict[str, Capsule],
        active_ids: Optional[Sequence[str]] = None,
    ) -> Tuple[Dict[str, Capsule], List[Tuple[str, str]]]:
        """Execute one full routing-by-agreement cycle.

        Parameters
        ----------
        capsules : dict
            All capsules in the system.
        active_ids : sequence, optional
            Subset of capsule IDs to consider (if None, all non-merged).

        Returns
        -------
        capsules : dict
            Updated capsule store (potentially with new merged capsules).
        merges : list of (primary_id, secondary_id) pairs
            Capsule IDs that were merged during this cycle.
        """
        if active_ids is None:
            active_ids = [cid for cid, c in capsules.items() if not c.merged_into and not c.hibernating]

        active_list = [capsules[cid] for cid in active_ids if cid in capsules]
        merges: List[Tuple[str, str]] = []

        for _round in range(self.routing_rounds):
            # Build pairwise agreement matrix
            agreements: Dict[Tuple[str, str], float] = {}
            for i, cap_i in enumerate(active_list):
                if cap_i.merged_into:
                    continue
                for cap_j in active_list[i + 1 :]:
                    if cap_j.merged_into:
                        continue
                    if cap_i.kind != cap_j.kind:
                        continue  # only merge capsules of same kind
                    score = cap_i.evaluate_agreement(cap_j)
                    self.routing_stats["total_agreements_computed"] += 1
                    if score >= self.agreement_threshold:
                        agreements[(cap_i.id, cap_j.id)] = score
                        # Update agreement graph
                        self.agreement_graph[cap_i.id][cap_j.id] = (
                            self.agreement_graph[cap_i.id].get(cap_j.id, 0) + score
                        ) / 2
                        self.agreement_graph[cap_j.id][cap_i.id] = (
                            self.agreement_graph[cap_j.id].get(cap_i.id, 0) + score
                        ) / 2

            # Execute merges (greedy by highest agreement)
            sorted_agreements = sorted(agreements.items(), key=lambda x: x[1], reverse=True)
            merged_in_round: set = set()

            for (id_a, id_b), _score in sorted_agreements:
                if id_a in merged_in_round or id_b in merged_in_round:
                    continue
                cap_a = capsules.get(id_a)
                cap_b = capsules.get(id_b)
                if not cap_a or not cap_b or cap_a.merged_into or cap_b.merged_into:
                    continue

                merged = cap_a.merge(cap_b)
                capsules[merged.id] = merged
                cap_a.merged_into = merged.id
                cap_b.merged_into = merged.id
                merges.append((id_a, id_b))
                merged_in_round.add(id_a)
                merged_in_round.add(id_b)
                self.routing_stats["merges_performed"] += 1

        self.routing_stats["cycles_run"] += 1
        return capsules, merges

# ---------------------------------------------------------------------------
# 4.  Gravity & Orbit Manager (Build Guide §7 & §8)
# ---------------------------------------------------------------------------

class GravityOrbitManager:
    """Manages gravity scores and orbit transitions.

    Build Guide §7 – Gravity:
      • Increases after successful agreement
      • Decays via inactivity
      • Controls orbit transitions

    Build Guide §8 – Orbit:
      | Level | Lane      | Meaning          |
      |-------|-----------|------------------|
      | 0     | inner     | Core cognition   |
      | 1     | middle    | Active reasoning |
      | 2     | outer     | Working memory   |
      | 3     | experim.  | Raw ingestion    |
    """

    # Gravity thresholds for orbit promotion / demotion
    PROMOTE_THRESHOLD: float = 0.8
    DEMOTE_THRESHOLD: float = 0.2

    def __init__(
        self,
        gravity_delta_success: float = 0.05,
        gravity_decay_rate: float = 0.01,
        time_constant_days: float = 7.0,
    ) -> None:
        self.gravity_delta_success = gravity_delta_success
        self.gravity_decay_rate = gravity_decay_rate
        self.time_constant_days = time_constant_days

    def update_gravity(
        self,
        capsules: Dict[str, Capsule],
        active_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        """Apply gravity updates and orbit transitions.

        Build Guide §7: g = min(1.0, g + delta)
        Build Guide §8: if gravity > 0.8 → orbit_level -= 1
        """
        if active_ids is None:
            active_ids = list(capsules.keys())

        results: Dict[str, float] = {}
        now = datetime.now()

        for cid in active_ids:
            cap = capsules.get(cid)
            if not cap or cap.merged_into:
                continue

            # Time-decay factor (exponential, half-life = time_constant_days)
            days_since_use = (now - cap.last_used_at).total_seconds() / 86400.0
            decay_factor = math.exp(-days_since_use / max(1.0, self.time_constant_days))

            # Base decay (all capsules experience mild entropy)
            cap.gravity_score *= (1.0 - self.gravity_decay_rate)

            # Apply recency bonus / penalty
            cap.gravity_score *= (0.5 + 0.5 * decay_factor)

            # Clamp to [0, 1]
            cap.gravity_score = max(0.0, min(1.0, cap.gravity_score))

            # --- Orbit transitions (Build Guide §8) ---
            if cap.gravity_score >= self.PROMOTE_THRESHOLD and cap.orbit_level > 0:
                cap.orbit_level = max(0, cap.orbit_level - 1)
            elif cap.gravity_score < self.DEMOTE_THRESHOLD and cap.orbit_level < 3:
                cap.orbit_level = min(3, cap.orbit_level + 1)

            # Sync lane from orbit_level
            cap.lane = ORBIT_LEVEL_TO_LANE.get(cap.orbit_level, LanePosition.OUTER)

            # Update orbit_radius to lane midpoint
            bounds = LANE_BOUNDARIES[cap.lane]
            cap.orbit_radius = (bounds[0] + bounds[1]) / 2.0

            # Hibernation flag
            cap.hibernating = cap.orbit_level == 3 and cap.gravity_score < 0.05

            results[cid] = cap.gravity_score

        return results

    def boost_gravity(
        self,
        capsule: Capsule,
        amount: Optional[float] = None,
    ) -> float:
        """Boost gravity after successful routing / usage (Build Guide §7)."""
        delta = amount if amount is not None else self.gravity_delta_success
        capsule.gravity_score = min(1.0, capsule.gravity_score + delta)
        capsule.touch()
        return capsule.gravity_score

# ---------------------------------------------------------------------------
# 5.  Temporal Graph / FrameNode (Build Guide §10)
# ---------------------------------------------------------------------------

@dataclass
class FrameNode:
    """One frame of system cognition — Build Guide §10.

    Every loop iteration produces a frame:
      • timestamp
      • active_capsules
      • orbit_distribution
      • edges (agreement relations)
    """
    frame_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = field(default_factory=datetime.now)
    active_capsule_ids: List[str] = field(default_factory=list)
    active_count: int = 0
    orbit_distribution: Dict[int, int] = field(default_factory=dict)
    max_gravity: float = 0.0
    merges_this_cycle: int = 0
    edges: List[Tuple[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameNode":
        data = dict(data)
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class TimelineStore:
    """Append-only temporal graph store (Build Guide §10)."""

    def __init__(self, store_path: Path = DEFAULT_TEMPORAL_STORE_PATH) -> None:
        self.store_path = store_path
        self.frames: List[FrameNode] = []

    def append(self, frame: FrameNode) -> None:
        self.frames.append(frame)
        # Append one JSON line for persistence
        with open(self.store_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(frame.to_dict(), default=str) + "\n")

    def load(self, max_frames: int = 2000) -> None:
        if not self.store_path.exists():
            return
        loaded: List[FrameNode] = []
        with open(self.store_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    loaded.append(FrameNode.from_dict(json.loads(line)))
                except Exception:
                    pass
        self.frames = loaded[-max_frames:]

    def replay(
        self,
        start_index: int = 0,
        end_index: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return frames for replay / debugging."""
        end = end_index if end_index is not None else len(self.frames)
        return [f.to_dict() for f in self.frames[start_index:end]]

    def stats(self) -> Dict[str, Any]:
        if not self.frames:
            return {"total_frames": 0}
        orbit_counts: Dict[int, int] = defaultdict(int)
        for f in self.frames:
            for level, count in f.orbit_distribution.items():
                orbit_counts[level] += count
        return {
            "total_frames": len(self.frames),
            "first_frame": self.frames[0].timestamp.isoformat(),
            "last_frame": self.frames[-1].timestamp.isoformat(),
            "avg_active_capsules": sum(f.active_count for f in self.frames) / len(self.frames),
            "total_merges": sum(f.merges_this_cycle for f in self.frames),
            "orbit_totals": dict(orbit_counts),
        }

# ---------------------------------------------------------------------------
# 6.  Adaptive Resource Manager (Build Guide §9)
# ---------------------------------------------------------------------------

class AdaptiveResourceManager:
    """Hardware-aware budget manager — Build Guide §9.

    Responsibilities:
      • Detect CPU / RAM
      • Set max active capsule budget (B)
      • Dynamically reduce workload under pressure
    """

    def __init__(self) -> None:
        self.cpu_count: int = self._detect_cpus()
        self.total_ram_mb: float = self._detect_ram_mb()
        self.max_budget: int = self._compute_max_budget()
        self.current_budget: int = self.max_budget
        self.pressure_level: float = 0.0  # 0.0 – 1.0
        self.stats: Dict[str, Any] = {
            "cpu_count": self.cpu_count,
            "total_ram_mb": self.total_ram_mb,
            "max_budget": self.max_budget,
            "budget_adjustments": 0,
        }

    @staticmethod
    def _detect_cpus() -> int:
        try:
            return os.cpu_count() or 4
        except Exception:
            return 4

    @staticmethod
    def _detect_ram_mb() -> float:
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 * 1024)
        except ImportError:
            return 4096.0  # assume 4 GB

    def _compute_max_budget(self) -> int:
        """Heuristic: 20 capsules per CPU core, capped by RAM."""
        cpu_budget = self.cpu_count * 20
        ram_budget = max(10, int(self.total_ram_mb / 64))  # 64 MB per capsule
        return min(cpu_budget, ram_budget)

    def assess_pressure(self) -> float:
        """Return current system pressure (0.0 = idle, 1.0 = max)."""
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.1) / 100.0
            ram = psutil.virtual_memory().percent / 100.0
            return max(cpu, ram)
        except ImportError:
            return 0.3  # neutral default

    def adjust_budget(self, active_count: int) -> int:
        """Dynamically adjust active budget based on pressure.

        Build Guide §9:
          • High load → reduce active set size
          • Low load → expand active cognition window
        """
        self.pressure_level = self.assess_pressure()
        if self.pressure_level > 0.8:
            self.current_budget = max(5, int(self.current_budget * 0.7))
        elif self.pressure_level > 0.6:
            self.current_budget = max(5, int(self.current_budget * 0.85))
        elif self.pressure_level < 0.3 and self.current_budget < self.max_budget:
            self.current_budget = min(self.max_budget, int(self.current_budget * 1.1))
        self.current_budget = min(self.current_budget, self.max_budget)
        self.stats["budget_adjustments"] += 1
        return self.current_budget

    def select_active_set(
        self,
        capsules: Dict[str, Capsule],
        max_capsules: Optional[int] = None,
    ) -> List[str]:
        """Select the active working set sorted by gravity_score.

        Build Guide §9: Dynamic active set size based on budget.
        """
        budget = max_capsules if max_capsules is not None else self.current_budget
        candidates = [
            (cid, cap.gravity_score)
            for cid, cap in capsules.items()
            if not cap.merged_into and not cap.hibernating
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in candidates[:budget]]

# ---------------------------------------------------------------------------
# 7.  Persistent Store (Build Guide §10 – persistence layer)
# ---------------------------------------------------------------------------

class PersistentStore:
    """Capsule persistence layer — saves / loads full capsule graph."""

    def __init__(self, store_path: Path = DEFAULT_KNOWLEDGE_BASE_PATH) -> None:
        self.store_path = store_path

    def save(self, capsules: Dict[str, Capsule], extra: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {
            "capsules": {cid: c.to_dict_safe() for cid, c in capsules.items()},
            "saved_at": datetime.now().isoformat(),
            "capsule_count": len(capsules),
        }
        if extra:
            payload.update(extra)
        with open(self.store_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)

    def load(self) -> Dict[str, Capsule]:
        if not self.store_path.exists():
            return {}
        with open(self.store_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        capsules: Dict[str, Capsule] = {}
        for cid, cdata in payload.get("capsules", {}).items():
            capsules[cid] = Capsule.from_dict(cdata)
        return capsules

# ---------------------------------------------------------------------------
# 8.  Embedding helpers (deterministic, no neural nets)
# ---------------------------------------------------------------------------

def deterministic_embed(text: str, dim: int = 128) -> np.ndarray:
    """Create a deterministic embedding vector from text using SHA-256.

    Build Guide §12: No gradient descent allowed.
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_\-]{2,}", text.lower())
    if not tokens:
        return vec
    for i, token in enumerate(tokens):
        h = hashlib.sha256(token.encode()).digest()
        bucket = int.from_bytes(h[:4], "big") % dim
        weight = 1.0 + (0.35 if i >= len(tokens) * 0.8 else 0.0)  # tail-boost
        vec[bucket] += weight
    norm = float(np.linalg.norm(vec))
    if norm > 1e-9:
        vec /= norm
    return vec

# ---------------------------------------------------------------------------
# 9.  Seed Capsule Initializer
# ---------------------------------------------------------------------------

def initialize_seed_capsules() -> Dict[str, Capsule]:
    """Create the initial seed capsule graph (Build Guide §11 step 2)."""
    seeds: Dict[str, Capsule] = {}

    nucleus_capsules = [
        {
            "id": "nucleus_core_identity",
            "name": "Core Identity",
            "state": {"traits": ["deterministic", "structured", "analytical"]},
            "content": {
                "keywords": ["roca", "capsule", "identity", "kernel"],
                "description": "Nucleus — anchors core cognitive identity.",
            },
        },
        {
            "id": "nucleus_routing_expert",
            "name": "Routing Expert",
            "state": {"specialties": ["agreement", "routing", "consensus"]},
            "content": {
                "keywords": ["routing", "agreement", "merge", "threshold"],
                "description": "Nucleus — specialises in routing-by-agreement logic.",
            },
        },
        {
            "id": "nucleus_orbit_keeper",
            "name": "Orbit Keeper",
            "state": {"domain": "orbital_mechanics"},
            "content": {
                "keywords": ["orbit", "gravity", "lane", "salience", "decay"],
                "description": "Nucleus — maintains orbital hierarchy and gravity scores.",
            },
        },
    ]

    for cap_data in nucleus_capsules:
        emb = deterministic_embed(cap_data["name"] + " " + " ".join(cap_data["content"]["keywords"]))
        seeds[cap_data["id"]] = Capsule(
            id=cap_data["id"],
            kind=CapsuleKind.NUCLEUS,
            name=cap_data["name"],
            state=cap_data["state"],
            content=cap_data["content"],
            embedding=emb,
            orbit_level=0,
            lane=LanePosition.INNER,
            orbit_radius=0.1,
            gravity_score=1.0,
            agreement_score=1.0,
            confidence=1.0,
        )

    # Add a few skill capsules as working examples
    skill_seeds = [
        ("skill_python", "Python Runtime", ["python", "exec", "interpreter"]),
        ("skill_json", "JSON Handler", ["json", "serialize", "parse"]),
        ("skill_file_io", "File I/O", ["file", "read", "write", "persistence"]),
    ]
    for cid, name, keywords in skill_seeds:
        emb = deterministic_embed(name + " " + " ".join(keywords))
        seeds[cid] = Capsule(
            id=cid,
            kind=CapsuleKind.SKILL,
            name=name,
            content={"keywords": keywords, "description": f"Skill capsule: {name}"},
            embedding=emb,
            orbit_level=2,
            lane=LanePosition.OUTER,
            gravity_score=0.3,
        )

    return seeds

# ---------------------------------------------------------------------------
# 10. ROCA Kernel — main loop (Build Guide §4 & §11)
# ---------------------------------------------------------------------------

class ROCAKernel:
    """Core kernel implementing the deterministic loop engine.

    Build Guide §4 — Main Loop:
        while system_running:
            ingest_input()
            active_set = select_capsules_by_budget()
            active_set = route_by_agreement(active_set)
            update_gravity(active_set)
            update_orbits(active_set)
            write_temporal_frame()
            persist_state()
            hibernate_low_gravity_capsules()
    """

    def __init__(
        self,
        load_existing: bool = True,
        store_path: Path = DEFAULT_KNOWLEDGE_BASE_PATH,
        timeline_path: Path = DEFAULT_TEMPORAL_STORE_PATH,
    ) -> None:
        # Subsystems
        self.store = PersistentStore(store_path)
        self.timeline = TimelineStore(timeline_path)
        self.router = RoutingEngine(agreement_threshold=0.63, routing_rounds=4)
        self.gravity_orbit = GravityOrbitManager()
        self.resource_mgr = AdaptiveResourceManager()

        # Capsule graph
        if load_existing:
            self.capsules = self.store.load()
        else:
            self.capsules: Dict[str, Capsule] = {}

        # Seed if empty
        if not self.capsules:
            self.capsules = initialize_seed_capsules()
            self.store.save(self.capsules)

        # Load timeline
        self.timeline.load()

        # State
        self.cycle_count: int = 0
        self.running: bool = False
        self._input_queue: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def ingest_input(self, text: str, source: str = "user") -> None:
        """Ingest external text and create temporary ingestion capsules."""
        chunks = self._chunk_text(text, max_chars=1200, overlap=200)
        for i, chunk in enumerate(chunks):
            cid = f"ingest_{self.cycle_count}_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
            emb = deterministic_embed(chunk)
            cap = Capsule(
                id=cid,
                kind=CapsuleKind.TOPIC,
                name=f"Ingest::{source}::{i + 1}",
                state={"raw_text": chunk[:500]},
                content={"keywords": self._extract_keywords(chunk)[:30], "text": chunk},
                embedding=emb,
                orbit_level=3,
                lane=LanePosition.EXPERIMENTAL,
                gravity_score=0.15,
            )
            self.capsules[cid] = cap
        self._input_queue.append({"text": text, "source": source, "chunk_count": len(chunks)})

    def run_cycle(self) -> Dict[str, Any]:
        """Execute one full cognition cycle (Build Guide §4 main loop).

        Returns a summary dict suitable for logging / CLI output.
        """
        self.cycle_count += 1
        start_time = time.time()

        # 1. Select active set by budget
        active_ids = self.resource_mgr.select_active_set(self.capsules)
        self.resource_mgr.adjust_budget(len(active_ids))

        # 2. Route by agreement
        self.capsules, merges = self.router.route(self.capsules, active_ids)

        # 3. Update gravity & orbits
        self.gravity_orbit.update_gravity(self.capsules, active_ids)

        # 4. Boost gravity for routed capsules
        for cid in active_ids:
            cap = self.capsules.get(cid)
            if cap and not cap.merged_into:
                self.gravity_orbit.boost_gravity(cap)

        # 5. Build temporal frame
        orbit_dist: Dict[int, int] = defaultdict(int)
        max_g = 0.0
        for cid in active_ids:
            cap = self.capsules.get(cid)
            if cap and not cap.merged_into:
                orbit_dist[cap.orbit_level] += 1
                max_g = max(max_g, cap.gravity_score)

        frame = FrameNode(
            timestamp=datetime.now(),
            active_capsule_ids=active_ids,
            active_count=len(active_ids),
            orbit_distribution=dict(orbit_dist),
            max_gravity=round(max_g, 4),
            merges_this_cycle=len(merges),
            edges=merges,
            metadata={"cycle": self.cycle_count, "pressure": self.resource_mgr.pressure_level},
        )
        self.timeline.append(frame)

        # 6. Persist
        self.store.save(
            self.capsules,
            extra={
                "cycle_count": self.cycle_count,
                "routing_stats": self.router.routing_stats,
                "timeline_stats": self.timeline.stats(),
            },
        )

        elapsed = time.time() - start_time
        return {
            "cycle": self.cycle_count,
            "active_capsules": len(active_ids),
            "merges_this_cycle": len(merges),
            "orbit_distribution": dict(orbit_dist),
            "max_gravity": round(max_g, 4),
            "total_capsules": len(self.capsules),
            "hibernating": sum(1 for c in self.capsules.values() if c.hibernating),
            "elapsed_seconds": round(elapsed, 4),
            "pressure": round(self.resource_mgr.pressure_level, 2),
        }

    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search capsules by semantic relevance (deterministic embedding)."""
        query_emb = deterministic_embed(text)
        scored: List[Tuple[float, Capsule]] = []
        for cap in self.capsules.values():
            if cap.merged_into or cap.hibernating:
                continue
            if cap.embedding is None:
                score = 0.1
            else:
                norm_q = float(np.linalg.norm(query_emb))
                norm_c = float(np.linalg.norm(cap.embedding))
                score = float(np.dot(query_emb, cap.embedding) / max(1e-9, norm_q * norm_c))
            scored.append((score, cap))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": c.id,
                "name": c.name,
                "kind": c.kind.value,
                "orbit_level": c.orbit_level,
                "gravity": round(c.gravity_score, 4),
                "score": round(score, 4),
                "content_summary": str(c.content.get("description", c.content.get("text", ""))[:120]),
            }
            for score, c in scored[:top_k]
        ]

    def status(self) -> Dict[str, Any]:
        """Full system status report."""
        orbit_counts: Dict[int, int] = defaultdict(int)
        kind_counts: Dict[str, int] = defaultdict(int)
        for cap in self.capsules.values():
            if not cap.merged_into:
                orbit_counts[cap.orbit_level] += 1
                kind_counts[cap.kind.value] += 1
        return {
            "total_capsules": len(self.capsules),
            "active_capsules": sum(1 for c in self.capsules.values() if not c.merged_into and not c.hibernating),
            "hibernating": sum(1 for c in self.capsules.values() if c.hibernating),
            "orbit_distribution": dict(orbit_counts),
            "kind_distribution": dict(kind_counts),
            "cycle_count": self.cycle_count,
            "routing_stats": dict(self.router.routing_stats),
            "timeline_stats": self.timeline.stats(),
            "resource_stats": dict(self.resource_mgr.stats),
            "pressure": round(self.resource_mgr.pressure_level, 2),
        }

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(0, end - overlap)
        return chunks

    @staticmethod
    def _extract_keywords(text: str, min_len: int = 3, max_keywords: int = 40) -> List[str]:
        tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_\-]{2,}", text.lower())
        freq: Dict[str, int] = defaultdict(int)
        for t in tokens:
            if len(t) >= min_len and t not in {"the", "and", "for", "with", "from", "this", "that"}:
                freq[t] += 1
        return sorted(freq, key=lambda k: freq[k], reverse=True)[:max_keywords]

# ---------------------------------------------------------------------------
# 11.  Bootstrap entry point (Build Guide §11 step 7)
# ---------------------------------------------------------------------------

def bootstrap_kernel(
    load_existing: bool = True,
    store_path: Optional[Path] = None,
    timeline_path: Optional[Path] = None,
) -> ROCAKernel:
    """Initialise and return a ready-to-run ROCA kernel.

    Build Guide §11 Boot Sequence:
      1. Kernel initialization
      2. Seed capsule injection [handled by ROCAKernel.__init__]
      3. Resource manager activation
      4. Main loop execution  [caller's responsibility]
      5. Timeline logging     [handled per cycle]
    """
    sp = store_path or DEFAULT_KNOWLEDGE_BASE_PATH
    tp = timeline_path or DEFAULT_TEMPORAL_STORE_PATH
    kernel = ROCAKernel(load_existing=load_existing, store_path=sp, timeline_path=tp)
    # Adjust budget for current hardware
    kernel.resource_mgr.adjust_budget(len(kernel.capsules))
    return kernel

# ---------------------------------------------------------------------------
# 12.  CLI demo / smoke test
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point — smoke-tests the kernel with a few cycles."""
    print("=" * 70)
    print("ROCA Unified Kernel — Smoke Test")
    print("=" * 70)

    kernel = bootstrap_kernel(load_existing=False)

    print(f"\nCapsules loaded: {len(kernel.capsules)}")
    status = kernel.status()
    print(f"Orbit distribution: {status['orbit_distribution']}")
    print(f"Kind distribution:  {status['kind_distribution']}")
    print(f"Max budget:         {kernel.resource_mgr.max_budget}")
    print(f"Current budget:     {kernel.resource_mgr.current_budget}")
    print(f"Pressure:           {kernel.resource_mgr.pressure_level:.2f}")

    # Ingest a sample input
    sample_text = (
        "ROCA uses routing-by-agreement to merge similar capsules. "
        "Capsules orbit at different levels based on their gravity scores. "
        "The system is deterministic and does not use gradient descent."
    )
    kernel.ingest_input(sample_text, source="demo")

    print("\nRunning 3 cognition cycles...\n")
    for _ in range(3):
        result = kernel.run_cycle()
        print(
            f"  Cycle {result['cycle']:>3} | "
            f"Active: {result['active_capsules']:>3} | "
            f"Merges: {result['merges_this_cycle']:>2} | "
            f"Max gravity: {result['max_gravity']:.3f} | "
            f"Orbits: {result['orbit_distribution']} | "
            f"Pressure: {result['pressure']:.2f}"
        )

    # Query
    query_text = "deterministic capsule routing"
    print(f"\nQuery: '{query_text}'")
    results = kernel.query(query_text, top_k=5)
    for r in results:
        print(f"  [{r['kind']:>9}] orbit={r['orbit_level']} grav={r['gravity']:.3f}  {r['name']}")

    # Final status
    final = kernel.status()
    print(f"\nFinal state:")
    print(f"  Total capsules:  {final['total_capsules']}")
    print(f"  Active:          {final['active_capsules']}")
    print(f"  Hibernating:     {final['hibernating']}")
    print(f"  Cycles run:      {final['cycle_count']}")
    print(f"  Merges total:    {final['routing_stats']['merges_performed']}")
    print(f"  Timeline frames: {final['timeline_stats']['total_frames']}")
    print(f"  Pressure:        {final['pressure']:.2f}")

    print("\nDone. Knowledge base saved to:", DEFAULT_KNOWLEDGE_BASE_PATH)
    print("Timeline saved to:", DEFAULT_TEMPORAL_STORE_PATH)


if __name__ == "__main__":
    main()