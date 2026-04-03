from typing import Optional
from core.capsule import Capsule, CapsuleKind


class CapsuleStore:
    def __init__(self):
        self._capsules: dict[str, Capsule] = {}
        # agreement scores keyed by frozenset-like sorted tuple
        self._edges: dict[tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, capsule: Capsule) -> None:
        self._capsules[capsule.id] = capsule

    def get(self, cid: str) -> Optional[Capsule]:
        return self._capsules.get(cid)

    def all(self) -> list[Capsule]:
        return list(self._capsules.values())

    def remove(self, cid: str) -> None:
        self._capsules.pop(cid, None)
        keys_to_remove = [k for k in self._edges if cid in k]
        for k in keys_to_remove:
            del self._edges[k]

    # ------------------------------------------------------------------
    # Agreement scores
    # ------------------------------------------------------------------

    @staticmethod
    def _edge_key(id_a: str, id_b: str) -> tuple[str, str]:
        return (min(id_a, id_b), max(id_a, id_b))

    def record_agreement(self, id_a: str, id_b: str, score: float = 1.0) -> None:
        key = self._edge_key(id_a, id_b)
        self._edges[key] = self._edges.get(key, 0.0) + score

    def agreement_score(self, id_a: str, id_b: str) -> float:
        return self._edges.get(self._edge_key(id_a, id_b), 0.0)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        capsules_list = []
        for c in self._capsules.values():
            entry = {
                "id": c.id,
                "kind": c.kind.value,
                "name": c.name,
                "created_at": c.created_at,
                "use_count": c.use_count,
                "last_used_at": c.last_used_at,
                "orbit_score": c.orbit_score,
                "asset_path": c.asset_path,
                "asset_hash": c.asset_hash,
            }
            capsules_list.append(entry)

        edges_serialized = {
            f"{k[0]}|{k[1]}": v for k, v in self._edges.items()
        }
        return {"capsules": capsules_list, "edges": edges_serialized}

    def from_dict(self, data: dict) -> None:
        self._capsules.clear()
        self._edges.clear()

        for entry in data.get("capsules", []):
            try:
                kind = CapsuleKind(entry["kind"])
            except ValueError:
                kind = CapsuleKind.UNASSIGNED
            cap = Capsule(
                id=entry["id"],
                kind=kind,
                name=entry["name"],
                created_at=entry.get("created_at", 0.0),
                use_count=entry.get("use_count", 0),
                last_used_at=entry.get("last_used_at"),
                orbit_score=entry.get("orbit_score", 0.0),
                asset_path=entry.get("asset_path"),
                asset_hash=entry.get("asset_hash"),
            )
            self._capsules[cap.id] = cap

        for raw_key, score in data.get("edges", {}).items():
            parts = raw_key.split("|", 1)
            if len(parts) == 2:
                key = self._edge_key(parts[0], parts[1])
                self._edges[key] = float(score)
