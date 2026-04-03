import uuid
import hashlib
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class CapsuleKind(Enum):
    POSE = "pose"
    TRANSITION = "transition"
    TIMING = "timing"
    CYCLE = "cycle"
    CHARACTER = "character"
    MEMORY = "memory"
    UNASSIGNED = "unassigned"


@dataclass
class Capsule:
    id: str               # UUID5 string
    kind: CapsuleKind
    name: str
    created_at: float = field(default_factory=time.time)
    use_count: int = 0
    last_used_at: Optional[float] = None
    orbit_score: float = 0.0
    asset_path: Optional[str] = None   # path to PNG on disk
    asset_hash: Optional[str] = None   # SHA256 of image bytes
    image_data: Optional[bytes] = None # in-memory PNG bytes


def make_capsule_id(kind: CapsuleKind, name: str) -> str:
    """Deterministic UUID5 from (kind.value, name)."""
    namespace = uuid.UUID("12345678-1234-5678-1234-567812345678")
    return str(uuid.uuid5(namespace, f"{kind.value}:{name}"))


def image_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]
