from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from PyQt6.QtCore import QPointF, QTimer, Qt, pyqtSignal
    from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QTextCursor
    from PyQt6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    _HAS_QT = True
except Exception:
    _HAS_QT = False
    QApplication = None
    QBrush = None
    QColor = None
    QHBoxLayout = None
    QLabel = None
    QLineEdit = None
    QMainWindow = object
    QPainter = None
    QPen = None
    QPointF = None
    QPushButton = None
    QTabWidget = None
    QTextEdit = None
    QTimer = None
    QVBoxLayout = None
    QWidget = None
    QFont = None
    QTextCursor = None
    Qt = None
    pyqtSignal = None


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
STATE_DIR = ROOT / "brain_state"
MEMORY_PATH = STATE_DIR / "doku_memory.json"
JOURNAL_PATH = STATE_DIR / "doku_dialogue.jsonl"

TEXT_EXTENSIONS = {".txt", ".md", ".json", ".csv"}
WORKSPACE_CODE_EXTENSIONS = {".py", ".md", ".yml", ".yaml", ".json", ".toml", ".txt"}
WORKSPACE_SKIP_DIRS = {
    ".git", ".venv", ".vscode", "__pycache__", "data", "brain_state",
    "logs", "models", "checkpoints", "archive", "resources",
}
MAX_FILE_BYTES = 1_000_000
MAX_FILES = 250
MAX_WORKSPACE_FILES = 500
TOP_K = 5


def _now() -> float:
    return time.time()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]{2,}", text.lower())


def _stable_capsule_id(path: Path, text: str) -> str:
    digest = hashlib.sha1()
    digest.update(str(path.relative_to(ROOT)).encode("utf-8", errors="replace"))
    digest.update(b"|")
    digest.update(text[:4000].encode("utf-8", errors="replace"))
    return digest.hexdigest()


@dataclass
class DokuCapsule:
    id: str
    name: str
    source_path: str
    text_excerpt: str
    token_counts: Dict[str, int]
    source_size: int
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    last_used_at: float = 0.0
    use_count: int = 0
    salience: float = 0.0

    def touch(self) -> None:
        self.use_count += 1
        self.last_used_at = _now()
        self.salience = self.compute_salience()

    def compute_salience(self, tau_seconds: float = 60.0 * 60.0 * 24.0 * 14.0) -> float:
        recency = 0.0
        if self.last_used_at > 0:
            age = max(0.0, _now() - self.last_used_at)
            recency = math.exp(-age / tau_seconds)
        usage = math.log1p(self.use_count)
        return round((0.65 * usage) + (0.35 * recency), 4)

    @property
    def token_set(self) -> set[str]:
        return set(self.token_counts)


class DokuMemory:
    def __init__(self) -> None:
        self.capsules: Dict[str, DokuCapsule] = {}

    def upsert(self, capsule: DokuCapsule) -> None:
        existing = self.capsules.get(capsule.id)
        if existing is None:
            capsule.salience = capsule.compute_salience()
            self.capsules[capsule.id] = capsule
            return

        capsule.use_count = existing.use_count
        capsule.last_used_at = existing.last_used_at
        capsule.created_at = existing.created_at
        capsule.salience = capsule.compute_salience()
        self.capsules[capsule.id] = capsule

    def save(self, path: Path = MEMORY_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": _now(),
            "capsules": [asdict(capsule) for capsule in self.capsules.values()],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: Path = MEMORY_PATH) -> None:
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for item in payload.get("capsules", []):
            try:
                capsule = DokuCapsule(**item)
            except TypeError:
                continue
            capsule.salience = capsule.compute_salience()
            self.capsules[capsule.id] = capsule

    def recall(self, query: str, top_k: int = TOP_K) -> List[tuple[float, DokuCapsule]]:
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return []

        ranked: List[tuple[float, DokuCapsule]] = []
        for capsule in self.capsules.values():
            overlap = query_tokens & capsule.token_set
            if not overlap:
                continue

            overlap_score = len(overlap) / max(1, len(query_tokens))
            density_score = len(overlap) / max(1, len(capsule.token_set))
            score = (0.70 * overlap_score) + (0.20 * density_score) + (0.10 * capsule.salience)
            ranked.append((score, capsule))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[:top_k]

    def stats(self) -> Dict[str, int]:
        return {
            "capsules": len(self.capsules),
            "used_capsules": sum(1 for capsule in self.capsules.values() if capsule.use_count > 0),
        }


class DokuEntity:
    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = data_dir
        self.memory = DokuMemory()
        self.memory.load()
        self._coding_assistant: Any | None = None
        self._coding_load_error: str | None = None
        self._workspace_files_cache: List[Path] | None = None

    def bootstrap(self, rescan: bool = False) -> str:
        if rescan or not self.memory.capsules:
            ingested = self.ingest_data_directory()
            self.memory.save()
            return f"Memory bootstrapped from data/: {ingested} capsule(s) available."
        stats = self.memory.stats()
        return f"Loaded existing memory with {stats['capsules']} capsule(s)."

    def ingest_data_directory(self) -> int:
        if not self.data_dir.exists():
            return 0

        count = 0
        for path in self._iter_candidate_files(self.data_dir):
            capsule = self._capsule_from_file(path)
            if capsule is None:
                continue
            self.memory.upsert(capsule)
            count += 1
        return count

    def _iter_candidate_files(self, root: Path) -> Iterable[Path]:
        seen = 0
        for path in sorted(root.rglob("*")):
            if seen >= MAX_FILES:
                break
            if not path.is_file():
                continue
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size <= 0 or size > MAX_FILE_BYTES:
                continue
            seen += 1
            yield path

    def _capsule_from_file(self, path: Path) -> DokuCapsule | None:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

        cleaned = " ".join(text.split())
        if not cleaned:
            return None

        tokens = _tokenize(cleaned)
        if not tokens:
            return None

        token_counts: Dict[str, int] = {}
        for token in tokens[:5000]:
            token_counts[token] = token_counts.get(token, 0) + 1

        relative = path.relative_to(ROOT)
        capsule_id = _stable_capsule_id(path, cleaned)
        excerpt = textwrap.shorten(cleaned, width=360, placeholder=" ...")
        try:
            size = path.stat().st_size
        except OSError:
            size = len(text.encode("utf-8", errors="replace"))

        return DokuCapsule(
            id=capsule_id,
            name=path.name,
            source_path=str(relative),
            text_excerpt=excerpt,
            token_counts=token_counts,
            source_size=size,
        )

    def answer(self, user_text: str) -> str:
        lower = user_text.strip().lower()
        if not lower:
            return "I did not receive a message."

        if lower in {"/help", "help"}:
            return self._help_text()
        if lower in {"/status", "status"} or self._looks_like_status_request(lower):
            return self._status_text()
        if lower in {"/rescan", "rescan"} or self._looks_like_rescan_request(lower):
            ingested = self.ingest_data_directory()
            self.memory.save()
            return f"Rescan complete. Memory now holds {len(self.memory.capsules)} capsule(s) after checking {ingested} file(s)."
        if lower in {"/sources", "sources"} or self._looks_like_sources_request(lower):
            return self._sources_text()
        if lower in {"/workspace", "workspace status", "/workspace status"}:
            return self._workspace_status_text()

        workspace_paths = self._extract_workspace_paths(user_text)
        if workspace_paths and self._looks_like_workspace_request(lower):
            return self._handle_workspace_request(user_text, lower, workspace_paths)

        if lower in {"/coding", "coding status", "/coding status"} or self._looks_like_coding_request(lower):
            return self._handle_coding_request(user_text, lower)

        matches = self.memory.recall(user_text, top_k=TOP_K)
        if not matches:
            return (
                "I could not ground that request in my current memory. "
                "Try asking about material that exists under data/, ask about a workspace file, or run /rescan."
            )

        for _, capsule in matches:
            capsule.touch()
        self.memory.save()
        return self._compose_grounded_response(user_text, matches)

    def _iter_workspace_files(self) -> Iterable[Path]:
        seen = 0
        for current_root, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [name for name in dirnames if name not in WORKSPACE_SKIP_DIRS]
            current_path = Path(current_root)
            for filename in sorted(filenames):
                if seen >= MAX_WORKSPACE_FILES:
                    return
                candidate = current_path / filename
                if candidate.suffix.lower() not in WORKSPACE_CODE_EXTENSIONS:
                    continue
                seen += 1
                yield candidate

    def _workspace_files(self) -> List[Path]:
        if self._workspace_files_cache is None:
            self._workspace_files_cache = list(self._iter_workspace_files())
        return self._workspace_files_cache

    def _extract_workspace_paths(self, user_text: str) -> List[Path]:
        lower = user_text.lower().replace("\\", "/")
        matches: List[Path] = []
        seen: set[str] = set()
        for candidate in self._workspace_files():
            rel = str(candidate.relative_to(ROOT)).replace("\\", "/")
            if rel.lower() in lower or candidate.name.lower() in lower:
                if rel not in seen:
                    matches.append(candidate)
                    seen.add(rel)
        return matches[:5]

    def _looks_like_workspace_request(self, lower: str) -> bool:
        return (
            self._looks_like_workspace_review_request(lower)
            or self._looks_like_workspace_edit_request(lower)
            or self._looks_like_workspace_explain_request(lower)
        )

    def _looks_like_workspace_review_request(self, lower: str) -> bool:
        return any(phrase in lower for phrase in (
            "review",
            "audit",
            "find bugs",
            "look for issues",
            "code review",
        ))

    def _looks_like_workspace_edit_request(self, lower: str) -> bool:
        return any(phrase in lower for phrase in (
            "edit",
            "change",
            "modify",
            "update",
            "refactor",
            "fix",
            "patch",
        ))

    def _looks_like_workspace_explain_request(self, lower: str) -> bool:
        return any(phrase in lower for phrase in (
            "what does",
            "explain",
            "summarize",
            "show structure",
            "how does",
            "read file",
        ))

    def _handle_workspace_request(self, user_text: str, lower: str, paths: List[Path]) -> str:
        if self._looks_like_workspace_review_request(lower):
            return self._review_workspace_files(paths)
        if self._looks_like_workspace_edit_request(lower):
            return self._plan_workspace_edit(user_text, paths)
        return self._explain_workspace_files(paths)

    def _read_workspace_file(self, path: Path) -> str:
        return path.read_text(encoding="utf-8-sig", errors="replace")

    def _workspace_status_text(self) -> str:
        files = self._workspace_files()
        preview = [str(path.relative_to(ROOT)) for path in files[:12]]
        lines = [
            "Workspace lane status",
            "---------------------",
            f"Indexed files: {len(files)}",
            "Examples:",
        ]
        lines.extend(f"- {item}" for item in preview)
        if len(files) > len(preview):
            lines.append(f"- ... and {len(files) - len(preview)} more")
        return "\n".join(lines)

    def _review_workspace_files(self, paths: List[Path]) -> str:
        findings: List[tuple[int, str]] = []
        summaries: List[str] = []
        for path in paths:
            rel = str(path.relative_to(ROOT))
            try:
                text = self._read_workspace_file(path)
            except OSError as exc:
                findings.append((0, f"[high] {rel}: could not read file ({exc})"))
                continue

            lines = text.splitlines()
            summaries.append(f"{rel}: {len(lines)} lines")

            for idx, line in enumerate(lines, start=1):
                stripped = line.strip()
                if stripped.startswith("except:"):
                    findings.append((0, f"[high] {rel}:{idx} bare except hides specific failure modes"))
                if "eval(" in stripped or "exec(" in stripped:
                    findings.append((0, f"[high] {rel}:{idx} dynamic code execution present"))
                if "TODO" in stripped or "FIXME" in stripped:
                    findings.append((2, f"[low] {rel}:{idx} contains TODO/FIXME marker"))
                if stripped.startswith("from ") and stripped.endswith(" import *"):
                    findings.append((1, f"[medium] {rel}:{idx} wildcard import reduces readability and safety"))

            if path.suffix.lower() == ".py":
                try:
                    tree = ast.parse(text)
                except SyntaxError as exc:
                    findings.append((0, f"[high] {rel}:{exc.lineno or 1} syntax error prevents reliable analysis"))
                    continue

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        for default in node.args.defaults:
                            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                                findings.append((1, f"[medium] {rel}:{node.lineno} mutable default argument in {node.name}()"))
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                        findings.append((2, f"[low] {rel}:{node.lineno} print() left in code path"))

        findings.sort(key=lambda item: item[0])
        if not findings:
            body = ["Findings:", "- No obvious static findings from the current heuristics."]
        else:
            body = ["Findings:"]
            body.extend(f"- {message}" for _, message in findings[:12])

        body.append("")
        body.append("Files reviewed:")
        body.extend(f"- {summary}" for summary in summaries)
        body.append("")
        body.append("Open questions:")
        body.append("- Static review is heuristic-only here; runtime behavior and tests still matter.")
        return "\n".join(body)

    def _explain_workspace_files(self, paths: List[Path]) -> str:
        lines = ["Workspace file summary:"]
        for path in paths:
            rel = str(path.relative_to(ROOT))
            try:
                text = self._read_workspace_file(path)
            except OSError as exc:
                lines.append(f"- {rel}: unreadable ({exc})")
                continue

            file_lines = text.splitlines()
            excerpt = textwrap.shorten(" ".join(text.split()), width=220, placeholder=" ...")
            if path.suffix.lower() == ".py":
                try:
                    tree = ast.parse(text)
                    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                except SyntaxError:
                    functions = []
                    classes = []
                lines.append(
                    f"- {rel}: {len(file_lines)} lines, {len(classes)} classes, {len(functions)} functions"
                )
            else:
                lines.append(f"- {rel}: {len(file_lines)} lines")
            if excerpt:
                lines.append(f"  {excerpt}")
        return "\n".join(lines)

    def _plan_workspace_edit(self, user_text: str, paths: List[Path]) -> str:
        rel_paths = [str(path.relative_to(ROOT)) for path in paths]
        lines = [
            "Workspace edit request detected.",
            "",
            "Target files:",
        ]
        lines.extend(f"- {rel}" for rel in rel_paths)
        lines.extend([
            "",
            "Edit plan:",
            f"- User intent: {user_text}",
            "- Read the target files and preserve existing behavior where possible.",
            "- Isolate the smallest safe change before writing code.",
            "- Run a focused smoke test after the edit.",
        ])

        assistant = self._get_coding_assistant()
        if assistant is not None:
            context_blocks = []
            for path in paths[:2]:
                rel = str(path.relative_to(ROOT))
                try:
                    text = self._read_workspace_file(path)
                except OSError:
                    continue
                snippet = textwrap.shorten(" ".join(text.split()), width=600, placeholder=" ...")
                context_blocks.append(f"File: {rel}\nSnippet: {snippet}")
            prompt = (
                "You are planning a workspace edit.\n"
                f"User request: {user_text}\n\n"
                + "\n\n".join(context_blocks)
            )
            try:
                guidance = assistant.chat(prompt)
            except Exception as exc:
                guidance = f"Coding lane could not generate an edit suggestion: {exc}"
            lines.extend(["", "Coding lane suggestion:", guidance])
        else:
            detail = self._coding_load_error or "not loaded yet"
            lines.extend(["", f"Coding lane suggestion unavailable: {detail}"])

        lines.extend([
            "",
            "Note:",
            "- Doku now recognizes edit intent against workspace files, but actual file writes still happen through the main coding workflow rather than from the chat UI itself.",
        ])
        return "\n".join(lines)

    def _get_coding_assistant(self) -> Any | None:
        if self._coding_assistant is not None:
            return self._coding_assistant
        if self._coding_load_error is not None:
            return None
        try:
            from Brain.ai_coding_assistant import AICodingAssistant
            self._coding_assistant = AICodingAssistant()
            return self._coding_assistant
        except Exception as exc:
            self._coding_load_error = str(exc)
            return None

    def _handle_coding_request(self, user_text: str, lower: str) -> str:
        if lower in {"/coding", "coding status", "/coding status"}:
            assistant = self._get_coding_assistant()
            if assistant is None:
                detail = self._coding_load_error or "unknown import error"
                return f"Coding lane is unavailable right now: {detail}"
            return (
                "Coding lane is active. "
                f"Knowledge capsules: {len(assistant.code_capsules)} | "
                f"Failed-code capsules: {len(assistant.failed_code_capsules)} | "
                f"Conversations: {len(assistant.conversation_history)}"
            )

        assistant = self._get_coding_assistant()
        if assistant is None:
            detail = self._coding_load_error or "unknown import error"
            return (
                "I recognized this as a coding request, but the coding lane could not load. "
                f"Reason: {detail}"
            )

        response = assistant.chat(user_text)
        try:
            assistant.save_knowledge_base()
        except Exception:
            pass
        return response

    def _looks_like_coding_request(self, lower: str) -> bool:
        coding_phrases = (
            "write code",
            "generate code",
            "convert architecture",
            "convert code",
            "explain roca",
            "what is roca",
            "failed code",
            "debug this code",
            "review this code",
            "error in code",
            "python function",
            "capsule network",
            "visualize capsule",
            "math formula",
        )
        return any(phrase in lower for phrase in coding_phrases)

    def _looks_like_status_request(self, lower: str) -> bool:
        return any(phrase in lower for phrase in (
            "status",
            "memory status",
            "how many capsules",
            "how much memory",
        ))

    def _looks_like_sources_request(self, lower: str) -> bool:
        return any(phrase in lower for phrase in (
            "what files are loaded",
            "which files are loaded",
            "what sources are loaded",
            "list sources",
            "show sources",
            "loaded files",
        ))

    def _looks_like_rescan_request(self, lower: str) -> bool:
        return any(phrase in lower for phrase in (
            "rescan",
            "scan data",
            "refresh memory",
            "reload data",
        ))

    def _compose_grounded_response(
        self,
        user_text: str,
        matches: List[tuple[float, DokuCapsule]],
    ) -> str:
        lead = (
            "I searched my local ROCA-style memory and found the strongest matches "
            "for your request."
        )

        lines = [lead, "", f"Question: {user_text}", ""]
        for index, (score, capsule) in enumerate(matches, start=1):
            lines.append(f"{index}. {capsule.name} [{capsule.source_path}] score={score:.3f}")
            lines.append(f"   {capsule.text_excerpt}")

        best = matches[0][1]
        lines.extend([
            "",
            "Best grounded source:",
            f"{best.name} from {best.source_path}",
        ])
        return "\n".join(lines)

    def _status_text(self) -> str:
        stats = self.memory.stats()
        coding_state = "not loaded"
        if self._coding_assistant is not None:
            coding_state = f"active ({len(self._coding_assistant.code_capsules)} code capsules)"
        elif self._coding_load_error is not None:
            coding_state = f"error: {self._coding_load_error}"
        workspace_state = f"indexed ({len(self._workspace_files())} files)"
        return (
            "Doku status\n"
            "-----------\n"
            f"Data directory: {self.data_dir}\n"
            f"Capsules: {stats['capsules']}\n"
            f"Used capsules: {stats['used_capsules']}\n"
            f"Coding lane: {coding_state}\n"
            f"Workspace lane: {workspace_state}\n"
            f"Memory file: {MEMORY_PATH}\n"
            f"Journal file: {JOURNAL_PATH}"
        )

    def _sources_text(self) -> str:
        paths = sorted(capsule.source_path for capsule in self.memory.capsules.values())
        if not paths:
            return "No sources are loaded yet."
        preview = paths[:20]
        lines = ["Loaded sources:"]
        lines.extend(f"- {path}" for path in preview)
        if len(paths) > len(preview):
            lines.append(f"- ... and {len(paths) - len(preview)} more")
        return "\n".join(lines)

    def _help_text(self) -> str:
        return (
            "Commands:\n"
            "/help      show commands\n"
            "/status    show Doku memory status\n"
            "/sources   list loaded source files\n"
            "/rescan    rescan data/ and refresh memory\n"
            "/coding    show coding-lane status\n"
            "/workspace show workspace-lane status\n"
            "/quit      exit\n\n"
            "Supported lanes:\n"
            "- General memory: asks about ingested data/ files\n"
            "- Coding lane: explain roca, write code, debug code\n"
            "- Workspace lane: explain doku_main.py, review Camaron.py, edit doku_main.py to ..."
        )

    def orbital_snapshot(self) -> List[Dict[str, Any]]:
        nodes: List[Dict[str, Any]] = [
            {
                "id": "doku-core",
                "name": "Doku Core",
                "lane": "nucleus",
                "gravity": 1.0,
                "detail": "Core identity node for the active Doku session.",
            }
        ]

        top_memory = sorted(
            self.memory.capsules.values(),
            key=lambda capsule: (capsule.use_count, capsule.salience, capsule.updated_at),
            reverse=True,
        )[:12]
        for capsule in top_memory:
            gravity = max(0.12, min(0.98, 0.2 + capsule.salience))
            nodes.append({
                "id": capsule.id,
                "name": capsule.name,
                "lane": "memory",
                "gravity": gravity,
                "detail": f"{capsule.source_path} | uses={capsule.use_count} | salience={capsule.salience}",
            })

        for path in self._workspace_files()[:10]:
            rel = str(path.relative_to(ROOT))
            nodes.append({
                "id": f"workspace:{rel}",
                "name": path.name,
                "lane": "workspace",
                "gravity": 0.35,
                "detail": f"Workspace file: {rel}",
            })

        if self._coding_assistant is not None:
            capsules = list(self._coding_assistant.code_capsules.values())[:8]
            for capsule in capsules:
                score = 0.25 + min(0.65, 0.08 * capsule.success_count + 0.04)
                nodes.append({
                    "id": f"coding:{capsule.id}",
                    "name": capsule.name,
                    "lane": "coding",
                    "gravity": score,
                    "detail": f"Coding capsule: {capsule.name} | successes={capsule.success_count} | failures={capsule.failure_count}",
                })
        else:
            nodes.append({
                "id": "coding-lane-placeholder",
                "name": "Coding Lane",
                "lane": "coding",
                "gravity": 0.25,
                "detail": "Coding lane available on demand. Use /coding or a coding prompt to activate it.",
            })

        return nodes

    def log_turn(self, speaker: str, text: str) -> None:
        JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": _now(),
            "speaker": speaker,
            "text": text,
        }
        with JOURNAL_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")


if _HAS_QT:
    class OrbitalWidget(QWidget):
        node_selected = pyqtSignal(object)

        LANES = {
            "nucleus": {"min": 0.00, "max": 0.08, "color": QColor(255, 214, 102)},
            "coding": {"min": 0.18, "max": 0.36, "color": QColor(85, 165, 255)},
            "workspace": {"min": 0.46, "max": 0.66, "color": QColor(120, 220, 160)},
            "memory": {"min": 0.74, "max": 0.96, "color": QColor(255, 140, 110)},
        }

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setMinimumSize(420, 420)
            self._nodes: List[Dict[str, Any]] = []
            self._positions: List[tuple[Dict[str, Any], float, float, float]] = []
            self._phase = 0.0
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._tick)
            self._timer.start(40)

        def set_nodes(self, nodes: List[Dict[str, Any]]) -> None:
            self._nodes = nodes
            self.update()

        def _tick(self) -> None:
            self._phase += 0.004
            self.update()

        def mousePressEvent(self, event) -> None:
            x = event.position().x()
            y = event.position().y()
            for node, px, py, radius in reversed(self._positions):
                if (x - px) ** 2 + (y - py) ** 2 <= radius ** 2:
                    self.node_selected.emit(node)
                    return
            super().mousePressEvent(event)

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), QColor(15, 23, 42))

            width = self.width()
            height = self.height()
            center_x = width / 2
            center_y = height / 2
            max_orbit = min(width, height) / 2 - 30
            self._positions = []

            painter.setPen(QPen(QColor(71, 85, 105), 1))
            for lane in self.LANES.values():
                for fraction in (lane["min"], lane["max"]):
                    radius = max(8, fraction * max_orbit)
                    painter.drawEllipse(QPointF(center_x, center_y), radius, radius)

            painter.setPen(QPen(QColor(148, 163, 184), 1))
            painter.setFont(QFont("Segoe UI", 8))
            for lane_name, lane in self.LANES.items():
                mid = ((lane["min"] + lane["max"]) / 2) * max_orbit
                painter.drawText(QPointF(center_x + 8, center_y - mid + 2), lane_name.title())

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(255, 214, 102)))
            painter.drawEllipse(QPointF(center_x, center_y), 14, 14)
            painter.setPen(QPen(QColor(15, 23, 42), 1))
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.drawText(QPointF(center_x - 12, center_y + 4), "Doku")

            painter.setFont(QFont("Segoe UI", 7))
            for index, node in enumerate(self._nodes):
                lane = self.LANES.get(node.get("lane", "memory"), self.LANES["memory"])
                gravity = float(node.get("gravity", 0.4))
                orbit_fraction = lane["min"] + (1.0 - gravity) * (lane["max"] - lane["min"])
                orbit_radius = max(6, orbit_fraction * max_orbit)
                angle_seed = int(hashlib.md5(node["id"].encode("utf-8")).hexdigest()[:8], 16)
                angle = ((angle_seed / 0xFFFFFFFF) * math.pi * 2.0) + self._phase * (1.0 + index * 0.02)
                px = center_x + orbit_radius * math.cos(angle)
                py = center_y + orbit_radius * math.sin(angle)
                node_radius = 5 + min(8, gravity * 8)
                color = QColor(lane["color"])
                color.setAlpha(220)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(color))
                painter.drawEllipse(QPointF(px, py), node_radius, node_radius)
                painter.setPen(QPen(QColor(226, 232, 240), 1))
                painter.drawText(QPointF(px + 6, py - 2), str(node.get("name", "node"))[:18])
                self._positions.append((node, px, py, node_radius + 4))


    class DokuChatWindow(QMainWindow):
        def __init__(self, entity: DokuEntity | None = None) -> None:
            super().__init__()
            self.entity = entity or DokuEntity()
            self.setWindowTitle("Doku")
            self.resize(1120, 780)
            self._build_ui()
            self._append_message("Doku", self.entity.bootstrap())
            self._append_message(
                "Doku",
                "Ask about local data, coding tasks, or workspace files. Quick actions: Status, Sources, Rescan, Coding, Workspace, Help.",
            )
            self._refresh_visuals()

        def _build_ui(self) -> None:
            central = QWidget()
            self.setCentralWidget(central)

            root = QVBoxLayout(central)
            root.setContentsMargins(14, 14, 14, 14)
            root.setSpacing(10)

            title = QLabel("Doku")
            title_font = QFont()
            title_font.setPointSize(18)
            title_font.setBold(True)
            title.setFont(title_font)
            title.setStyleSheet("QLabel { color: #e2e8f0; }")
            root.addWidget(title)

            self.status_label = QLabel(f"Data root: {self.entity.data_dir}")
            self.status_label.setStyleSheet(
                "QLabel { color: #94a3b8; font-size: 12px; padding: 2px; }"
            )
            root.addWidget(self.status_label)

            self.tabs = QTabWidget()
            self.tabs.setStyleSheet(
                "QTabWidget::pane { border: 1px solid #1e293b; background: #0f172a; }"
                "QTabBar::tab { background: #1e293b; color: #cbd5e1; padding: 8px 14px; margin-right: 2px; }"
                "QTabBar::tab:selected { background: #2563eb; color: white; }"
            )
            root.addWidget(self.tabs, 1)

            self._build_chat_tab()
            self._build_orbital_tab()

        def _build_chat_tab(self) -> None:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(10)

            self.chat_view = QTextEdit()
            self.chat_view.setReadOnly(True)
            self.chat_view.setFont(QFont("Consolas", 10))
            self.chat_view.setStyleSheet(
                "QTextEdit { background: #0f172a; color: #e2e8f0; border: 1px solid #1e293b; padding: 10px; }"
            )
            layout.addWidget(self.chat_view, 1)

            action_row = QHBoxLayout()
            self.status_button = QPushButton("Status")
            self.sources_button = QPushButton("Sources")
            self.rescan_button = QPushButton("Rescan")
            self.coding_button = QPushButton("Coding")
            self.workspace_button = QPushButton("Workspace")
            self.help_button = QPushButton("Help")
            for button in (
                self.status_button,
                self.sources_button,
                self.rescan_button,
                self.coding_button,
                self.workspace_button,
                self.help_button,
            ):
                button.setStyleSheet(
                    "QPushButton { background: #e2e8f0; border: 1px solid #cbd5e1; padding: 6px 12px; }"
                    "QPushButton:hover { background: #f8fafc; }"
                )
                action_row.addWidget(button)
            action_row.addStretch(1)
            layout.addLayout(action_row)

            input_row = QHBoxLayout()
            self.input_line = QLineEdit()
            self.input_line.setPlaceholderText("Ask Doku about data/, code, or workspace files")
            self.input_line.setStyleSheet(
                "QLineEdit { background: #ffffff; border: 1px solid #cbd5e1; padding: 8px; }"
            )
            self.send_button = QPushButton("Send")
            self.send_button.setStyleSheet(
                "QPushButton { background: #2563eb; color: white; border: 0; padding: 8px 16px; }"
                "QPushButton:hover { background: #1d4ed8; }"
            )
            input_row.addWidget(self.input_line, 1)
            input_row.addWidget(self.send_button)
            layout.addLayout(input_row)

            self.send_button.clicked.connect(self._send_current_message)
            self.input_line.returnPressed.connect(self._send_current_message)
            self.status_button.clicked.connect(lambda: self._run_quick_command("/status"))
            self.sources_button.clicked.connect(lambda: self._run_quick_command("/sources"))
            self.rescan_button.clicked.connect(lambda: self._run_quick_command("/rescan"))
            self.coding_button.clicked.connect(lambda: self._run_quick_command("/coding"))
            self.workspace_button.clicked.connect(lambda: self._run_quick_command("/workspace"))
            self.help_button.clicked.connect(lambda: self._run_quick_command("/help"))

            self.tabs.addTab(tab, "Chat")

        def _build_orbital_tab(self) -> None:
            tab = QWidget()
            layout = QHBoxLayout(tab)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(12)

            self.orbital_widget = OrbitalWidget()
            self.orbital_widget.node_selected.connect(self._show_orbital_detail)
            layout.addWidget(self.orbital_widget, 2)

            side = QVBoxLayout()
            side.setSpacing(8)
            side.addWidget(QLabel("Orbital Details"))

            self.orbital_details = QTextEdit()
            self.orbital_details.setReadOnly(True)
            self.orbital_details.setStyleSheet(
                "QTextEdit { background: #0f172a; color: #e2e8f0; border: 1px solid #1e293b; padding: 10px; }"
            )
            side.addWidget(self.orbital_details, 1)

            refresh_button = QPushButton("Refresh Orbit")
            refresh_button.setStyleSheet(
                "QPushButton { background: #e2e8f0; border: 1px solid #cbd5e1; padding: 6px 12px; }"
                "QPushButton:hover { background: #f8fafc; }"
            )
            refresh_button.clicked.connect(self._refresh_visuals)
            side.addWidget(refresh_button)

            layout.addLayout(side, 1)
            self.tabs.addTab(tab, "Orbital")

        def _append_message(self, speaker: str, text: str) -> None:
            self.chat_view.append(f"{speaker}> {text}")
            self.chat_view.append("")
            cursor = self.chat_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.chat_view.setTextCursor(cursor)

        def _run_quick_command(self, command: str) -> None:
            self.input_line.setText(command)
            self._send_current_message()

        def _show_orbital_detail(self, node: Dict[str, Any]) -> None:
            detail = [
                f"Name: {node.get('name', '')}",
                f"Lane: {node.get('lane', '')}",
                f"Gravity: {node.get('gravity', 0):.3f}",
                "",
                str(node.get('detail', '')),
            ]
            self.orbital_details.setPlainText("\n".join(detail))

        def _refresh_visuals(self) -> None:
            stats = self.entity.memory.stats()
            self.status_label.setText(
                f"Data root: {self.entity.data_dir} | Capsules: {stats['capsules']} | Used: {stats['used_capsules']} | Workspace: {len(self.entity._workspace_files())}"
            )
            nodes = self.entity.orbital_snapshot()
            self.orbital_widget.set_nodes(nodes)
            self.orbital_details.setPlainText(
                "ROCA-style orbital view\n\n"
                "- nucleus: Doku core\n"
                "- coding: code generation and ROCA knowledge\n"
                "- workspace: repo-aware file lane\n"
                "- memory: ingested data capsules\n\n"
                "Click a node to inspect it."
            )

        def _send_current_message(self) -> None:
            user_text = self.input_line.text().strip()
            if not user_text:
                return

            self.input_line.clear()
            self.entity.log_turn("user", user_text)
            self._append_message("You", user_text)
            reply = self.entity.answer(user_text)
            self.entity.log_turn("doku", reply)
            self._append_message("Doku", reply)
            self._refresh_visuals()



def _safe_console_text(text: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")



def run_cli() -> int:
    entity = DokuEntity()
    boot_message = entity.bootstrap()
    print(_safe_console_text(boot_message))
    print(_safe_console_text("Type /help for commands."))

    while True:
        try:
            user_text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Doku.")
            return 0

        if user_text.lower() in {"/quit", "quit", "exit"}:
            print(_safe_console_text("Exiting Doku."))
            return 0

        entity.log_turn("user", user_text)
        reply = entity.answer(user_text)
        entity.log_turn("doku", reply)
        print(_safe_console_text(f"Doku> {reply}"))



def run_gui() -> int:
    if not _HAS_QT or QApplication is None:
        print("PyQt6 is unavailable, falling back to CLI mode.")
        return run_cli()

    app = QApplication(sys.argv)
    window = DokuChatWindow()
    window.show()
    return app.exec()



def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Doku launcher")
    parser.add_argument("--cli", action="store_true", help="run the terminal chat instead of the GUI")
    args = parser.parse_args(argv)

    if args.cli:
        return run_cli()
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
