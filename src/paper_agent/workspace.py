from __future__ import annotations

import json
import shutil
from pathlib import Path

WORKSPACE_DIRS = [
    "raw",
    "parsed",
    "scratch",
    "web",
    "research",
    "final",
    "logs",
    "concepts",
    "chat",
    "memory",
    "math",
]


class Workspace:
    """Structured workspace for context engineering.

    The agent should avoid keeping every artifact in the prompt. Large artifacts live here.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.ensure()

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in WORKSPACE_DIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        """Delete generated workspace artifacts while keeping the workspace folder itself.

        This prevents stale final reports from a previous PDF from appearing after a parse-only
        run or failed agent run. Only known generated subdirectories are removed.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        for name in WORKSPACE_DIRS:
            path = self.root / name
            if path.exists():
                shutil.rmtree(path)
        # Remove old backward-compatible flat files created by previous versions.
        for name in [
            "extracted_text.md",
            "metadata.json",
            "sections.json",
            "equation_candidates.json",
            "final_study_guide.md",
        ]:
            path = self.root / name
            if path.exists() and path.is_file():
                path.unlink()
        self.ensure()

    def path(self, rel_path: str | Path) -> Path:
        rel = Path(rel_path)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Unsafe workspace path: {rel_path}")
        return self.root / rel

    def write_text(self, rel_path: str | Path, content: str) -> Path:
        path = self.path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def write_json(self, rel_path: str | Path, obj: object) -> Path:
        path = self.path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def read_text(self, rel_path: str | Path, max_chars: int | None = None) -> str:
        path = self.path(rel_path)
        text = path.read_text(encoding="utf-8")
        if max_chars is not None and len(text) > max_chars:
            return text[:max_chars] + "\n\n[TRUNCATED BY read_text max_chars]"
        return text

    def list_files(self) -> list[str]:
        files = []
        for path in self.root.rglob("*"):
            if path.is_file():
                files.append(str(path.relative_to(self.root)).replace("\\", "/"))
        return sorted(files)
