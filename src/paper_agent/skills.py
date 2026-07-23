from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SkillInfo:
    name: str
    path: Path
    description: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_skills_dir() -> Path:
    return repo_root() / "skills"


def read_skill_description(skill_md: Path) -> str:
    text = skill_md.read_text(encoding="utf-8", errors="replace")
    frontmatter = parse_skill_frontmatter(text)
    if frontmatter.get("description"):
        return frontmatter["description"][:500]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    # Convention: first heading is name, next paragraph-ish lines give the purpose.
    desc_lines: list[str] = []
    for line in lines[1:12]:
        if line.startswith("## "):
            break
        if not line.startswith("#"):
            desc_lines.append(line)
    return " ".join(desc_lines)[:500]


def parse_skill_frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", text, re.DOTALL)
    if not match:
        return {}
    fields: dict[str, str] = {}
    for raw_line in match.group(1).splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in {"name", "description"} and value:
            fields[key] = value
    return fields


def discover_skills(skills_dir: str | Path | None = None) -> list[SkillInfo]:
    root = Path(skills_dir) if skills_dir is not None else default_skills_dir()
    if not root.exists():
        return []
    skills: list[SkillInfo] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        skill_md = child / "SKILL.md"
        if skill_md.exists():
            frontmatter = parse_skill_frontmatter(
                skill_md.read_text(encoding="utf-8", errors="replace")
            )
            skills.append(
                SkillInfo(
                    name=frontmatter.get("name", child.name),
                    path=child,
                    description=read_skill_description(skill_md),
                )
            )
    return skills


def skill_paths(skills_dir: str | Path | None = None) -> list[str]:
    return [str(info.path) for info in discover_skills(skills_dir)]


def skill_manifest(skills_dir: str | Path | None = None) -> list[dict[str, str]]:
    return [
        {"name": info.name, "path": str(info.path), "description": info.description}
        for info in discover_skills(skills_dir)
    ]


def format_skill_manifest_for_prompt(skills_dir: str | Path | None = None) -> str:
    skills = discover_skills(skills_dir)
    if not skills:
        return "No skills discovered."
    return "\n".join(f"- {s.name}: {s.description}" for s in skills)


def create_deep_agent_with_optional_skills(create_deep_agent_fn: Any, **kwargs: Any) -> Any:
    """Call create_deep_agent with skills when supported, fallback otherwise.

    Deep Agents skills are optional across package versions. This keeps the starter runnable
    even when a user's installed deepagents version predates the skills argument.
    """
    paths = skill_paths()
    try:
        parameters = inspect.signature(create_deep_agent_fn).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_any = any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values())
    if "memory" in kwargs and parameters and not accepts_any and "memory" not in parameters:
        kwargs.pop("memory")
    if paths and (not parameters or accepts_any or "skills" in parameters):
        kwargs["skills"] = paths
    return create_deep_agent_fn(**kwargs)
