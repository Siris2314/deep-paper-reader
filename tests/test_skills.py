from pathlib import Path

from paper_agent.skills import (
    create_deep_agent_with_optional_skills,
    discover_skills,
    parse_skill_frontmatter,
)


def test_repo_skills_have_agent_skill_frontmatter():
    root = Path(__file__).resolve().parents[1]
    skills = discover_skills(root / "skills")

    assert skills
    for info in skills:
        frontmatter = parse_skill_frontmatter((info.path / "SKILL.md").read_text(encoding="utf-8"))
        assert frontmatter["name"] == info.name
        assert frontmatter["description"] == info.description
        assert len(info.description) > 40


def test_agent_factory_uses_memory_when_supported():
    captured = {}

    def modern_factory(*, model=None, skills=None, memory=None):
        captured.update(model=model, skills=skills, memory=memory)
        return captured

    create_deep_agent_with_optional_skills(
        modern_factory,
        model="chat-model",
        memory=["memory.md"],
    )

    assert captured["memory"] == ["memory.md"]
    assert captured["skills"]


def test_agent_factory_drops_unsupported_memory_and_skills():
    def legacy_factory(*, model=None):
        return model

    assert (
        create_deep_agent_with_optional_skills(
            legacy_factory,
            model="chat-model",
            memory=["memory.md"],
        )
        == "chat-model"
    )
