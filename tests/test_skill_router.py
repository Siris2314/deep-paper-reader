from paper_agent.skill_router import route_question_to_skills


def names(question: str) -> list[str]:
    return [trigger.name for trigger in route_question_to_skills(question)]


def test_math_question_triggers_math_skill():
    routed = names("Explain the softmax equation and derive the loss.")
    assert "math-walkthrough" in routed
    assert "chat-answer" in routed


def test_results_question_triggers_visual_skill():
    routed = names("Which benchmark table supports the result?")
    assert "figure-table-analysis" in routed


def test_prior_work_question_triggers_research_lineage_skill():
    routed = names("Which prior paper does this method build on, and how was it extended?")

    assert "research-lineage" in routed
