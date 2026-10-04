from paper_agent.paper_chat import ChatEvidence, verify_chat_answer


def evidence(text="The system scores 71 percent accuracy on two datasets."):
    return [ChatEvidence(id="P1.1", source="paper", kind="page", text=text, page=1)]


def test_existing_citation_does_not_prove_claim():
    result = verify_chat_answer(
        "The system achieves 100 percent accuracy on every possible dataset [P1.1].",
        ["P1.1"],
        evidence(),
    )
    assert not result.passed
    assert result.claims[0].status == "unresolved"


def test_verbatim_sentence_avoids_model_call():
    def never(_):
        raise AssertionError("No model needed")

    result = verify_chat_answer(
        "The system scores 71 percent accuracy on two datasets [P1.1].",
        ["P1.1"],
        evidence(),
        judge=never,
    )
    assert result.passed


def test_substring_cannot_drop_negation():
    result = verify_chat_answer(
        "The system achieves perfect accuracy on every dataset [P1.1].",
        ["P1.1"],
        evidence("It is false that the system achieves perfect accuracy on every dataset."),
    )
    assert not result.passed


def test_judge_contradiction_is_exposed():
    def judge(claims):
        return [
            {
                "index": 0,
                "status": "contradicted",
                "quote": "The system scores 71 percent accuracy on two datasets.",
                "reason": "The number and scope disagree.",
            }
        ]

    result = verify_chat_answer(
        "The system achieves 100 percent accuracy on every possible dataset [P1.1].",
        ["P1.1"],
        evidence(),
        judge=judge,
    )
    assert result.claims[0].status == "contradicted"
    assert not result.passed


def test_invented_quote_and_numerical_support_fail_closed():
    for quote in ["The system is always perfect.", evidence()[0].text]:
        result = verify_chat_answer(
            "The system scores 100 percent accuracy on two datasets [P1.1].",
            ["P1.1"],
            evidence(),
            judge=lambda _: [{"index": 0, "status": "supported", "quote": quote}],
        )
        assert not result.passed


def test_uncited_sentence_and_hidden_invalid_inline_id_fail():
    result = verify_chat_answer(
        "The system scores 71 percent accuracy on two datasets [P1.1]. It also solves every problem [P9.1].",
        ["P1.1"],
        evidence(),
    )
    assert not result.passed
    assert not result.checks["citations_resolve"]


def test_omitted_claims_and_oversized_answers_fail_closed():
    answer = "\n".join(f"Unverified statement number {i} [P1.1]." for i in range(25))
    result = verify_chat_answer(answer, ["P1.1"], evidence(), judge=lambda _: [])
    assert not result.passed
    assert not result.checks["all_claims_checked"]


def test_explicit_inference_is_not_presented_as_verified():
    result = verify_chat_answer(
        "Inference: this method may generalize to larger datasets [P1.1].",
        ["P1.1"],
        evidence(),
        judge=lambda _: [
            {"index": 0, "status": "inference", "reason": "Not tested in the passage."}
        ],
    )
    assert result.claims[0].status == "inference"
    assert not result.passed
