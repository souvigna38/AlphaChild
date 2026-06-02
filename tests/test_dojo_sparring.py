"""Sparring Partner guardrails and parsing."""

from dojo.sparring_partner import (
    SOCRATIC_SYSTEM_PROMPT,
    build_messages,
    extract_question,
    guardrail_check,
    guardrail_check as check,
)


def test_extract_question_with_mention():
    q = extract_question("<@123456789> Why is my tensor wrong?", 123456789)
    assert q == "Why is my tensor wrong?"


def test_extract_question_no_mention():
    assert extract_question("hello world", 99) is None


def test_guardrail_blocks_large_code():
    big = "```python\n" + "\n".join(f"x = {i}" for i in range(30)) + "\n```"
    assert check("ok " + big) is not None


def test_guardrail_blocks_pass_token():
    assert check("Use PASS-C2-L12-abcdef012345") is not None


def test_system_prompt_forbids_solutions():
    assert "NEVER write complete code" in SOCRATIC_SYSTEM_PROMPT


def test_build_messages_includes_system():
    msgs = build_messages("shape mismatch?")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["content"] == "shape mismatch?"
