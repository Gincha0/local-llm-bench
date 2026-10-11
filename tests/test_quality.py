import json

from local_llm_bench.quality import json_matches

EXPECTED = {
    "device": "sw-core-02",
    "interface": "sfp-sfpplus3",
    "event": "link down",
    "timestamp": "2026-09-14T09:47:58Z",
}


def answer(**changes: str) -> str:
    return json.dumps(EXPECTED | changes)


def test_the_right_answer_matches() -> None:
    assert json_matches(answer(), EXPECTED)


def test_spelling_of_the_event_does_not_matter() -> None:
    assert json_matches(answer(event="Link-Down"), EXPECTED)


def test_a_wrong_value_fails() -> None:
    # picking the ether1 line below it: the most common wrong answer across the runs
    assert not json_matches(answer(interface="ether1"), EXPECTED)


def test_an_extra_or_missing_key_fails() -> None:
    assert not json_matches(answer(severity="warning"), EXPECTED)
    assert not json_matches(json.dumps({"device": "sw-core-02"}), EXPECTED)


def test_text_that_is_not_json_fails() -> None:
    assert not json_matches("The link went down at 09:47:58.", EXPECTED)
    assert not json_matches('["sw-core-02"]', EXPECTED)
