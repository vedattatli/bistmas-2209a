from bistmas.shared.hashing import canonical_json, sha256_json, sha256_text


def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1})


def test_sha256_json_changes_with_content():
    assert sha256_json({"x": 1}) != sha256_json({"x": 2})
    assert sha256_json({"x": 1}) == sha256_json({"x": 1})


def test_sha256_text_known_value():
    assert sha256_text("abc").startswith("ba7816bf8f01cfea")
