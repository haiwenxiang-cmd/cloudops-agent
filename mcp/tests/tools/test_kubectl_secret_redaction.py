from tools.kubectl import _redact_secret_payload


def test_redacts_single_secret_data_but_preserves_metadata():
    payload = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": "demo", "annotations": {"operation": "123"}},
        "data": {"token": "c2VjcmV0"},
        "stringData": {"password": "secret"},
    }
    sanitized = _redact_secret_payload(payload)
    assert "data" not in sanitized
    assert "stringData" not in sanitized
    assert sanitized["metadata"]["name"] == "demo"


def test_redacts_secret_list_items():
    payload = {
        "kind": "SecretList",
        "items": [
            {"kind": "Secret", "metadata": {"name": "one"}, "data": {"key": "dmFsdWU="}}
        ],
    }
    sanitized = _redact_secret_payload(payload)
    assert "data" not in sanitized["items"][0]
