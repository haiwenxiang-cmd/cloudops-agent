"""Pure helpers shared by mutation journaling, policy and verification."""

import hashlib
import hmac
import json
import re
from typing import Any, Dict, Iterable, List

import yaml

_SENSITIVE_KEY = re.compile(
    r"(^|_)(password|passwd|secret|token|api_?key|authorization|credential)s?($|_)",
    re.IGNORECASE,
)


def redact_sensitive(value: Any, key: str = "") -> Any:
    if key and _SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): redact_sensitive(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hmac_fingerprint(secret: str, *values: Any) -> str:
    payload = "\x1f".join(canonical_json(value) for value in values).encode("utf-8")
    key = (secret or "skyflo-unsafe-development-key").encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def redact_manifest_content(content: str) -> str:
    """Remove Secret payloads before a Kubernetes manifest enters the journal."""
    documents: List[Dict[str, Any]] = []
    try:
        parsed = yaml.safe_load_all(content or "")
        for raw in parsed:
            if not isinstance(raw, dict):
                continue
            if str(raw.get("kind") or "").lower() == "secret":
                if "data" in raw:
                    raw["data"] = "[REDACTED]"
                if "stringData" in raw:
                    raw["stringData"] = "[REDACTED]"
            documents.append(redact_sensitive(raw))
    except yaml.YAMLError:
        return "[UNPARSEABLE MANIFEST REDACTED]"
    return yaml.safe_dump_all(documents, sort_keys=False, allow_unicode=True)


def redact_tool_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Build an audit/SSE-safe copy while preserving the execution arguments."""
    redacted = redact_sensitive(args)
    if tool_name == "k8s_apply" and isinstance(redacted.get("content"), str):
        redacted["content"] = redact_manifest_content(redacted["content"])
    if tool_name == "helm_install_with_values" and isinstance(redacted.get("values"), str):
        try:
            parsed_values = yaml.safe_load(redacted["values"])
            redacted["values"] = yaml.safe_dump(
                redact_sensitive(parsed_values), sort_keys=False, allow_unicode=True
            )
        except yaml.YAMLError:
            redacted["values"] = "[UNPARSEABLE HELM VALUES REDACTED]"
    return redacted


def parse_manifest_targets(
    content: str, default_namespace: str = "default"
) -> List[Dict[str, Any]]:
    """Parse a multi-document manifest into non-secret verification/policy targets."""
    targets: List[Dict[str, Any]] = []
    for raw in yaml.safe_load_all(content or ""):
        if not isinstance(raw, dict):
            continue
        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        kind = str(raw.get("kind") or "").strip()
        name = str(metadata.get("name") or "").strip()
        if not kind or not name:
            continue
        namespace = str(metadata.get("namespace") or default_namespace).strip()
        target: Dict[str, Any] = {
            "api_version": str(raw.get("apiVersion") or "v1"),
            "kind": kind,
            "name": name,
            "namespace": namespace,
        }
        if kind.lower() != "secret" and isinstance(raw.get("spec"), dict):
            target["desired_spec"] = raw["spec"]
        targets.append(target)
    return targets


def target_namespaces(targets: Iterable[Dict[str, Any]]) -> set[str]:
    return {str(target.get("namespace") or "default") for target in targets}


def attach_operation_metadata(
    tool_name: str,
    args: Dict[str, Any],
    operation_id: str,
    desired_state_hmac: str,
) -> Dict[str, Any]:
    """Return execution args carrying an external reconciliation marker."""
    prepared = dict(args)
    if tool_name == "k8s_apply":
        documents: List[Dict[str, Any]] = []
        for raw in yaml.safe_load_all(str(args.get("content") or "")):
            if not isinstance(raw, dict):
                continue
            metadata = raw.setdefault("metadata", {})
            annotations = metadata.setdefault("annotations", {})
            annotations["skyflo.ai/operation-id"] = operation_id
            annotations["skyflo.ai/desired-state-hmac"] = desired_state_hmac
            documents.append(raw)
        prepared["content"] = yaml.safe_dump_all(
            documents, sort_keys=False, allow_unicode=True
        )
    elif tool_name in {
        "helm_install",
        "helm_install_with_values",
        "helm_upgrade",
        "helm_rollback",
    }:
        prepared["skyflo_operation_id"] = operation_id
    return prepared
