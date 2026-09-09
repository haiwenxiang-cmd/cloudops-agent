"""Deterministic authorization policy for infrastructure tool mutations."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import settings
from .mutation_utils import parse_manifest_targets, target_namespaces


@dataclass(frozen=True)
class MutationPolicyDecision:
    allowed: bool
    rule: str
    reason: str
    risk_level: str = "medium"
    targets: List[Dict[str, Any]] = field(default_factory=list)


class MutationPolicyService:
    """Fail-closed policy decision point. Approval cannot override a denial."""

    _FORBIDDEN_KINDS = {
        "namespace",
        "customresourcedefinition",
        "clusterrole",
        "clusterrolebinding",
        "role",
        "rolebinding",
        "validatingwebhookconfiguration",
        "mutatingwebhookconfiguration",
    }
    _CLUSTER_MUTATION_TOOLS = {
        "k8s_cordon",
        "k8s_uncordon",
        "k8s_drain",
    }

    def _allowed_namespaces(self) -> set[str]:
        raw = settings.MUTATION_ALLOWED_NAMESPACES
        return {item.strip() for item in raw.split(",") if item.strip()}

    def evaluate(
        self,
        *,
        tool_name: str,
        args: Dict[str, Any],
        user_id: Optional[str],
        user_role: Optional[str],
        environment: Optional[str] = None,
    ) -> MutationPolicyDecision:
        environment = (environment or settings.SKYFLO_ENVIRONMENT).strip().lower()
        role = (user_role or "").strip().lower()

        if not user_id:
            return MutationPolicyDecision(
                False,
                "authenticated_mutation_required",
                "Mutation denied: an authenticated user is required.",
                "high",
            )
        if role not in {"admin", "superuser"}:
            return MutationPolicyDecision(
                False,
                "admin_mutation_required",
                "Mutation denied: role "
                f"'{role or 'unknown'}' is not allowed to mutate infrastructure.",
                "high",
            )

        targets: List[Dict[str, Any]] = []
        if tool_name == "k8s_apply":
            try:
                targets = parse_manifest_targets(
                    str(args.get("content") or ""), str(args.get("namespace") or "default")
                )
            except Exception as exc:
                return MutationPolicyDecision(
                    False,
                    "manifest_parse_failed",
                    f"Mutation denied: manifest could not be parsed safely ({exc}).",
                    "high",
                )
            if not targets:
                return MutationPolicyDecision(
                    False,
                    "manifest_has_no_targets",
                    "Mutation denied: manifest contains no identifiable Kubernetes resources.",
                    "high",
                )
            forbidden = sorted(
                {
                    t["kind"]
                    for t in targets
                    if str(t.get("kind", "")).lower() in self._FORBIDDEN_KINDS
                }
            )
            if forbidden:
                return MutationPolicyDecision(
                    False,
                    "cluster_or_rbac_resource_denied",
                    "Mutation denied by policy for resource kinds: " + ", ".join(forbidden),
                    "critical",
                    targets,
                )

        resource_type = str(args.get("resource_type") or "").lower().replace("-", "")
        if resource_type.rstrip("s") in self._FORBIDDEN_KINDS:
            return MutationPolicyDecision(
                False,
                "cluster_or_rbac_resource_denied",
                f"Mutation denied by policy for resource type '{resource_type}'.",
                "critical",
            )

        if tool_name in self._CLUSTER_MUTATION_TOOLS and environment == "production":
            return MutationPolicyDecision(
                False,
                "production_cluster_mutation_denied",
                f"Mutation tool '{tool_name}' is disabled in production.",
                "critical",
            )

        namespaces = target_namespaces(targets) if targets else {
            str(args.get("namespace") or "default")
        }
        allowed_namespaces = self._allowed_namespaces()
        disallowed = sorted(
            namespace for namespace in namespaces if namespace not in allowed_namespaces
        )
        if disallowed:
            return MutationPolicyDecision(
                False,
                "namespace_not_allowed",
                "Mutation denied outside namespace allowlist: " + ", ".join(disallowed),
                "high",
                targets,
            )

        risk = "high" if tool_name.startswith("helm_") or "delete" in tool_name else "medium"
        return MutationPolicyDecision(
            True,
            "admin_allowed_namespace",
            f"Authenticated admin mutation allowed in {environment} for namespaces: "
            + ", ".join(sorted(namespaces)),
            risk,
            targets,
        )
