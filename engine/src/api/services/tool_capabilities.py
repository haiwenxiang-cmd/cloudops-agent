"""Authoritative runtime capabilities for externally visible tools.

MCP annotations are useful descriptive metadata, but they are not an
authorization boundary.  Infrastructure tools therefore have to appear in
this registry before they can be exposed or executed.  Unknown and
mis-annotated infrastructure tools fail closed.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ToolEffect(str, Enum):
    READ = "read"
    MUTATION = "mutation"


@dataclass(frozen=True)
class ToolCapability:
    effect: ToolEffect
    enabled: bool = True
    controlled: bool = False
    verification_adapter: Optional[str] = None

    @property
    def requires_approval(self) -> bool:
        return self.effect == ToolEffect.MUTATION


def _read() -> ToolCapability:
    return ToolCapability(ToolEffect.READ)


def _controlled(adapter: str) -> ToolCapability:
    return ToolCapability(
        ToolEffect.MUTATION,
        enabled=True,
        controlled=True,
        verification_adapter=adapter,
    )


def _disabled_mutation() -> ToolCapability:
    return ToolCapability(ToolEffect.MUTATION, enabled=False)


READ_ONLY_TOOLS = {
    "k8s_logs",
    "k8s_get",
    "k8s_describe",
    "wait_for_x_seconds",
    "k8s_rollout_status",
    "k8s_rollout_history",
    "k8s_cluster_info",
    "k8s_top_pods",
    "k8s_top_nodes",
    "helm_list_releases",
    "helm_status",
    "helm_history",
    "helm_get_values",
    "helm_get_manifest",
    "helm_show_values",
    "helm_search_repo",
    "helm_template",
    "argo_list_rollouts",
    "argo_status",
    "argo_history",
    "argo_describe",
    "argo_list_experiments",
    "argo_list_analysisruns",
    "jenkins_get_job",
    "jenkins_get_jobs",
    "jenkins_get_build",
    "jenkins_get_last_builds",
    "jenkins_get_build_log",
    "jenkins_get_job_scm",
    "jenkins_get_build_scm",
    "jenkins_get_build_changesets",
    "jenkins_whoami",
    "jenkins_get_job_parameters",
}

CONTROLLED_MUTATIONS = {
    "k8s_apply": _controlled("kubernetes_apply"),
    "helm_install": _controlled("helm_release"),
    "helm_install_with_values": _controlled("helm_release"),
    "helm_upgrade": _controlled("helm_release"),
    "helm_rollback": _controlled("helm_release"),
}

# Code for these tools remains available in MCP, but Engine does not expose or
# execute them until a deterministic verification adapter is registered.
UNVERIFIED_MUTATIONS = {
    "k8s_patch",
    "k8s_set_image",
    "k8s_rollout_restart",
    "k8s_scale",
    "k8s_delete",
    "k8s_rollout_undo",
    "k8s_cordon",
    "k8s_uncordon",
    "k8s_drain",
    "k8s_run_pod",
    "k8s_exec",
    "k8s_port_forward",
    "helm_repo_add",
    "helm_repo_update",
    "helm_repo_remove",
    "helm_uninstall",
    "argo_promote",
    "argo_pause_rollout",
    "argo_resume_rollout",
    "argo_abort_rollout",
    "argo_set_image",
    "argo_rollout_restart",
    "argo_undo",
    "jenkins_trigger_build",
    "jenkins_update_build",
    "jenkins_stop_build",
}

TOOL_CAPABILITIES: Dict[str, ToolCapability] = {
    **{name: _read() for name in READ_ONLY_TOOLS},
    **CONTROLLED_MUTATIONS,
    **{name: _disabled_mutation() for name in UNVERIFIED_MUTATIONS},
}

INFRASTRUCTURE_PREFIXES = ("k8s_", "helm_", "argo_", "jenkins_")


def is_infrastructure_tool(name: str) -> bool:
    return name.startswith(INFRASTRUCTURE_PREFIXES) or name == "wait_for_x_seconds"


def get_tool_capability(name: str) -> Optional[ToolCapability]:
    return TOOL_CAPABILITIES.get(name)


def capability_error(name: str, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return a fail-closed reason for an unavailable or conflicting tool."""
    if not is_infrastructure_tool(name):
        return None
    if metadata is None:
        return f"Infrastructure tool '{name}' metadata is unavailable; execution fails closed."
    capability = get_tool_capability(name)
    if capability is None:
        return f"Infrastructure tool '{name}' is absent from the runtime capability registry."
    if not capability.enabled:
        return (
            f"Infrastructure mutation '{name}' is disabled because no deterministic "
            "verification adapter is registered."
        )
    annotations = (metadata or {}).get("annotations") or {}
    annotated_read_only = bool(annotations.get("readOnlyHint", False))
    expected_read_only = capability.effect == ToolEffect.READ
    if annotated_read_only != expected_read_only:
        return (
            f"Infrastructure tool '{name}' metadata conflicts with the runtime "
            "capability registry."
        )
    return None


def is_effectively_read_only(name: str, metadata: Optional[Dict[str, Any]]) -> bool:
    capability = get_tool_capability(name)
    if capability is not None:
        return capability.effect == ToolEffect.READ
    annotations = (metadata or {}).get("annotations") or {}
    return bool(annotations.get("readOnlyHint", False))
