"""Tests for tools.jenkins module."""

import base64
import json

import httpx
import pytest

from tools.jenkins import (
    JenkinsClient,
    _parse_credentials_ref,
    build_job_path,
    resolve_credentials_from_k8s,
)


class TestParseCredentialsRef:
    """Test cases for _parse_credentials_ref function."""

    def test_parse_credentials_ref_valid(self):
        """Test parsing valid credentials reference."""
        namespace, name = _parse_credentials_ref("default/my-secret")
        assert namespace == "default"
        assert name == "my-secret"

    def test_parse_credentials_ref_valid_with_hyphens(self):
        """Test parsing credentials reference with hyphens."""
        namespace, name = _parse_credentials_ref("production/my-app-secret")
        assert namespace == "production"
        assert name == "my-app-secret"

    def test_parse_credentials_ref_invalid_format(self):
        """Test parsing invalid credentials reference format."""
        with pytest.raises(
            ValueError, match="credentials_ref must be in the form 'namespace/name'"
        ):
            _parse_credentials_ref("invalid-format")

    def test_parse_credentials_ref_empty_namespace(self):
        """Test parsing credentials reference with empty namespace."""
        with pytest.raises(ValueError, match="credentials_ref components cannot be empty"):
            _parse_credentials_ref("/my-secret")

    def test_parse_credentials_ref_empty_name(self):
        """Test parsing credentials reference with empty name."""
        with pytest.raises(ValueError, match="credentials_ref components cannot be empty"):
            _parse_credentials_ref("default/")

    def test_parse_credentials_ref_invalid_characters(self):
        """Test parsing credentials reference with invalid characters."""
        with pytest.raises(ValueError, match="credentials_ref contains invalid characters"):
            _parse_credentials_ref("default/my_secret!")

    def test_parse_credentials_ref_colon_in_name(self):
        """Test parsing credentials reference with colon in name."""
        with pytest.raises(
            ValueError, match="credentials_ref must be in the form 'namespace/name'"
        ):
            _parse_credentials_ref("default/my:secret")


class TestBuildJobPath:
    """Test cases for build_job_path function."""

    def test_build_job_path_simple(self):
        """Test simple job path building."""
        path = build_job_path("my-job")
        assert path == "/job/my-job"

    def test_build_job_path_nested(self):
        """Test nested job path building."""
        path = build_job_path("folder/my-job")
        assert path == "/job/folder/job/my-job"

    def test_build_job_path_deeply_nested(self):
        """Test deeply nested job path building."""
        path = build_job_path("a/b/c/my-job")
        assert path == "/job/a/job/b/job/c/job/my-job"


@pytest.mark.asyncio
async def test_credentials_use_reliable_async_command_runner(mocker):
    encoded_user = base64.b64encode(b"jenkins-user").decode()
    encoded_token = base64.b64encode(b"token").decode()
    run_command = mocker.patch("tools.jenkins.run_command", new=mocker.AsyncMock())
    run_command.return_value = {
        "output": json.dumps(
            {"data": {"username": encoded_user, "api-token": encoded_token}}
        ),
        "error": False,
    }

    result = await resolve_credentials_from_k8s("ci/jenkins-secret")

    assert result == ("jenkins-user", "token")
    run_command.assert_awaited_once_with(
        "kubectl",
        ["-n", "ci", "get", "secret", "jenkins-secret", "-o", "json"],
        timeout_seconds=5.0,
        retry=True,
    )


@pytest.mark.asyncio
async def test_jenkins_get_retries_transient_status(mocker, monkeypatch):
    client = JenkinsClient("https://jenkins.example", "user", "token")
    first = httpx.Response(503, request=httpx.Request("GET", "https://jenkins.example/api"))
    second = httpx.Response(200, request=httpx.Request("GET", "https://jenkins.example/api"))
    request = mocker.patch.object(
        client.client,
        "request",
        new=mocker.AsyncMock(side_effect=[first, second]),
    )
    sleep = mocker.patch("asyncio.sleep", new=mocker.AsyncMock())
    monkeypatch.setattr("tools.jenkins.settings.COMMAND_MAX_RETRY_ATTEMPTS", 2)

    response = await client.get("/api")
    await client.close()

    assert response.status_code == 200
    assert request.await_count == 2
    sleep.assert_awaited_once()


@pytest.mark.asyncio
async def test_jenkins_post_is_never_retried(mocker, monkeypatch):
    client = JenkinsClient("https://jenkins.example", "user", "token")
    response = httpx.Response(
        503,
        request=httpx.Request("POST", "https://jenkins.example/job/demo/build"),
    )
    request = mocker.patch.object(
        client.client,
        "request",
        new=mocker.AsyncMock(return_value=response),
    )
    mocker.patch.object(client, "_crumb_headers", new=mocker.AsyncMock(return_value={}))
    monkeypatch.setattr("tools.jenkins.settings.COMMAND_MAX_RETRY_ATTEMPTS", 5)

    result = await client.post("/job/demo/build")
    await client.close()

    assert result.status_code == 503
    request.assert_awaited_once()
