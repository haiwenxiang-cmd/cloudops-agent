"""Tests for utils.commands module."""

import asyncio

import pytest

from utils.commands import cancel_run_processes, run_command


class TestRunCommand:
    """Test cases for run_command function."""

    @pytest.mark.asyncio
    async def test_run_command_success(self, mocker):
        """Test successful command execution."""
        mock_subprocess = mocker.patch("asyncio.create_subprocess_exec")
        mock_proc = mocker.AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = mocker.AsyncMock(return_value=(b"success output", b""))
        mock_subprocess.return_value = mock_proc

        result = await run_command("test_cmd", ["arg1", "arg2"])

        assert result["output"] == "success output"
        assert result["error"] is False
        mock_subprocess.assert_called_once_with(
            "test_cmd",
            "arg1",
            "arg2",
            stdout=-1,
            stderr=-1,
            stdin=None,
            start_new_session=True,
        )

    @pytest.mark.asyncio
    async def test_run_command_with_stdin(self, mocker):
        """Test command execution with stdin input."""
        mock_subprocess = mocker.patch("asyncio.create_subprocess_exec")
        mock_proc = mocker.AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = mocker.AsyncMock(return_value=(b"output with stdin", b""))
        mock_subprocess.return_value = mock_proc

        result = await run_command("test_cmd", ["arg1"], stdin="test input")

        assert result["output"] == "output with stdin"
        assert result["error"] is False
        mock_proc.communicate.assert_called_once_with(input=b"test input")

    @pytest.mark.asyncio
    async def test_run_command_error_return_code(self, mocker):
        """Test command execution with error return code."""
        mock_subprocess = mocker.patch("asyncio.create_subprocess_exec")
        mock_proc = mocker.AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = mocker.AsyncMock(return_value=(b"", b"error message"))
        mock_subprocess.return_value = mock_proc

        result = await run_command("test_cmd", ["arg1"])

        assert "Error executing command" in result["output"]
        assert result["error"] is True

    @pytest.mark.asyncio
    async def test_run_command_stderr_output(self, mocker):
        """Warnings on stderr do not turn a zero exit code into a failure."""
        mock_subprocess = mocker.patch("asyncio.create_subprocess_exec")
        mock_proc = mocker.AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = mocker.AsyncMock(return_value=(b"", b"warning message"))
        mock_subprocess.return_value = mock_proc

        result = await run_command("test_cmd", ["arg1"])

        assert result["output"] == "warning message"
        assert result["error"] is False

    @pytest.mark.asyncio
    async def test_run_command_no_output(self, mocker):
        """Test command execution with no output."""
        mock_subprocess = mocker.patch("asyncio.create_subprocess_exec")
        mock_proc = mocker.AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = mocker.AsyncMock(return_value=(b"", b""))
        mock_subprocess.return_value = mock_proc

        result = await run_command("test_cmd", ["arg1"])

        assert "successfully" in result["output"]
        assert result["error"] is False

    @pytest.mark.asyncio
    async def test_run_command_exception(self, mocker):
        """Test command execution with exception."""
        mock_subprocess = mocker.patch("asyncio.create_subprocess_exec")
        mock_subprocess.side_effect = Exception("Process creation failed")

        result = await run_command("test_cmd", ["arg1"])

        assert "Error executing command" in result["output"]
        assert "Process creation failed" in result["output"]
        assert result["error"] is True
        assert result["error_type"] == "command_failed"

    @pytest.mark.asyncio
    async def test_timeout_terminates_and_retries_read_only_command(self, mocker, monkeypatch):
        """A transient timeout retries only an inferred read-only command."""
        mock_subprocess = mocker.patch("asyncio.create_subprocess_exec")
        first = mocker.AsyncMock()
        first.returncode = None
        first.pid = None
        first.communicate.side_effect = asyncio.TimeoutError
        first.wait = mocker.AsyncMock(return_value=0)
        first.terminate = mocker.Mock()
        first.kill = mocker.Mock()
        second = mocker.AsyncMock()
        second.returncode = 0
        second.communicate = mocker.AsyncMock(return_value=(b"pods", b""))
        mock_subprocess.side_effect = [first, second]
        sleep = mocker.patch("asyncio.sleep", new=mocker.AsyncMock())
        monkeypatch.setattr("utils.commands.settings.COMMAND_MAX_RETRY_ATTEMPTS", 2)

        result = await run_command("kubectl", ["get", "pods"], timeout_seconds=0.01)

        assert result["error"] is False
        assert result["output"] == "pods"
        assert result["attempts"] == 2
        first.terminate.assert_called_once_with()
        sleep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_mutation_is_not_retried(self, mocker, monkeypatch):
        """Mutation commands fail once to avoid duplicate side effects."""
        mock_subprocess = mocker.patch("asyncio.create_subprocess_exec")
        proc = mocker.AsyncMock()
        proc.returncode = 1
        proc.communicate = mocker.AsyncMock(return_value=(b"", b"connection refused"))
        mock_subprocess.return_value = proc
        monkeypatch.setattr("utils.commands.settings.COMMAND_MAX_RETRY_ATTEMPTS", 5)

        result = await run_command("kubectl", ["apply", "-f", "-"])

        assert result["error"] is True
        assert result["attempts"] == 1
        assert result["error_type"] == "transient"
        mock_subprocess.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancelled_task_terminates_child_and_propagates(self, mocker):
        """Cancelling the MCP handler cannot orphan its child process."""
        mocker.patch("asyncio.create_subprocess_exec")
        proc = mocker.AsyncMock()
        proc.returncode = None
        proc.pid = None
        proc.communicate.side_effect = asyncio.CancelledError
        proc.wait = mocker.AsyncMock(return_value=0)
        proc.terminate = mocker.Mock()
        proc.kill = mocker.Mock()
        asyncio.create_subprocess_exec.return_value = proc

        with pytest.raises(asyncio.CancelledError):
            await run_command("kubectl", ["get", "pods"])

        proc.terminate.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_safe_fallback_is_reported(self, mocker, monkeypatch):
        """A read-only primary command may return explicitly configured fallback evidence."""
        mock_subprocess = mocker.patch("asyncio.create_subprocess_exec")
        primary = mocker.AsyncMock()
        primary.returncode = 1
        primary.communicate = mocker.AsyncMock(return_value=(b"", b"unknown command"))
        fallback = mocker.AsyncMock()
        fallback.returncode = 0
        fallback.communicate = mocker.AsyncMock(return_value=(b"fallback state", b""))
        mock_subprocess.side_effect = [primary, fallback]
        monkeypatch.setattr("utils.commands.settings.COMMAND_MAX_RETRY_ATTEMPTS", 1)

        result = await run_command(
            "kubectl",
            ["argo", "rollouts", "status", "demo"],
            fallback=("kubectl", ["get", "rollout", "demo", "-o", "yaml"]),
        )

        assert result["error"] is False
        assert result["fallback_used"] is True
        assert "fallback state" in result["output"]

    @pytest.mark.asyncio
    async def test_cancel_run_processes_kills_registered_process(self, mocker):
        """The internal cancel route can terminate a child by workflow run id."""
        request = mocker.Mock()
        request.headers = {"x-skyflo-run-id": "run-123"}
        mocker.patch("utils.commands.get_http_request", return_value=request)
        created = asyncio.Event()
        release = asyncio.Event()
        proc = mocker.AsyncMock()
        proc.returncode = None
        proc.pid = None

        async def communicate(input=None):
            created.set()
            await release.wait()
            return b"", b""

        async def wait():
            proc.returncode = -15
            release.set()
            return -15

        proc.communicate = communicate
        proc.wait = wait
        proc.terminate = mocker.Mock()
        proc.kill = mocker.Mock()
        mocker.patch("asyncio.create_subprocess_exec", return_value=proc)

        task = asyncio.create_task(run_command("kubectl", ["get", "pods"]))
        await created.wait()
        result = await cancel_run_processes("run-123")
        await task

        assert result == {"matched": 1, "terminated": 1, "killed": 0, "cancelled": 1}
        proc.terminate.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_cancel_tombstone_rejects_late_process_start(self, mocker):
        """A child created after Stop cannot escape the run cancellation barrier."""
        request = mocker.Mock()
        request.headers = {"x-skyflo-run-id": "run-late"}
        mocker.patch("utils.commands.get_http_request", return_value=request)
        proc = mocker.AsyncMock()
        proc.returncode = None
        proc.pid = None

        async def wait():
            proc.returncode = -15
            return -15

        proc.wait = wait
        proc.terminate = mocker.Mock()
        proc.kill = mocker.Mock()
        mocker.patch("asyncio.create_subprocess_exec", return_value=proc)

        cancelled = await cancel_run_processes("run-late")
        result = await run_command("kubectl", ["apply", "-f", "-"])

        assert cancelled["cancelled"] == 1
        assert result["error"] is True
        assert result["error_type"] == "cancelled"
        assert result["ambiguous_outcome"] is True
        assert result["external_execution_started"] is True
        proc.terminate.assert_called_once_with()
