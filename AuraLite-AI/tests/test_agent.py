"""Tests for the AuraLite Agent framework (sandbox + tools + agent loop)."""
from __future__ import annotations

import sys
import os
import threading
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Sandbox tests ─────────────────────────────────────────────────────────────

class TestSandbox:
    def test_run_python_basic(self):
        from agent.sandbox import Sandbox
        with Sandbox(timeout=10) as sb:
            r = sb.run_python("print('hello')")
            assert r.success
            assert "hello" in r.stdout

    def test_run_python_syntax_error(self):
        from agent.sandbox import Sandbox
        with Sandbox(timeout=10) as sb:
            r = sb.run_python("def bad(:\n    pass")
            assert not r.success
            assert r.returncode != 0

    def test_run_python_timeout(self):
        from agent.sandbox import Sandbox
        with Sandbox(timeout=2) as sb:
            r = sb.run_python("import time; time.sleep(60)")
            assert r.timed_out

    def test_write_and_read_file(self):
        from agent.sandbox import Sandbox
        with Sandbox(timeout=10) as sb:
            r = sb.write_file("test.txt", "hello world")
            assert r.success
            r2 = sb.read_file("test.txt")
            assert "hello world" in r2.stdout

    def test_path_escape_blocked(self):
        from agent.sandbox import Sandbox
        with Sandbox(timeout=10) as sb:
            with pytest.raises(ValueError):
                sb._safe_path("../../etc/passwd")

    def test_list_files(self):
        from agent.sandbox import Sandbox
        with Sandbox(timeout=10) as sb:
            sb.write_file("a.txt", "x")
            sb.write_file("b.py", "y")
            r = sb.list_files(".")
            assert r.success
            assert "a.txt" in r.stdout
            assert "b.py" in r.stdout

    def test_shell_blocked_hard(self):
        from agent.sandbox import Sandbox
        with Sandbox(timeout=10) as sb:
            r = sb.run_shell("rm -rf /")
            assert r.blocked

    def test_shell_blocked_whitelist(self):
        from agent.sandbox import Sandbox
        with Sandbox(timeout=10) as sb:
            r = sb.run_shell("evil_custom_binary --hack")
            assert r.blocked

    def test_shell_echo_allowed(self):
        from agent.sandbox import Sandbox
        with Sandbox(timeout=10) as sb:
            r = sb.run_shell("echo hello_sandbox")
            assert "hello_sandbox" in r.stdout or r.timed_out  # timed_out=edge case on CI

    def test_output_cap(self):
        from agent.sandbox import Sandbox
        with Sandbox(timeout=10, max_output_bytes=500) as sb:
            r = sb.run_python("print('x' * 10000)")
            assert len(r.stdout.encode()) <= 600  # some slack for truncation msg

    def test_combined_output(self):
        from agent.sandbox import Sandbox
        with Sandbox(timeout=10) as sb:
            r = sb.run_python("print('out')")
            text = r.combined_output()
            assert "out" in text

    def test_combined_output_blocked(self):
        from agent.sandbox import Sandbox
        with Sandbox(timeout=10) as sb:
            r = sb.run_shell("sudo rm -rf /")
            assert "blocked" in r.combined_output().lower()


# ── Tool tests ────────────────────────────────────────────────────────────────

class TestTools:
    def test_python_tool(self):
        from agent.sandbox import Sandbox
        from agent.tools import dispatch_tool
        with Sandbox(timeout=10) as sb:
            result = dispatch_tool("python", {"code": "print(2+2)"}, sb)
            assert result.success
            assert "4" in result.output

    def test_write_file_tool(self):
        from agent.sandbox import Sandbox
        from agent.tools import dispatch_tool
        with Sandbox(timeout=10) as sb:
            r = dispatch_tool("write_file", {"filename": "test.txt", "content": "hello"}, sb)
            assert r.success

    def test_read_file_tool(self):
        from agent.sandbox import Sandbox
        from agent.tools import dispatch_tool
        with Sandbox(timeout=10) as sb:
            sb.write_file("r.txt", "content here")
            r = dispatch_tool("read_file", {"filename": "r.txt"}, sb)
            assert "content here" in r.output

    def test_list_files_tool(self):
        from agent.sandbox import Sandbox
        from agent.tools import dispatch_tool
        with Sandbox(timeout=10) as sb:
            sb.write_file("listed.txt", "x")
            r = dispatch_tool("list_files", {"directory": "."}, sb)
            assert "listed.txt" in r.output

    def test_calculate_tool(self):
        from agent.sandbox import Sandbox
        from agent.tools import dispatch_tool
        with Sandbox(timeout=10) as sb:
            r = dispatch_tool("calculate", {"expression": "2 ** 10"}, sb)
            assert "1024" in r.output

    def test_unknown_tool(self):
        from agent.sandbox import Sandbox
        from agent.tools import dispatch_tool
        with Sandbox(timeout=5) as sb:
            r = dispatch_tool("nonexistent_tool", {}, sb)
            assert not r.success
            assert "Unknown tool" in r.error

    def test_web_search_tool_mock(self):
        from agent.sandbox import Sandbox
        from agent.tools import dispatch_tool
        with Sandbox(timeout=5) as sb:
            with patch("web_tools.build_web_context", return_value="[1] Test: snippet"):
                r = dispatch_tool("web_search", {"query": "test query"}, sb)
                assert r.success
                assert "snippet" in r.output


# ── XML parser tests ──────────────────────────────────────────────────────────

class TestToolParser:
    def test_parse_simple(self):
        from agent.tools import parse_tool_calls
        text = '<tool name="python">print("hi")</tool>'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "python"
        assert calls[0]["args"]["code"] == 'print("hi")'

    def test_parse_multiple(self):
        from agent.tools import parse_tool_calls
        text = (
            'First <tool name="calculate">2+2</tool> '
            'then <tool name="python">print(1)</tool>'
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 2
        assert calls[0]["name"] == "calculate"
        assert calls[1]["name"] == "python"

    def test_parse_with_attr(self):
        from agent.tools import parse_tool_calls
        text = '<tool name="write_file" filename="out.txt">content here</tool>'
        calls = parse_tool_calls(text)
        assert calls[0]["args"]["filename"] == "out.txt"
        assert calls[0]["args"]["content"] == "content here"

    def test_parse_no_calls(self):
        from agent.tools import parse_tool_calls
        calls = parse_tool_calls("Just a plain response with no tools.")
        assert calls == []

    def test_parse_unclosed_tag_ignored(self):
        from agent.tools import parse_tool_calls
        text = '<tool name="python">print("unclosed")'
        calls = parse_tool_calls(text)
        assert calls == []

    def test_system_prompt_contains_tool_names(self):
        from agent.tools import build_system_prompt, TOOL_REGISTRY
        prompt = build_system_prompt()
        for name in TOOL_REGISTRY:
            assert name in prompt


# ── Agent integration tests ───────────────────────────────────────────────────

class MockEngine:
    """Minimal mock that mimics AuraLiteEngine for agent testing."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._call_count = 0

    def generate_chat(self, messages, **kwargs) -> str:
        if self._call_count < len(self._responses):
            r = self._responses[self._call_count]
        else:
            r = "I'm done."
        self._call_count += 1
        return r

    def generate_chat_streaming(self, messages, **kwargs):
        text = self.generate_chat(messages, **kwargs)
        for char in text:
            yield char

    def generate(self, prompt, *args, **kwargs) -> str:
        return self.generate_chat([])


class TestAgent:
    def test_agent_no_tools(self):
        from agent.agent import AuraLiteAgent
        from agent.sandbox import Sandbox
        engine = MockEngine(["Hello! No tools needed here."])
        with Sandbox(timeout=10) as sb:
            agent = AuraLiteAgent(engine, sandbox=sb, max_iterations=3)
            result = agent.run("Say hello")
            assert "Hello" in result

    def test_agent_with_tool_call(self):
        from agent.agent import AuraLiteAgent
        from agent.sandbox import Sandbox
        responses = [
            '<tool name="python">print("from agent")</tool>',
            "I executed the code successfully.",
        ]
        engine = MockEngine(responses)
        with Sandbox(timeout=10) as sb:
            agent = AuraLiteAgent(engine, sandbox=sb, max_iterations=5)
            result = agent.run("Run some Python code")
        # Result should be the final response (iteration 2)
        assert "executed" in result or "agent" in result

    def test_agent_stop(self):
        from agent.agent import AuraLiteAgent
        from agent.sandbox import Sandbox

        stop_event = threading.Event()

        class SlowEngine:
            def generate_chat(self, messages, **kwargs):
                return "plain response"
            def generate_chat_streaming(self, messages, **kwargs):
                import time
                for i in range(100):
                    time.sleep(0.01)
                    yield f"token{i}"

        with Sandbox(timeout=5) as sb:
            agent = AuraLiteAgent(SlowEngine(), sandbox=sb, max_iterations=10)
            # Stop immediately
            agent.stop()
            result = agent.run("anything")
        # Should return empty or partial (no crash)
        assert isinstance(result, str)

    def test_agent_streaming(self):
        from agent.agent import AuraLiteAgent
        from agent.sandbox import Sandbox
        engine = MockEngine(["Quick answer without tools."])
        with Sandbox(timeout=10) as sb:
            agent = AuraLiteAgent(engine, sandbox=sb, max_iterations=3)
            chunks = list(agent.run_streaming("Tell me something"))
        assert len(chunks) > 0
        assert "".join(chunks)  # non-empty

    def test_agent_max_iterations(self):
        """Agent should stop after max_iterations even if model keeps calling tools."""
        from agent.agent import AuraLiteAgent
        from agent.sandbox import Sandbox
        # Always returns a tool call — agent must stop at max_iterations
        engine = MockEngine(['<tool name="calculate">1+1</tool>'] * 20)
        with Sandbox(timeout=10) as sb:
            agent = AuraLiteAgent(engine, sandbox=sb, max_iterations=3)
            result = agent.run("Keep calling tools")
        # Should not run forever
        assert engine._call_count <= 4  # max_iterations + 1 at most

    def test_agent_on_step_callback(self):
        from agent.agent import AuraLiteAgent, AgentStep
        from agent.sandbox import Sandbox
        steps = []
        engine = MockEngine([
            '<tool name="python">print(42)</tool>',
            "Done.",
        ])
        with Sandbox(timeout=10) as sb:
            agent = AuraLiteAgent(engine, sandbox=sb, max_iterations=5,
                                  on_step=lambda s: steps.append(s))
            agent.run("Run code")
        assert len(steps) >= 1
        # First step should have a tool call
        assert any(len(s.tool_calls) > 0 for s in steps)
