"""Tests for the AuraLite Agent framework."""
from __future__ import annotations

import sys, os, threading
from unittest.mock import patch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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

    def test_run_python_timeout(self):
        from agent.sandbox import Sandbox
        with Sandbox(timeout=2) as sb:
            r = sb.run_python("import time; time.sleep(60)")
            assert r.timed_out

    def test_write_and_read_file(self):
        from agent.sandbox import Sandbox
        with Sandbox(timeout=10) as sb:
            sb.write_file("test.txt", "hello world")
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
            assert "a.txt" in r.stdout and "b.py" in r.stdout

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
            assert "hello_sandbox" in r.stdout or r.timed_out

    def test_output_cap(self):
        from agent.sandbox import Sandbox
        with Sandbox(timeout=10, max_output_bytes=500) as sb:
            r = sb.run_python("print('x' * 10000)")
            assert len(r.stdout.encode()) <= 600

    def test_combined_output_blocked(self):
        from agent.sandbox import Sandbox
        with Sandbox(timeout=10) as sb:
            r = sb.run_shell("sudo rm -rf /")
            assert "blocked" in r.combined_output().lower()


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
            assert not r.success and "Unknown tool" in r.error

    def test_web_search_tool_mock(self):
        from agent.sandbox import Sandbox
        from agent.tools import dispatch_tool
        with Sandbox(timeout=5) as sb:
            with patch("web_tools.build_web_context", return_value="[1] Test: snippet"):
                r = dispatch_tool("web_search", {"query": "test query"}, sb)
                assert r.success and "snippet" in r.output


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
        text = 'First <tool name="calculate">2+2</tool> then <tool name="python">print(1)</tool>'
        calls = parse_tool_calls(text)
        assert len(calls) == 2

    def test_parse_with_attr(self):
        from agent.tools import parse_tool_calls
        text = '<tool name="write_file" filename="out.txt">content here</tool>'
        calls = parse_tool_calls(text)
        assert calls[0]["args"]["filename"] == "out.txt"
        assert calls[0]["args"]["content"] == "content here"

    def test_parse_no_calls(self):
        from agent.tools import parse_tool_calls
        assert parse_tool_calls("Just a plain response.") == []

    def test_parse_unclosed_tag_ignored(self):
        from agent.tools import parse_tool_calls
        assert parse_tool_calls('<tool name="python">print("unclosed")') == []

    def test_system_prompt_contains_tool_names(self):
        from agent.tools import build_system_prompt, TOOL_REGISTRY
        prompt = build_system_prompt()
        for name in TOOL_REGISTRY:
            assert name in prompt


class MockEngine:
    def __init__(self, responses):
        self._responses = list(responses)
        self._call_count = 0

    def generate_chat(self, messages, **kwargs):
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

    def generate(self, prompt, *args, **kwargs):
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
        responses = ['<tool name="python">print("from agent")</tool>', "I executed the code."]
        engine = MockEngine(responses)
        with Sandbox(timeout=10) as sb:
            agent = AuraLiteAgent(engine, sandbox=sb, max_iterations=5)
            result = agent.run("Run some Python code")
        assert "executed" in result or "I" in result

    def test_agent_stop(self):
        from agent.agent import AuraLiteAgent
        from agent.sandbox import Sandbox
        class SlowEngine:
            def generate_chat(self, messages, **kwargs): return "plain response"
            def generate_chat_streaming(self, messages, **kwargs):
                import time
                for i in range(100):
                    time.sleep(0.05)
                    yield f"token{i}"
        with Sandbox(timeout=5) as sb:
            agent = AuraLiteAgent(SlowEngine(), sandbox=sb, max_iterations=10)
            agent.stop()
            result = agent.run("anything")
        assert isinstance(result, str)

    def test_agent_streaming(self):
        from agent.agent import AuraLiteAgent
        from agent.sandbox import Sandbox
        engine = MockEngine(["Quick answer without tools."])
        with Sandbox(timeout=10) as sb:
            agent = AuraLiteAgent(engine, sandbox=sb, max_iterations=3)
            chunks = list(agent.run_streaming("Tell me something"))
        assert len(chunks) > 0 and "".join(chunks)

    def test_agent_max_iterations(self):
        from agent.agent import AuraLiteAgent
        from agent.sandbox import Sandbox
        engine = MockEngine(['<tool name="calculate">1+1</tool>'] * 20)
        with Sandbox(timeout=10) as sb:
            agent = AuraLiteAgent(engine, sandbox=sb, max_iterations=3)
            agent.run("Keep calling tools")
        assert engine._call_count <= 4

    def test_agent_on_step_callback(self):
        from agent.agent import AuraLiteAgent
        from agent.sandbox import Sandbox
        steps = []
        engine = MockEngine(['<tool name="python">print(42)</tool>', "Done."])
        with Sandbox(timeout=10) as sb:
            agent = AuraLiteAgent(engine, sandbox=sb, max_iterations=5,
                                  on_step=lambda s: steps.append(s))
            agent.run("Run code")
        assert len(steps) >= 1
        assert any(len(s.tool_calls) > 0 for s in steps)
