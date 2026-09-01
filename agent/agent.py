"""
agent/agent.py — AuraLite Agent: reasoning + tool-calling loop.

Works with ANY backend (GGUF, HF, native torch).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

from .sandbox import Sandbox
from .tools import TOOL_REGISTRY, ToolResult, build_system_prompt, dispatch_tool, parse_tool_calls


@dataclass
class AgentStep:
    """One turn in the agent's reasoning loop."""
    iteration: int
    model_output: str
    tool_calls: list
    tool_results: list
    final: bool = False


class AuraLiteAgent:
    """Tool-calling agent powered by an AuraLiteEngine backend."""

    def __init__(
        self,
        engine,
        *,
        sandbox: Optional[Sandbox] = None,
        enabled_tools: Optional[list] = None,
        max_iterations: int = 8,
        max_tokens_per_step: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        on_step: Optional[Callable] = None,
        chat_template: str = "chatml",
        timeout_per_tool: float = 15.0,
    ):
        self.engine = engine
        self.sandbox = sandbox or Sandbox(timeout=timeout_per_tool)
        self._owns_sandbox = sandbox is None
        self.enabled_tools = enabled_tools or list(TOOL_REGISTRY.keys())
        self.max_iterations = max_iterations
        self.max_tokens_per_step = max_tokens_per_step
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.on_step = on_step
        self.chat_template = chat_template
        self.timeout_per_tool = timeout_per_tool
        self._system_prompt = build_system_prompt(self.enabled_tools)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def reset(self):
        self._stop_event.clear()

    def cleanup(self):
        if self._owns_sandbox:
            self.sandbox.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cleanup()

    def run(self, user_message: str, history=None, *, stream_callback=None) -> str:
        self.reset()
        messages = self._build_initial_messages(user_message, history)
        final_answer = ""

        for iteration in range(1, self.max_iterations + 1):
            if self._stop_event.is_set():
                break

            model_output = self._generate(messages, stream_callback)
            final_answer = model_output
            tool_calls = parse_tool_calls(model_output)

            step = AgentStep(
                iteration=iteration,
                model_output=model_output,
                tool_calls=tool_calls,
                tool_results=[],
                final=(not tool_calls),
            )

            if not tool_calls:
                if self.on_step:
                    self.on_step(step)
                break

            tool_result_parts = []
            for call in tool_calls:
                if self._stop_event.is_set():
                    break
                name = call["name"]
                args = call["args"]
                result: ToolResult = dispatch_tool(name, args, self.sandbox)
                step.tool_results.append(result)
                tool_result_parts.append(f"[TOOL RESULT: {name}]\n{result.to_text()}\n[/TOOL RESULT]")

            if self.on_step:
                self.on_step(step)

            if self._stop_event.is_set():
                break

            messages.append({"role": "assistant", "content": model_output})
            messages.append({
                "role": "user",
                "content": "\n\n".join(tool_result_parts) + "\n\nPlease continue.",
            })

        return final_answer

    def run_streaming(self, user_message: str, history=None) -> Iterator[str]:
        self.reset()
        messages = self._build_initial_messages(user_message, history)

        for iteration in range(1, self.max_iterations + 1):
            if self._stop_event.is_set():
                return

            model_output_parts: list = []
            try:
                for chunk in self._generate_streaming(messages):
                    if self._stop_event.is_set():
                        return
                    model_output_parts.append(chunk)
                    yield chunk
            except Exception as e:
                yield f"\n[Agent generation error: {e}]\n"
                return

            model_output = "".join(model_output_parts)
            tool_calls = parse_tool_calls(model_output)

            if not tool_calls:
                return

            messages.append({"role": "assistant", "content": model_output})
            result_parts = []

            for call in tool_calls:
                if self._stop_event.is_set():
                    return
                name = call["name"]
                args = call["args"]

                yield f"\n\n[TOOL RESULT: {name}]\n"
                result: ToolResult = dispatch_tool(name, args, self.sandbox)
                out = result.to_text()
                yield out
                yield "\n[/TOOL RESULT]\n"
                result_parts.append(f"[TOOL RESULT: {name}]\n{out}\n[/TOOL RESULT]")

                if self.on_step:
                    self.on_step(AgentStep(
                        iteration=iteration,
                        model_output=model_output,
                        tool_calls=tool_calls,
                        tool_results=[result],
                    ))

            messages.append({
                "role": "user",
                "content": "\n\n".join(result_parts) + "\n\nPlease continue.",
            })

    def _build_initial_messages(self, user_message: str, history) -> list:
        messages: list = [{"role": "system", "content": self._system_prompt}]
        if history:
            for msg in history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    if msg["role"] != "system":
                        messages.append(msg)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _generate(self, messages, stream_callback=None) -> str:
        kwargs = dict(
            max_new_tokens=self.max_tokens_per_step,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
        )
        if stream_callback:
            parts: list = []
            try:
                for chunk in self.engine.generate_chat_streaming(messages, **kwargs):
                    parts.append(chunk)
                    stream_callback(chunk)
            except Exception:
                result = self._batch_generate(messages)
                stream_callback(result)
                return result
            return "".join(parts)
        else:
            return self._batch_generate(messages)

    def _batch_generate(self, messages) -> str:
        kwargs = dict(
            max_new_tokens=self.max_tokens_per_step,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
        )
        try:
            return self.engine.generate_chat(messages, **kwargs)
        except Exception:
            prompt = self._flatten_messages(messages)
            return self.engine.generate(prompt, self.max_tokens_per_step,
                                        self.temperature, self.top_k, self.top_p,
                                        self.repetition_penalty)

    def _generate_streaming(self, messages) -> Iterator[str]:
        kwargs = dict(
            max_new_tokens=self.max_tokens_per_step,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
        )
        try:
            yield from self.engine.generate_chat_streaming(messages, **kwargs)
        except Exception:
            result = self._batch_generate(messages)
            yield result

    @staticmethod
    def _flatten_messages(messages) -> str:
        lines = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            lines.append(f"{role.upper()}: {content}")
        lines.append("ASSISTANT:")
        return "\n\n".join(lines)
