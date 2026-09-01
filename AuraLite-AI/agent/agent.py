"""
agent/agent.py — AuraLite Agent: reasoning + tool-calling loop.

The agent works with ANY backend (GGUF, HF, native torch):
  1. Build a system prompt describing available tools
  2. Run the model in streaming or batch mode
  3. Parse <tool ...> blocks from the output
  4. Execute each tool in the Sandbox
  5. Append results to context and repeat until no more tool calls
     or max_iterations is reached

Works on Windows 10 / Linux / macOS.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

from .sandbox import Sandbox
from .tools import (
    TOOL_REGISTRY,
    ToolResult,
    build_system_prompt,
    dispatch_tool,
    parse_tool_calls,
)


# ── Agent step event ─────────────────────────────────────────────────────────

@dataclass
class AgentStep:
    """One turn in the agent's reasoning loop."""
    iteration: int
    model_output: str
    tool_calls: list[dict]
    tool_results: list[ToolResult]
    final: bool = False


# ── Agent ─────────────────────────────────────────────────────────────────────

class AuraLiteAgent:
    """
    Tool-calling agent powered by an AuraLiteEngine backend.

    Parameters
    ----------
    engine : AuraLiteEngine
        The loaded model engine (any backend).
    sandbox : Sandbox | None
        Sandbox instance to use.  If None, a new one is created.
    enabled_tools : list[str] | None
        Subset of TOOL_REGISTRY keys to expose.  None = all tools.
    max_iterations : int
        Max tool-call rounds before forcing a final answer.
    max_tokens_per_step : int
        Max new tokens per generation call.
    temperature, top_p, top_k, repetition_penalty : float/int
        Sampling parameters forwarded to the engine.
    on_step : callable | None
        Callback(AgentStep) called after each iteration (useful for streaming UI).
    chat_template : str
        Chat template name used when building prompts.
    """

    def __init__(
        self,
        engine,
        *,
        sandbox: Optional[Sandbox] = None,
        enabled_tools: Optional[list[str]] = None,
        max_iterations: int = 8,
        max_tokens_per_step: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        on_step: Optional[Callable[[AgentStep], None]] = None,
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
        """Signal the agent to stop after the current step."""
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

    # ── Main entry ───────────────────────────────────────────────────────────

    def run(
        self,
        user_message: str,
        history: Optional[list[dict]] = None,
        *,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Run the full agent loop.

        Parameters
        ----------
        user_message : str
            The user's request.
        history : list[dict] | None
            Prior conversation messages [{role, content}, ...].
        stream_callback : callable | None
            If provided, called with each token chunk as it's generated.

        Returns
        -------
        str
            The agent's final answer (last model output after all tool calls).
        """
        self.reset()
        messages = self._build_initial_messages(user_message, history)
        final_answer = ""

        for iteration in range(1, self.max_iterations + 1):
            if self._stop_event.is_set():
                break

            # ── Generate ────────────────────────────────────────────────
            model_output = self._generate(messages, stream_callback)
            final_answer = model_output

            # ── Parse tool calls ─────────────────────────────────────────
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

            # ── Execute tools ─────────────────────────────────────────────
            tool_result_parts = []
            for call in tool_calls:
                if self._stop_event.is_set():
                    break
                name = call["name"]
                args = call["args"]
                result: ToolResult = dispatch_tool(name, args, self.sandbox)
                step.tool_results.append(result)
                tool_result_parts.append(
                    f"[TOOL RESULT: {name}]\n{result.to_text()}\n[/TOOL RESULT]"
                )

            if self.on_step:
                self.on_step(step)

            if self._stop_event.is_set():
                break

            # ── Append model output + results to context ─────────────────
            messages.append({"role": "assistant", "content": model_output})
            messages.append({
                "role": "user",
                "content": "\n\n".join(tool_result_parts) + "\n\nPlease continue.",
            })

        return final_answer

    def run_streaming(
        self,
        user_message: str,
        history: Optional[list[dict]] = None,
    ) -> Iterator[str]:
        """
        Generator version of run() — yields text chunks in real time.

        Yields model tokens as they arrive, then yields tool results as
        formatted blocks, then continues to the next iteration.
        """
        self.reset()
        messages = self._build_initial_messages(user_message, history)

        for iteration in range(1, self.max_iterations + 1):
            if self._stop_event.is_set():
                return

            # ── Streaming generation ─────────────────────────────────────
            model_output_parts: list[str] = []

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

            # ── Parse tool calls ─────────────────────────────────────────
            tool_calls = parse_tool_calls(model_output)
            if not tool_calls:
                return  # Done

            # ── Execute tools ─────────────────────────────────────────────
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

    # ── Private helpers ──────────────────────────────────────────────────────

    def _build_initial_messages(
        self,
        user_message: str,
        history: Optional[list[dict]],
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": self._system_prompt}]
        if history:
            for msg in history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    if msg["role"] != "system":  # don't duplicate system
                        messages.append(msg)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _generate(
        self,
        messages: list[dict],
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Generate a full response (with optional streaming callback)."""
        engine = self.engine
        kwargs = dict(
            max_new_tokens=self.max_tokens_per_step,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
        )

        if stream_callback:
            parts: list[str] = []
            try:
                for chunk in engine.generate_chat_streaming(messages, **kwargs):
                    parts.append(chunk)
                    stream_callback(chunk)
            except Exception:
                # Fallback to batch
                result = self._batch_generate(messages)
                stream_callback(result)
                return result
            return "".join(parts)
        else:
            return self._batch_generate(messages)

    def _batch_generate(self, messages: list[dict]) -> str:
        engine = self.engine
        kwargs = dict(
            max_new_tokens=self.max_tokens_per_step,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
        )
        try:
            return engine.generate_chat(messages, **kwargs)
        except Exception:
            # Last resort: flatten to a string prompt
            prompt = self._flatten_messages(messages)
            return engine.generate(prompt, self.max_tokens_per_step,
                                   self.temperature, self.top_k, self.top_p,
                                   self.repetition_penalty)

    def _generate_streaming(self, messages: list[dict]) -> Iterator[str]:
        engine = self.engine
        kwargs = dict(
            max_new_tokens=self.max_tokens_per_step,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
        )
        try:
            yield from engine.generate_chat_streaming(messages, **kwargs)
        except Exception:
            result = self._batch_generate(messages)
            yield result

    @staticmethod
    def _flatten_messages(messages: list[dict]) -> str:
        lines = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            lines.append(f"{role.upper()}: {content}")
        lines.append("ASSISTANT:")
        return "\n\n".join(lines)
