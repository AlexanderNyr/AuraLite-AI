"""
Chat / Instruction Interface for AuraLite AI (v2.3+)

Provides:
- Structured chat message handling (system / user / assistant)
- Multiple chat templates (ChatML, Llama-2, Mistral, Gemma, Phi, etc.)
- History management
- Integration with native, GGUF, and HF backends
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Callable
import json


Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    """Single chat message."""
    role: Role
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, d: dict) -> "ChatMessage":
        return cls(role=d["role"], content=d["content"])


@dataclass
class ChatHistory:
    """Conversation history."""
    messages: List[ChatMessage] = field(default_factory=list)
    max_turns: int = 20  # limit to prevent context overflow

    def add(self, role: Role, content: str):
        self.messages.append(ChatMessage(role=role, content=content))
        # Trim old messages if needed (keep system + last N turns)
        if len(self.messages) > self.max_turns * 2 + 1:
            # Keep the first system message if present
            system_msgs = [m for m in self.messages if m.role == "system"]
            other_msgs = [m for m in self.messages if m.role != "system"][-self.max_turns * 2:]
            self.messages = system_msgs + other_msgs

    def clear(self):
        self.messages.clear()

    def to_list(self) -> List[Dict[str, str]]:
        return [m.to_dict() for m in self.messages]

    @classmethod
    def from_list(cls, data: List[Dict]) -> "ChatHistory":
        return cls(messages=[ChatMessage.from_dict(m) for m in data])


# ======================================================================
# Chat Templates
# ======================================================================

CHAT_TEMPLATES = {
    "chatml": {
        "name": "ChatML (Qwen, Yi, etc.)",
        "system": "<|im_start|>system\n{system}<|im_end|>\n",
        "user": "<|im_start|>user\n{user}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{assistant}<|im_end|>\n",
        "stop": ["<|im_end|>", "<|endoftext|>"],
    },
    "llama2": {
        "name": "Llama-2 Chat",
        "system": "<<SYS>>\n{system}\n<</SYS>>\n\n",
        "user": "[INST] {user} [/INST]",
        "assistant": " {assistant}",
        "stop": ["</s>"],
    },
    "mistral": {
        "name": "Mistral / Mixtral",
        "system": "",  # Mistral usually doesn't use system
        "user": "[INST] {user} [/INST]",
        "assistant": " {assistant}",
        "stop": ["</s>"],
    },
    "gemma": {
        "name": "Gemma (Google)",
        "system": "<start_of_turn>system\n{system}<end_of_turn>\n",
        "user": "<start_of_turn>user\n{user}<end_of_turn>\n",
        "assistant": "<start_of_turn>model\n{assistant}<end_of_turn>\n",
        "stop": ["<end_of_turn>"],
    },
    "phi": {
        "name": "Phi-3 / Phi-4",
        "system": "<|system|>\n{system}<|end|>\n",
        "user": "<|user|>\n{user}<|end|>\n",
        "assistant": "<|assistant|>\n{assistant}<|end|>\n",
        "stop": ["<|end|>"],
    },
    "simple": {
        "name": "Simple (no special tokens)",
        "system": "System: {system}\n",
        "user": "User: {user}\n",
        "assistant": "Assistant: {assistant}\n",
        "stop": [],
    },
}


def apply_chat_template(
    messages: List[ChatMessage] | ChatHistory,
    template_name: str = "chatml",
    add_generation_prompt: bool = True,
) -> str:
    """
    Convert a list of messages into a single prompt string using the chosen template.

    Args:
        messages: List of ChatMessage or ChatHistory
        template_name: one of CHAT_TEMPLATES keys
        add_generation_prompt: whether to add the start of assistant response

    Returns:
        Formatted prompt string ready for generation.
    """
    if isinstance(messages, ChatHistory):
        messages = messages.messages

    if template_name not in CHAT_TEMPLATES:
        template_name = "simple"

    tpl = CHAT_TEMPLATES[template_name]

    prompt_parts = []

    for msg in messages:
        if msg.role == "system":
            if tpl["system"]:
                prompt_parts.append(tpl["system"].format(system=msg.content))
        elif msg.role == "user":
            prompt_parts.append(tpl["user"].format(user=msg.content))
        elif msg.role == "assistant":
            prompt_parts.append(tpl["assistant"].format(assistant=msg.content))

    if add_generation_prompt:
        # Add the beginning of the assistant response
        if template_name == "chatml":
            prompt_parts.append("<|im_start|>assistant\n")
        elif template_name == "gemma":
            prompt_parts.append("<start_of_turn>model\n")
        elif template_name == "phi":
            prompt_parts.append("<|assistant|>\n")
        elif template_name in ("llama2", "mistral"):
            # Llama-2 / Mistral style already ends with [/INST]
            pass
        else:
            prompt_parts.append("Assistant: ")

    return "".join(prompt_parts)


def get_stop_tokens(template_name: str) -> List[str]:
    """Return stop tokens for the given template."""
    return CHAT_TEMPLATES.get(template_name, {}).get("stop", [])


# ======================================================================
# High-level Chat Engine helpers
# ======================================================================

def build_chat_prompt(
    history: ChatHistory,
    template: str = "chatml",
    system_prompt: Optional[str] = None,
) -> str:
    """
    Convenience function to build a chat prompt.

    If system_prompt is provided and there is no system message yet,
    it will be prepended.
    """
    if system_prompt and not any(m.role == "system" for m in history.messages):
        history.messages.insert(0, ChatMessage(role="system", content=system_prompt))

    return apply_chat_template(history, template_name=template, add_generation_prompt=True)