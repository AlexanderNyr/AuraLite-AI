"""Fast unit tests for chat_interface.py."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat_interface import (
    CHAT_TEMPLATES,
    ChatHistory,
    ChatMessage,
    apply_chat_template,
    build_chat_prompt,
    get_stop_tokens,
)


class TestChatMessage:
    def test_to_dict_roundtrip(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.to_dict() == {"role": "user", "content": "Hello"}
        assert ChatMessage.from_dict(msg.to_dict()) == msg

    def test_from_dict_keeps_extra_data_out(self):
        msg = ChatMessage.from_dict({"role": "assistant", "content": "Hi", "id": 123})
        assert msg.role == "assistant"
        assert msg.content == "Hi"
        assert msg.to_dict() == {"role": "assistant", "content": "Hi"}


class TestChatHistory:
    def test_add_and_to_list(self):
        history = ChatHistory(max_turns=2)
        history.add("system", "You are helpful")
        history.add("user", "Hi")
        history.add("assistant", "Hello")
        assert history.to_list() == [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]

    def test_from_list(self):
        history = ChatHistory.from_list([
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
        ])
        assert [m.role for m in history.messages] == ["user", "assistant"]
        assert [m.content for m in history.messages] == ["A", "B"]

    def test_clear(self):
        history = ChatHistory()
        history.add("user", "Hi")
        history.clear()
        assert history.messages == []

    def test_trim_keeps_system_and_last_turns(self):
        history = ChatHistory(max_turns=2)
        history.add("system", "rules")
        for i in range(5):
            history.add("user", f"u{i}")
            history.add("assistant", f"a{i}")

        # One system message plus the last 4 non-system messages.
        assert history.messages[0] == ChatMessage("system", "rules")
        assert [(m.role, m.content) for m in history.messages[1:]] == [
            ("user", "u3"),
            ("assistant", "a3"),
            ("user", "u4"),
            ("assistant", "a4"),
        ]

    def test_trim_keeps_all_system_messages(self):
        history = ChatHistory(max_turns=1)
        history.add("system", "first")
        history.add("system", "second")
        history.add("user", "u0")
        history.add("assistant", "a0")
        history.add("user", "u1")
        history.add("assistant", "a1")

        assert [(m.role, m.content) for m in history.messages] == [
            ("system", "first"),
            ("system", "second"),
            ("user", "u1"),
            ("assistant", "a1"),
        ]


class TestChatTemplates:
    def test_all_templates_have_required_keys(self):
        for name, tpl in CHAT_TEMPLATES.items():
            assert {"name", "system", "user", "assistant", "stop"}.issubset(tpl)
            assert isinstance(tpl["stop"], list), name

    def test_chatml_template_with_generation_prompt(self):
        history = ChatHistory.from_list([
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ])
        prompt = apply_chat_template(history, "chatml", add_generation_prompt=True)
        assert prompt == (
            "<|im_start|>system\nBe concise<|im_end|>\n"
            "<|im_start|>user\nHi<|im_end|>\n"
            "<|im_start|>assistant\nHello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def test_simple_template_unknown_template_fallback(self):
        prompt = apply_chat_template(
            [ChatMessage("user", "Question?")],
            template_name="does-not-exist",
            add_generation_prompt=True,
        )
        assert prompt == "User: Question?\nAssistant: "

    def test_simple_template_without_generation_prompt(self):
        prompt = apply_chat_template(
            [ChatMessage("user", "Question?")],
            template_name="simple",
            add_generation_prompt=False,
        )
        assert prompt == "User: Question?\n"

    def test_gemma_generation_prompt(self):
        prompt = apply_chat_template(
            [ChatMessage("user", "Hi")],
            template_name="gemma",
            add_generation_prompt=True,
        )
        assert prompt.endswith("<start_of_turn>model\n")

    def test_phi_generation_prompt(self):
        prompt = apply_chat_template(
            [ChatMessage("user", "Hi")],
            template_name="phi",
            add_generation_prompt=True,
        )
        assert prompt.endswith("<|assistant|>\n")

    def test_llama_and_mistral_do_not_append_extra_generation_prefix(self):
        for template in ("llama2", "mistral"):
            prompt = apply_chat_template(
                [ChatMessage("user", "Hi")],
                template_name=template,
                add_generation_prompt=True,
            )
            assert prompt.endswith("[/INST]")

    def test_mistral_omits_system_message(self):
        prompt = apply_chat_template(
            [ChatMessage("system", "rules"), ChatMessage("user", "Hi")],
            template_name="mistral",
            add_generation_prompt=False,
        )
        assert "rules" not in prompt
        assert prompt == "[INST] Hi [/INST]"

    def test_get_stop_tokens_known_and_unknown(self):
        assert "<|im_end|>" in get_stop_tokens("chatml")
        assert get_stop_tokens("missing") == []

    def test_build_chat_prompt_inserts_system_once(self):
        history = ChatHistory.from_list([{"role": "user", "content": "Hi"}])
        prompt1 = build_chat_prompt(history, template="simple", system_prompt="rules")
        prompt2 = build_chat_prompt(history, template="simple", system_prompt="new rules")

        assert prompt1.startswith("System: rules\n")
        assert prompt2.count("System:") == 1
        assert "new rules" not in prompt2

    def test_build_chat_prompt_keeps_existing_system(self):
        history = ChatHistory.from_list([
            {"role": "system", "content": "existing"},
            {"role": "user", "content": "Hi"},
        ])
        prompt = build_chat_prompt(history, template="simple", system_prompt="ignored")
        assert prompt.startswith("System: existing\n")
        assert "ignored" not in prompt
