"""2.26: the Studio's Images tab, the chat's agent mode and insights - and the parts the
Studio and the classic interface now share.

Part 3 of moving everything the tkinter interface does into AuraLite Studio: the Images tab
(an image generator of one's own: every setting of the classic Image tab, samples, the
images and the model saved, the quality measured), the chat's agent mode (the tools turned
on, the steps and the seconds, one temporary folder a chat) and its insights (the reply's
tokens, one layer's attention: a head or the average). The toolkit-free parts run in both
interfaces: the image settings, the agent's options, the insights, the training runs as a
window starts them (`studio/runner.py`), the Fine-tune form, settings files and the Start
tab's lines.
"""
from __future__ import annotations

import json
import os
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from auralite.studio import agentmode, finetuneform, imagegen, insights, runner, startinfo
from auralite.studio.recipes import FineTuneSettings, SettingsError, finetune_params
from auralite.studio.settingsfile import SETTINGS_FORMAT, read_settings, write_settings

GB = 1 << 30
TEXT = ("AuraLite trains small language models on CPU. It supports RMSNorm, RoPE, SwiGLU "
        "and grouped-query attention. ") * 30
TINY = {"tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 4, "n_kv_heads": 2,
        "n_layers": 2, "seq_length": 16, "batch_size": 8, "lr": 2e-2, "epochs": 1,
        "val_split": 0.0, "seed": 0, "optimizer": "adamw", "amp_dtype": "none",
        "progress_every_seconds": 100.0}


def _hardware(memory_gb: float = 16, gflops: float = 200.0):
    from auralite.studio.hardware import Hardware
    return Hardware(system="Linux", cpu="Test CPU", cores=4, threads=8, simd="AVX2",
                    x86_64_v2=True, memory_total=int(memory_gb * GB),
                    memory_available=int(memory_gb * 0.75 * GB), train_threads=4,
                    gflops=gflops)


def _dialogs_file(path, n: int = 60) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            a, b = i % 7, i % 5
            f.write(json.dumps({"messages": [
                {"role": "user", "content": f"what is {a} plus {b}?"},
                {"role": "assistant", "content": f"{a} plus {b} is {a + b}."}]}) + "\n")
    return str(path)


def _pairs_file(path, n: int = 30) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"prompt": f"what is {i % 9} plus 1?",
                                "chosen": f"{i % 9} plus 1 is {i % 9 + 1}.",
                                "rejected": "I do not know."}) + "\n")
    return str(path)


def _tiny_engine():
    from auralite.engine import AuraLiteEngine
    engine = AuraLiteEngine()
    engine.train(TEXT, dict(TINY))
    return engine


def _tiny_base(tmp_path) -> str:
    path = str(tmp_path / "base.pt")
    _tiny_engine().save_model(path)
    return path


# ----------------------------------------------------------------------
#  Toolkit-free: the image generator's settings
# ----------------------------------------------------------------------
def test_image_fields_are_the_classic_tabs():
    keys = [f.key for f in imagegen.FIELDS]
    assert len(keys) == len(set(keys)) and set(imagegen.FIELD_BY_KEY) == set(keys)
    assert {f.section for f in imagegen.FIELDS} == {key for key, _ in imagegen.SECTIONS}
    values = imagegen.defaults()
    assert (values["img_size"], values["color"], values["epochs"], values["lr"]) == \
        ("32", "RGB", "80", "0.0006")
    assert values["use_ema"] is True and values["continue_training"] is False
    assert imagegen.FIELD_BY_KEY["latent_dim"].applies_to("gan")
    assert not imagegen.FIELD_BY_KEY["latent_dim"].applies_to("diffusion")
    assert imagegen.FIELD_BY_KEY["attn_heads"].applies_to("flow")
    assert not imagegen.FIELD_BY_KEY["attn_heads"].applies_to("gan")
    tiny = imagegen.preset_values("Tiny (CPU-friendly)")
    assert tiny["img_size"] == "32" or tiny["img_size"].isdigit()
    assert tiny.get("model_type", "diffusion") == "diffusion"
    assert imagegen.preset_values("Flow Tiny (CPU-friendly)")["model_type"] == "flow"
    assert imagegen.preset_values("no such preset") == {}
    assert imagegen.parse_int_list("16, 32;64") == [16, 32, 64]
    assert imagegen.parse_int_list(" Auto ") is None and imagegen.parse_int_list("") is None


def test_train_params_are_checked_with_the_fields_names():
    params = imagegen.train_params(imagegen.defaults(), "diffusion")
    assert params["model_type"] == "diffusion" and params["channels"] == 3
    assert "color" not in params and params["channel_mults"] is None
    assert params["img_size"] == 32 and params["lr"] == pytest.approx(6e-4)
    gray = imagegen.train_params({**imagegen.defaults(), "color": "Grayscale",
                                  "autosave_every": "", "channel_mults": "1,2"}, "flow")
    assert gray["channels"] == 1 and gray["autosave_every"] == 0
    assert gray["channel_mults"] == [1, 2] and gray["model_type"] == "flow"
    with pytest.raises(SettingsError, match=r"Image size \(px\): 'big' is not a whole number"):
        imagegen.train_params({**imagegen.defaults(), "img_size": "big"}, "diffusion")
    with pytest.raises(SettingsError, match="Color: 'CMYK' is not one of RGB, Grayscale"):
        imagegen.train_params({**imagegen.defaults(), "color": "CMYK"}, "diffusion")
    with pytest.raises(SettingsError, match="comma list"):
        imagegen.train_params({**imagegen.defaults(), "channel_mults": "1,x"}, "diffusion")
    with pytest.raises(SettingsError, match=r"^Image size \(px\) "):  # the engine's own rule
        imagegen.train_params({**imagegen.defaults(), "img_size": "20"}, "diffusion")
    with pytest.raises(SettingsError, match="unknown model type"):
        imagegen.train_params(imagegen.defaults(), "vae")


def test_generation_kwargs_follow_the_model_type():
    gan = imagegen.generation_kwargs("gan", n=99, seed=" 7 ", truncation="0.5",
                                     class_choice=imagegen.ALL_CLASSES)
    assert gan == {"n": imagegen.MAX_SAMPLES, "use_ema": True, "seed": 7, "truncation": 0.5,
                   "class_label": "all"}
    ddpm = imagegen.generation_kwargs("diffusion", n=2, sampler="ddpm", steps=0, eta=0.0)
    assert (ddpm["sampler"], ddpm["eta"], ddpm["steps"], ddpm["seed"]) == ("ddpm", 1.0, 1, None)
    assert "class_label" not in ddpm and "guidance_scale" not in ddpm
    flow = imagegen.generation_kwargs("flow", sampler="ddpm", class_choice="cats", guidance=0)
    assert flow["sampler"] in imagegen.SAMPLERS_IN_ROW["flow"] + ("ddim",)
    assert flow["class_label"] == "cats" and flow["guidance_scale"] == 1.0
    with pytest.raises(ValueError):
        imagegen.generation_kwargs("diffusion", steps="many")
    assert imagegen.class_choices([]) == [imagegen.ANY_CLASS]
    assert imagegen.class_choices(["a", "b"]) == [imagegen.ANY_CLASS, imagegen.ALL_CLASSES,
                                                  "a", "b"]
    assert imagegen.class_label(imagegen.ANY_CLASS) is None


def test_image_texts():
    assert imagegen.summary_text(None) == ""
    line = imagegen.summary_text({"steps": 40, "seconds": 75, "best_val_loss": 0.1234,
                                  "best_epoch": 3, "stopped_early": True, "restored_best": True,
                                  "augment": "ada", "augment_p": 0.25})
    assert line.startswith("40 steps · ") and "best val 0.1234 (epoch 3)" in line
    assert "early stop, best weights restored" in line and "augmentation p 0.25" in line
    assert "SWD" in imagegen.summary_text({"best_val_loss": 2.0, "val_metric": "swd"})
    setup = imagegen.describe_progress(0, 2, None, {"phase": "setup", "message": "Loading",
                                                    "stage_percent": 50})
    assert setup["phase"] == "setup" and setup["percent"] == 50.0
    end = imagegen.describe_progress(1, 2, 0.5, {"is_epoch_end": True, "val_loss": 0.4,
                                                 "d_loss": 0.7})
    assert end["status"] == "Epoch 1/2 · G 0.5000 · D 0.7000 · val 0.4000"
    assert (end["loss"], end["val"], end["d_loss"], end["swd"]) == (0.5, 0.4, 0.7, False)
    quality = imagegen.quality_text({"swd": 12.34, "real_floor": 3.0, "levels": {16: 10.0},
                                     "images": 24, "seconds": 2})
    assert quality.startswith("Quality: SWD 12.3 (lower is better; two sets of real images")
    assert "16px 10.0" in quality and "24 images" in quality
    gan = SimpleNamespace(class_names=["a", "b"], discriminator=None)
    assert imagegen.loaded_text({"model_type": "gan", "img_size": 16, "channels": 3,
                                 "latent_dim": 64}, gan) == \
        "Loaded a 16px 3ch model (GAN, latent 64, no discriminator, 2 classes)."


# ----------------------------------------------------------------------
#  Toolkit-free: agent mode
# ----------------------------------------------------------------------
def test_agent_options_come_from_the_form():
    options = agentmode.options_from_form(["calculate", "python"], " 12 ", "30",
                                          temperature=0.2)
    assert (options.tools, options.max_iterations, options.timeout, options.temperature) == \
        (("calculate", "python"), 12, 30.0, 0.2)
    for steps, seconds in (("1", "1"), ("30", "600")):          # the limits themselves
        edge = agentmode.options_from_form(["calculate"], steps, seconds)
        assert (edge.max_iterations, edge.timeout) == (int(steps), float(seconds))
    for steps in ("0", "31", "x", ""):
        with pytest.raises(ValueError, match="steps are a whole number from 1 to 30"):
            agentmode.options_from_form(["calculate"], steps)
    for seconds in ("0", "601", "soon"):
        with pytest.raises(ValueError, match="from 1 to 600"):
            agentmode.options_from_form(["calculate"], 8, seconds)
    with pytest.raises(ValueError, match="at least one tool"):
        agentmode.options_from_form([], 8, 15)


def test_the_agent_fields_start_from_the_environment(monkeypatch):
    monkeypatch.delenv("AURALITE_AGENT_MAX_ITERATIONS", raising=False)
    monkeypatch.delenv("AURALITE_AGENT_TOOL_TIMEOUT", raising=False)
    assert agentmode.form_defaults() == (8, 15.0)
    monkeypatch.setenv("AURALITE_AGENT_MAX_ITERATIONS", "12")
    monkeypatch.setenv("AURALITE_AGENT_TOOL_TIMEOUT", "30")
    assert agentmode.form_defaults() == (12, 30.0)
    monkeypatch.setenv("AURALITE_AGENT_MAX_ITERATIONS", "99")          # within the limits
    monkeypatch.setenv("AURALITE_AGENT_TOOL_TIMEOUT", "0.1")
    assert agentmode.form_defaults() == (30, 1.0)
    monkeypatch.setenv("AURALITE_AGENT_MAX_ITERATIONS", "many")        # not a number
    monkeypatch.setenv("AURALITE_AGENT_TOOL_TIMEOUT", "nan")
    assert agentmode.form_defaults() == (8, 15.0)


def test_the_python_tools_need_a_python(monkeypatch):
    import sys

    from auralite import bundle
    from auralite.agent import TOOL_REGISTRY
    rows = {row.name: row for row in agentmode.tool_rows(None)}
    assert list(rows) == list(TOOL_REGISTRY)
    assert not rows["python"].available and not rows["install"].available
    assert "needs Python" in rows["python"].note and rows["calculate"].available
    assert agentmode.python_for_tools() == sys.executable          # not the app: this one
    monkeypatch.setattr(bundle, "packaged", lambda: True)          # the app: on the PATH
    found = {"python3": r"C:\Users\me\AppData\Local\Microsoft\WindowsApps\python3.exe",
             "python": r"C:\Python313\python.exe"}
    monkeypatch.setattr(agentmode.shutil, "which", lambda name: found.get(name))
    assert agentmode.python_for_tools() == r"C:\Python313\python.exe"   # not the Store's
    assert {r.name: r for r in agentmode.tool_rows()}["python"].note == \
        r"runs C:\Python313\python.exe"
    monkeypatch.setattr(agentmode.shutil, "which", lambda name: None)
    assert agentmode.python_for_tools() is None


def test_the_agent_gets_the_tools_and_limits_it_may_use(tmp_path):
    options = agentmode.options_from_form(["python", "calculate"], 5, 20,
                                          max_tokens_per_step=8, chat_template="")
    agent = agentmode.make_agent(object(), options, python=None)     # no Python: left out
    try:
        assert agent.enabled_tools == ["calculate"]
        assert (agent.max_iterations, agent.timeout_per_tool) == (5, 20.0)
        assert agent.max_tokens_per_step == 16 and agent.chat_template == "chatml"
        assert agent.sandbox.timeout == 20.0
    finally:
        agent.cleanup()
    with pytest.raises(ValueError, match="at least one tool"):
        agentmode.make_agent(object(), agentmode.AgentOptions(tools=("python",)), python=None)
    sandbox = agentmode.make_sandbox(15, python=None)
    try:
        kept = agentmode.make_agent(object(), options, sandbox=sandbox)
        assert kept.sandbox is sandbox and sandbox.timeout == 20.0
        kept.cleanup()                                   # a chat's sandbox stays
        assert sandbox.workdir.is_dir()
    finally:
        sandbox.cleanup()
    step = SimpleNamespace(tool_calls=[{"name": "calculate"}, "odd"],
                           tool_results=[SimpleNamespace(success=True, to_text=lambda: "= 4"),
                                         SimpleNamespace(success=False,
                                                         to_text=lambda: "x" * 400)])
    assert agentmode.step_lines(step) == [("→ Tool: calculate", "tool"), ("→ Tool: odd", "tool"),
                                          ("= 4", "ok"), ("x" * 300, "error")]


def test_the_agent_runs_only_the_tools_it_was_given(tmp_path):
    """2.26: a call to a tool left out of `enabled_tools` gets an error - it was run."""
    from auralite.agent import AuraLiteAgent, Sandbox
    marker = tmp_path / "ran.txt"

    class Model:
        def __init__(self):
            self.prompts: list[Any] = []

        def generate_chat(self, messages, **kwargs):
            self.prompts.append(messages)
            if len(self.prompts) == 1:
                return (f'<tool name="python">open({str(marker)!r}, "w").write("1")</tool>'
                        '<tool name="calculate">2+2</tool>')
            return "Done."

    model = Model()
    sandbox = Sandbox(timeout=10)
    steps: list[Any] = []
    agent = AuraLiteAgent(model, sandbox=sandbox, enabled_tools=["calculate"],
                          max_iterations=3, on_step=steps.append)
    try:
        assert agent.run("write a file") == "Done."
    finally:
        sandbox.cleanup()
    assert not marker.exists()                                  # python never ran
    refused, calculated = steps[0].tool_results
    assert not refused.success and "not enabled here" in refused.to_text()
    assert "Enabled tools: calculate" in refused.to_text()
    assert calculated.success and "4" in calculated.to_text()
    assert "not enabled here" in model.prompts[1][-1]["content"]   # the model reads why


def test_a_streamed_step_reports_the_call_its_result_answers():
    from auralite.agent import AuraLiteAgent, Sandbox

    class Model:
        calls = 0

        def generate_chat_streaming(self, messages, **kwargs):
            Model.calls += 1
            yield ('<tool name="calculate">1+1</tool><tool name="calculate">2+3</tool>'
                   if Model.calls == 1 else "Five.")

    steps: list[Any] = []
    sandbox = Sandbox(timeout=10)
    agent = AuraLiteAgent(Model(), sandbox=sandbox, enabled_tools=["calculate"],
                          max_iterations=2, on_step=steps.append)
    try:
        text = "".join(agent.run_streaming("sums"))
    finally:
        sandbox.cleanup()
    assert "Five." in text
    assert [[c["args"] for c in s.tool_calls] for s in steps[:2]] == \
        [[{"expression": "1+1"}], [{"expression": "2+3"}]]


# ----------------------------------------------------------------------
#  Toolkit-free: insights
# ----------------------------------------------------------------------
def test_insights_helpers():
    assert insights.visible("a b\n\t\r") == "a␣b⏎⇥␍" and insights.visible("") == "∅"
    chat = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "more"}, {"role": "assistant", "content": ""}]
    assert insights.last_reply(chat) == "hello" and insights.last_reply([]) == ""
    assert insights.last_exchange(chat) == "hello\nmore"
    assert insights.inspectable(SimpleNamespace(model=None)) == "There is no model in memory."
    gguf = SimpleNamespace(model=object(), is_gguf_model=lambda: True)
    assert "GGUF" in insights.inspectable(gguf)
    assert insights.inspectable(SimpleNamespace(model=object(), hf_proxy=object()))
    assert insights.inspectable(SimpleNamespace(model=object(), is_gguf_model=lambda: False,
                                                hf_proxy=None)) == ""
    assert insights.top_tokens([(" a", 3), ("b", 2), ("c", 1)], 2) == [("␣a", 3), ("b", 2)]
    assert insights.parse_head("") is None and insights.parse_head("AVG") is None
    assert insights.parse_head("0") is None and insights.parse_head(" 3 ") == 2
    for bad in ("-1", "two"):
        with pytest.raises(ValueError, match="number from 1, or avg"):
            insights.parse_head(bad)
    assert insights.parse_tokens("") == insights.DEFAULT_TOKENS == 64
    assert insights.parse_tokens("2") == 8 and insights.parse_tokens("9999") == 256
    with pytest.raises(ValueError, match="from 8 to 256"):
        insights.parse_tokens("lots")
    assert insights.attention_caption(1, None, 4) == \
        "Layer 1, the average of 4 heads: each row is where that token looked"
    assert insights.attention_caption(0, 2, 4).startswith("Layer 0, head 3 of 4")


def test_attention_view_reads_one_head_or_the_average():
    import numpy as np
    engine = _tiny_engine()
    assert insights.layer_count(engine) == 2
    text = "AuraLite trains small models"
    average, labels, heads = insights.attention_view(engine, text, layer=1, max_tokens=12)
    assert heads == 4 and average.shape == (12, 12) and len(labels) == 12
    assert labels[-1] == "s" and "␣" in labels                   # the end of the text
    maps = np.asarray(engine.attention_maps(text, layer=1, max_tokens=12))
    assert np.allclose(average, maps.mean(axis=0))
    one, _labels, _heads = insights.attention_view(engine, text, layer=1, head=3,
                                                   max_tokens=12)
    assert np.allclose(one, maps[3]) and np.allclose(one.sum(axis=-1), 1.0, atol=1e-4)
    with pytest.raises(ValueError, match="head must be from 1 to 4"):
        insights.attention_view(engine, text, layer=0, head=4)


# ----------------------------------------------------------------------
#  Toolkit-free: the Fine-tune form, settings files, the Start tab's lines
# ----------------------------------------------------------------------
def test_the_finetune_form_reads_every_field():
    s = finetuneform.settings_from_form({})
    assert (s.method, s.lora_rank, s.lora_lr, s.epochs, s.seq_length, s.batch_size) == \
        ("qlora", 16, 0.001, 2, 512, 8)
    assert (s.optimizer, s.memory_plan, s.objective, s.prompt_weight, s.preference_beta) == \
        ("adamw8bit", "auto", "auto", None, None)
    full = finetuneform.settings_from_form({"method": "full", "lr": "1e-5", "adam8": False,
                                            "prompt_weight": "0.2", "beta": "0.3",
                                            "objective": "simpo", "memory_plan": False,
                                            "pref_epochs": "2", "lora_targets": "attn"},
                                           pairs=True)
    assert (full.lora_lr, full.lr, full.optimizer, full.prompt_weight) == \
        (None, 1e-5, "adamw", 0.2)
    assert (full.preference_beta, full.objective, full.memory_plan, full.preference_epochs,
            full.lora_targets) == (0.3, "simpo", "off", 2, "attn")
    assert finetuneform.settings_from_form({"objective": "simpo"}).objective == "auto"
    for form, message in (({"lora_rank": "0"}, "the LoRA rank must be from 1 to 512"),
                          ({"lora_lr": "fast"}, "the LoRA learning rate must be a number"),
                          ({"epochs": "1.5"}, "the passes over the data must be a whole"),
                          ({"seq_length": "32"}, "the window must be from 64 to 32,768"),
                          ({"prompt_weight": "half"}, "the prompt weight is a number"),
                          ({"beta": "?"}, "beta is a number above 0"),
                          ({"method": "dora"}, "the method is one of qlora, lora, full")):
        with pytest.raises(SettingsError, match=message):
            finetuneform.settings_from_form(form)


def test_the_finetune_base_and_estimate(tmp_path):
    from auralite.studio.advisor import recommend_finetune
    assert finetuneform.hub_spec(" Qwen/Qwen3.5-0.8B ") == "hf:Qwen/Qwen3.5-0.8B"
    assert finetuneform.hub_spec("hf:org/name") == "hf:org/name"
    folder = tmp_path / "org" / "model"
    folder.mkdir(parents=True)
    assert finetuneform.hub_spec(str(folder)) == str(folder)       # a folder stays one
    assert finetuneform.hub_spec("plain") == "plain"
    rec = recommend_finetune(_hardware())
    choice = rec.choice
    assert choice is not None
    label = finetuneform.catalog_label(choice, True)
    assert label.startswith(choice.model.display_name) and label.endswith("★ recommended")
    assert " · fits" in label or " · tight" in label
    lines = finetuneform.base_lines(choice, "catalog")
    assert lines[0].startswith(f"{choice.model.name}: ") and "download" in lines[0]
    assert finetuneform.base_lines(None, "memory") == ["Fine-tunes the model that is loaded now."]
    assert finetuneform.base_lines(None, "file") == ["Loads the file, then fine-tunes it."]
    assert finetuneform.base_lines(None, "hub") == []
    text = finetuneform.estimate_text(choice.model, choice.method, rec, 512)
    assert text.startswith("Estimate: ~") and "of memory (" in text
    assert finetuneform.estimate_text(None, "qlora", rec, 512) == ""
    assert finetuneform.estimate_text(choice.model, "qlora", None, 512) == ""


def test_settings_files_are_shared(tmp_path):
    path = str(tmp_path / "s.json")
    settings = FineTuneSettings(method="lora", lora_rank=4).to_dict()
    assert write_settings(path, "finetune", settings, ["chats.jsonl"], base="x.pt") == path
    doc = read_settings(path, "finetune")
    assert (doc["format"], doc["version"], doc["tab"], doc["data"], doc["base"]) == \
        (SETTINGS_FORMAT, 1, "finetune", ["chats.jsonl"], "x.pt")
    assert FineTuneSettings.from_dict(doc["settings"]).lora_rank == 4
    with pytest.raises(ValueError, match="holds settings of the finetune tab, not of the train"):
        read_settings(path, "train")
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"format": "something else"}), encoding="utf-8")
    with pytest.raises(ValueError, match="is not a settings file of AuraLite Studio"):
        read_settings(str(other), "finetune")
    with pytest.raises(OSError):
        read_settings(str(tmp_path / "missing.json"), "finetune")


def test_the_start_tabs_lines():
    lines = startinfo.fit_lines(_hardware())
    assert lines[0].startswith("Memory a run may use: ")
    assert lines[1].startswith("Recommended to fine-tune: ")
    assert "Times are rough until the speed test runs." not in lines
    assert "Times are rough until the speed test runs." in startinfo.fit_lines(_hardware(gflops=0))
    small = startinfo.fit_lines(_hardware(memory_gb=1))
    assert small[1] == "None of the catalog's models fits: a small model trained from scratch does"
    run = SimpleNamespace(name="chat-run", progress={"step": 1234},
                          record={"checkpoint_saved": {"step": 1200, "time": 1760000000.0}})
    headline, rest = startinfo.interrupted_text(run, 2)
    assert headline == "A run was interrupted:"
    assert rest.startswith(" chat-run, at step 1,234. Its checkpoint is from step 1,200 (")
    assert rest.endswith("exactly as it would have gone on. (2 more in the library.)")
    bare = SimpleNamespace(name="r2", progress={}, record={})
    assert startinfo.interrupted_text(bare) == \
        ("A run was interrupted:",
         " r2. It continues from its checkpoint exactly as it would have gone on.")


def test_merging_into_a_4_bit_base_asks_first():
    from auralite.studio.modelinfo import merge_question
    assert merge_question(SimpleNamespace(model=None)) == ""
    engine = _tiny_engine()
    assert merge_question(engine) == ""                         # full precision: nothing
    engine.quantize_nf4()
    question = merge_question(engine)
    assert question.startswith("The base is stored in 4 bits. Merged, every weight is a "
                               "float32 one: about ")
    assert question.endswith("Merge now?")


# ----------------------------------------------------------------------
#  Toolkit-free: training runs as a window starts them
# ----------------------------------------------------------------------
def test_records_of_the_machine_and_the_base():
    from auralite.engine import AuraLiteEngine
    engine = AuraLiteEngine()
    plain = runner.machine_record(engine)
    assert "threads" in plain or "cpu" in plain
    hw = _hardware()
    info = runner.machine_record(engine, hw)
    assert (info["cpu"], info["cores"], info["threads"], info["memory_gb"], info["system"]) == \
        ("Test CPU", 4, 8, 16.0, "Linux")
    assert runner.base_record("memory", "", "/m/model.pt", "model.pt") == \
        {"spec": "/m/model.pt", "label": "model.pt"}
    assert runner.base_record("memory", "")["label"] == "the model in memory"
    assert runner.base_record("catalog", "hf:Qwen/Qwen3.5-0.8B")["spec"] == \
        "hf:Qwen/Qwen3.5-0.8B"


def test_a_failed_run_is_recorded_and_raised():
    calls: list[Any] = []

    class Recorder:
        def callback(self, cb):
            return cb

        def fail(self, exc):
            calls.append(("fail", str(exc)))

        def save_and_finish(self, summary, engine):          # pragma: no cover - not reached
            calls.append("saved")

    class Engine:
        def train(self, *a, **k):
            raise MemoryError("out of memory")

    with pytest.raises(MemoryError):
        runner.train_recorded(Engine(), Recorder(), "data", {})
    assert calls == [("fail", "out of memory")]

    class Done(Engine):
        def train(self, *a, **k):
            return {"stopped_during": "training"}

    events: list[Any] = []

    class Saving(Recorder):
        def save_and_finish(self, summary, engine):
            calls.append("saved")

    calls.clear()
    assert runner.train_recorded(Done(), Saving(), "d", {}, on_event=lambda *e: events.append(e))
    assert calls == ["saved"] and events == []         # a stopped run: no "Saving…" stage


def test_a_finetune_from_a_file_is_recorded_stopped_and_continued(tmp_path, monkeypatch):
    from auralite import runs as runlib
    from auralite.engine import AuraLiteEngine
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    base = _tiny_base(tmp_path)
    chats = _dialogs_file(tmp_path / "chats.jsonl")
    settings = finetuneform.settings_from_form({"method": "lora", "lora_rank": "4",
                                                "epochs": "3", "seq_length": "64",
                                                "batch_size": "4", "memory_plan": False})
    params = finetune_params(settings, "dialogs")
    engine = AuraLiteEngine()
    stop = threading.Event()
    events: list[Any] = []

    def progress(current, *rest, **kw):
        if current and int(current) >= 3:
            stop.set()

    record = {"base": runner.base_record("file", base), "settings": settings.to_dict(),
              "source": "gui", "hardware": runner.machine_record(engine, _hardware()),
              "data": {"paths": [chats], "kind": "dialogs", "records": 60}}
    summary = runner.finetune(engine, source="file", spec=base, settings=settings,
                              params=params, paths=[chats], record=record,
                              progress_callback=progress, stop_event=stop,
                              on_event=lambda *e: events.append(e))
    assert isinstance(summary, dict) and engine.model is not None and engine.has_lora()
    kinds = [e[0] for e in events]
    assert kinds[:5] == ["unloaded", "stage", "loaded", "stage", "recorder"]
    assert events[1] == ("stage", "Loading base.pt…", None) and events[2] == ("loaded", base)
    run = runlib.find_run("last")
    assert (run.kind, run.method, run.status, run.source) == ("finetune", "lora", "stopped",
                                                              "gui")
    assert run.base["spec"] == base and run.record["hardware"]["cpu"] == "Test CPU"
    assert run.continue_state() == (True, "")
    assert "(the classic interface" in dict(runlib.describe_run(run))["Started"]

    again = AuraLiteEngine()
    more: list[Any] = []
    done_summary = runner.continue_run(again, run, on_event=lambda *e: more.append(e))
    assert [e[0] for e in more][:5] == ["stage", "unloaded", "continued", "stage", "recorder"]
    assert more[-1] == ("stage", "Saving the run in the library…", None)
    done = runlib.find_run(run.id)
    assert done.status == "finished" and len(done.record["sessions"]) == 2
    assert isinstance(done_summary, dict) and done.summary["continued_run"]
    assert done.summary["epochs_completed"] == 3
    assert again.has_lora()
    # a finished run cannot continue: said before the model in memory is touched
    model = again.model
    with pytest.raises(ValueError, match="cannot continue"):
        runner.continue_run(again, done, on_event=lambda *e: more.append(e))
    assert again.model is model and more[-1][0] == "stage" and "Saving" in more[-1][1]


def test_a_stop_while_the_base_loads_ends_before_the_data(tmp_path, monkeypatch):
    from auralite import runs as runlib
    from auralite.engine import AuraLiteEngine
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    base = _tiny_base(tmp_path)
    stop = threading.Event()
    stop.set()
    settings = FineTuneSettings(method="lora")
    out = runner.finetune(AuraLiteEngine(), source="file", spec=base, settings=settings,
                          params={}, paths=["never-read.jsonl"], record={}, stop_event=stop)
    assert out == {"stopped_during": "loading the base model"}
    assert runlib.list_runs() == []


def test_a_hugging_face_base_is_imported_4_bit_for_qlora():
    """The catalog and a Hugging Face name import the base - in 4 bits (NF4) for QLoRA -
    with the download's progress passed on; a stop during the import ends the run."""
    class Engine:
        def __init__(self):
            self.calls: list[Any] = []

        def unload_model(self):
            self.calls.append("unload")

        def import_hf_model(self, spec, quantize=None, progress_callback=None,
                            stop_event=None):
            self.calls.append((spec, quantize))
            assert progress_callback is not None and stop_event is not None
            progress_callback({"stage": "download", "all_bytes": 5, "all_total": 10})
            stop_event.set()                                    # Stop while it downloads

    for method, quantize in (("qlora", "nf4"), ("lora", None), ("full", None)):
        engine, events = Engine(), []
        out = runner.finetune(engine, source="hub", spec="hf:org/name",
                              settings=FineTuneSettings(method=method), params={},
                              paths=["never-read.jsonl"], record={},
                              stop_event=threading.Event(),
                              on_event=lambda *e, events=events: events.append(e))
        assert out == {"stopped_during": "loading the base model"}
        assert engine.calls == ["unload", ("hf:org/name", quantize)]
        assert events == [("unloaded",), ("stage", "Getting hf:org/name…", 0.0),
                          ("import", {"stage": "download", "all_bytes": 5, "all_total": 10}),
                          ("loaded", "hf:org/name")]


def test_pairs_with_sft_train_on_the_chosen_replies(tmp_path, monkeypatch):
    from auralite import runs as runlib
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    engine = _tiny_engine()
    pairs = _pairs_file(tmp_path / "pairs.jsonl")
    settings = finetuneform.settings_from_form({"method": "lora", "lora_rank": "4",
                                                "epochs": "1", "max_steps": "2",
                                                "seq_length": "64", "batch_size": "4",
                                                "objective": "sft", "memory_plan": False},
                                               pairs=True)
    params = finetune_params(settings, "preferences")
    runner.finetune(engine, source="memory", spec="", settings=settings, params=params,
                    paths=[pairs], record={"base": runner.base_record("memory", ""),
                                           "data": {"paths": [pairs]}})
    run = runlib.find_run("last")
    assert run.kind == "finetune" and run.status == "finished"
    assert run.data["options"] == {"as_dialogs": True}


# ----------------------------------------------------------------------
#  The window
# ----------------------------------------------------------------------
def _qt():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from tests._integration import require
    return require("PySide6.QtWidgets"), require("PySide6.QtCore")


@pytest.fixture()
def dialogs():
    _qt()
    from auralite.studio.qt import common
    seen: dict[str, Any] = {"error": [], "inform": [], "confirm": []}
    answers: dict[str, Any] = {"confirm": True, "save": "", "open": "", "dir": "", "text": ""}
    common.DIALOGS.update(
        error=lambda friendly: seen["error"].append(friendly),
        inform=lambda title, text: seen["inform"].append((title, text)),
        confirm=lambda title, text: seen["confirm"].append((title, text)) or answers["confirm"],
        save=lambda title: answers["save"], open=lambda title: answers["open"],
        dir=lambda title: answers["dir"], text=lambda title: answers["text"])
    seen["answers"] = answers
    yield seen
    common.DIALOGS.clear()


def _window(tmp_path, monkeypatch, engine=None):
    QtWidgets, _QtCore = _qt()
    from auralite.engine import AuraLiteEngine
    from auralite.studio.qt.window import MainWindow
    from auralite.studio.settings import StudioSettings
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    window = MainWindow(engine=engine or AuraLiteEngine(),
                        settings=StudioSettings(first_run_done=True), detect=False,
                        capture_streams=False, save_on_close=False, auto_speed_test=False)
    window.hardware = _hardware()
    window.hardware_changed.emit(window.hardware)
    window.resize(1000, 760)
    window.show()
    app.processEvents()
    window.qapp = app                                        # type: ignore[attr-defined]
    return window


def _close(window) -> None:
    _QtWidgets, QtCore = _qt()
    window.model_dirty = False
    window.images_tab.unsaved = False
    for job in window.running_jobs():
        job.stop_event.set()
        if job.thread is not None:
            job.thread.join(timeout=30)
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    window.qapp.processEvents()


@pytest.fixture()
def studio(tmp_path, monkeypatch, dialogs):
    window = _window(tmp_path, monkeypatch)
    yield window
    _close(window)


def wait(app, condition, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    while not condition():
        if time.time() > deadline:
            raise AssertionError("timed out waiting for the window")
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


def _image_folder(tmp_path) -> str:
    Image = pytest.importorskip("PIL.Image", reason="training reads images with Pillow")
    import numpy as np
    folder = tmp_path / "pictures"
    for name, channel in (("red", 0), ("blue", 2)):
        (folder / name).mkdir(parents=True)
        for i in range(10):
            pixels = np.zeros((16, 16, 3), np.uint8)
            pixels[..., channel] = 120 + i * 10
            Image.fromarray(pixels).save(folder / name / f"{i}.png")
    return str(folder)


def test_the_images_tab_has_the_classic_tabs_choices(studio, dialogs, tmp_path):
    tab = studio.images_tab
    assert "Images" in studio.tabs.tabText(5)
    assert tab.model_type() == "diffusion" and tab.values() == imagegen.defaults()
    tab.set_model_type("gan")
    assert tab.model_type() == "gan" and not tab.sampler.isEnabled()
    assert tab.fields["latent_dim"].isEnabled() and not tab.fields["timesteps"].isEnabled()
    assert tab.truncation.isEnabled() and not tab.steps.isEnabled()
    tab.set_model_type("flow")
    assert [tab.sampler.itemText(i) for i in range(tab.sampler.count())] == \
        list(imagegen.SAMPLERS_IN_ROW["flow"])
    tab.start()                                                    # no folder yet
    assert "choose the folder of images" in dialogs["error"][-1].message
    (tmp_path / "empty").mkdir()
    assert not tab.set_folder(str(tmp_path / "empty"))
    assert "there are no images in" in dialogs["error"][-1].message
    tab.folder = "somewhere"
    tab.pillow_note = "Needs the package «PIL»: pip install Pillow"
    tab.start()
    assert "reading image files needs Pillow" in dialogs["error"][-1].message
    assert studio.busy == ""


def test_the_images_tab_trains_generates_saves_and_measures(studio, tmp_path, dialogs):
    folder = _image_folder(tmp_path)
    tab = studio.images_tab
    tab.pillow_note = ""
    assert tab.set_folder(folder) and tab.folder_label.text().startswith("20 images")
    tab.set_values({"img_size": "16", "base_channels": "8", "num_res_blocks": "1",
                    "epochs": "2", "batch_size": "8", "timesteps": "20", "val_split": "0.2",
                    "class_conditional": True, "warmup_steps": "0"})
    tab.set_values({"img_size": "20"})
    tab.start()                                                     # the engine's rule
    assert dialogs["error"].pop().message == \
        "Image size (px) must be one of 16/24/32/48/64/96/128, got 20"
    tab.set_values({"img_size": "16"})
    tab.start()
    assert studio.busy and tab.job_kind == "train"
    assert not studio.begin("something else", quiet=True)          # one job at a time
    wait(studio.qapp, lambda: tab.job is None, timeout=300)
    assert dialogs["error"] == [], [e.message for e in dialogs["error"]]
    assert len(tab.losses) == 2 and len(tab.val_losses) == 2 and tab.unsaved
    assert tab.params_label.text().startswith("Parameters: ")
    assert [tab.class_choice.itemText(i) for i in range(tab.class_choice.count())] == \
        imagegen.class_choices(["blue", "red"])
    tab.count.setValue(3)
    tab.steps.setValue(4)
    tab.class_choice.setCurrentIndex(1)                             # one of each
    tab.generate()
    wait(studio.qapp, lambda: tab.job is None, timeout=120)
    assert len(tab.images) == 3 and tab.preview.pixmap() is not None
    out = tmp_path / "out"
    out.mkdir()
    written = tab.write_images(str(out))
    assert len(written) >= 3 and all(open(p, "rb").read(8) == b"\x89PNG\r\n\x1a\n"
                                     for p in written)
    # opening another generator asks first while this one is not saved
    dialogs["answers"]["confirm"] = False
    assert tab.load_model(str(tmp_path / "other.pt")) is False
    dialogs["answers"]["confirm"] = True
    dialogs["answers"]["save"] = str(tmp_path / "gen.pt")
    tab.save_model()
    assert os.path.isfile(tmp_path / "gen.pt") and not tab.unsaved
    tab.measure_quality()
    wait(studio.qapp, lambda: tab.job is None, timeout=300)
    assert tab.status.text().startswith("Quality: SWD ")
    assert tab.load_model(str(tmp_path / "gen.pt"))
    assert tab.status.text().startswith("Loaded a 16px 3ch model (") and "2 classes" in \
        tab.status.text()
    assert dialogs["error"] == [], [e.message for e in dialogs["error"]]


def test_quitting_asks_about_an_unsaved_image_generator(studio, dialogs, monkeypatch):
    from PySide6.QtGui import QCloseEvent
    tab = studio.images_tab
    monkeypatch.setattr(tab, "has_model", lambda: True)
    tab.unsaved = True
    dialogs["answers"]["confirm"] = False
    event = QCloseEvent()
    studio.closeEvent(event)
    assert not event.isAccepted()
    assert dialogs["confirm"][-1] == ("Quit AuraLite Studio?",
                                      "The image generator in memory is not saved. Quit anyway?")
    tab.unsaved = False


@pytest.fixture()
def chat_studio(tmp_path, monkeypatch, dialogs):
    monkeypatch.setenv("AURALITE_AGENT_MAX_ITERATIONS", "12")
    monkeypatch.setenv("AURALITE_AGENT_TOOL_TIMEOUT", "30")
    window = _window(tmp_path, monkeypatch, engine=_tiny_engine())
    window.model_changed.emit()
    window.qapp.processEvents()
    yield window
    _close(window)


def test_chat_insights_show_tokens_and_attention(chat_studio, dialogs):
    chat = chat_studio.chat_tab
    chat.show_token_frequencies()                                  # no reply yet
    assert dialogs["inform"][-1][0] == "Insights"
    assert chat.layer.maximum() == 1
    chat.input.setPlainText("Aura")
    chat.send()
    wait(chat_studio.qapp, lambda: chat.job is None)
    assert chat.messages[-1]["role"] == "assistant"
    chat.show_token_frequencies()
    assert chat.insight_view.mode == "bars" and 0 < len(chat.insight_view.bars) <= 20
    assert "different tokens in" in chat.insight_status.text()
    chat.layer.setValue(1)
    chat.attention_tokens.setValue(16)
    chat.show_attention()
    assert chat.insight_view.mode == "heatmap"
    assert chat.insight_view.matrix.shape[0] == chat.insight_view.matrix.shape[1] <= 16
    assert "the average of 4 heads" in chat.insight_view.title
    chat.head.setValue(2)
    chat.show_attention()
    assert "head 2 of 4" in chat.insight_view.title
    chat.head.setValue(chat.head.maximum())
    if chat.head.maximum() > 4:                                    # a head the layer lacks
        chat.show_attention()
        assert "head must be from 1 to 4" in dialogs["error"][-1].message
    chat.new_chat()
    assert (chat.insight_view.mode, chat.insight_view.matrix, chat.messages) == ("", None, [])


def test_chat_agent_mode_warns_once_and_uses_the_form(chat_studio, dialogs, monkeypatch):
    chat = chat_studio.chat_tab
    assert (chat.agent_steps.value(), chat.agent_timeout.value()) == (12, 30)   # the env
    dialogs["answers"]["confirm"] = False
    chat.agent_mode.setChecked(True)
    assert not chat.agent_mode.isChecked() and len(dialogs["confirm"]) == 1
    dialogs["answers"]["confirm"] = True
    chat.agent_mode.setChecked(True)
    assert chat.agent_mode.isChecked() and len(dialogs["confirm"]) == 2
    chat.agent_mode.setChecked(False)
    chat.agent_mode.setChecked(True)
    assert len(dialogs["confirm"]) == 2                            # once a session

    for box in chat.tool_boxes.values():
        box.setChecked(False)
    chat.input.setPlainText("2+2?")
    chat.send()
    assert "at least one tool" in dialogs["error"][-1].message
    assert chat.input.toPlainText() == "2+2?" and chat.job is None and chat.messages == []

    made: list[Any] = []
    sandboxes: list[Any] = []

    class FakeSandbox:
        def __init__(self, timeout):
            self.timeout = timeout
            self.cleaned = False
            sandboxes.append(self)

        def cleanup(self):
            self.cleaned = True

    class FakeAgent:
        def __init__(self, options, sandbox):
            self.options, self.sandbox = options, sandbox
            self.on_step: Any = None
            self.history: Any = None
            self.cleaned = False

        def run_streaming(self, message, history=None):
            self.history = history
            self.on_step(SimpleNamespace(
                tool_calls=[{"name": "calculate"}],
                tool_results=[SimpleNamespace(success=True, to_text=lambda: "= 4")]))
            yield "It is "
            yield "4."

        def stop(self):
            pass

        def cleanup(self):
            self.cleaned = True

    def make_agent(engine, options, python=False, sandbox=None):
        made.append(FakeAgent(options, sandbox))
        return made[-1]

    monkeypatch.setattr(agentmode, "make_sandbox", lambda timeout, python=False:
                        FakeSandbox(timeout))
    monkeypatch.setattr(agentmode, "make_agent", make_agent)
    chat.tool_boxes["calculate"].setChecked(True)
    chat.agent_steps.setValue(5)
    chat.agent_timeout.setValue(40)
    chat.messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]
    chat.input.setPlainText("2+2?")
    chat.send()
    wait(chat_studio.qapp, lambda: chat.job is None)
    agent = made[-1]
    assert agent.options.tools == ("calculate",)
    assert (agent.options.max_iterations, agent.options.timeout) == (5, 40.0)
    assert agent.history == [{"role": "user", "content": "hi"},
                             {"role": "assistant", "content": "hey"}]   # without the message
    assert chat.messages[-1] == {"role": "assistant", "content": "It is 4."}
    assert agent.cleaned and agent.sandbox is sandboxes[0] and not sandboxes[0].cleaned
    log = chat.sandbox_log.toPlainText()
    assert "→ Tool: calculate" in log and "= 4" in log and "Agent done." in log
    chat.input.setPlainText("again")
    chat.send()
    wait(chat_studio.qapp, lambda: chat.job is None)
    assert made[-1].sandbox is sandboxes[0] and len(sandboxes) == 1   # one folder a chat
    chat.new_chat()
    assert sandboxes[0].cleaned and chat.sandbox is None
    chat.input.setPlainText("new")
    chat.send()
    wait(chat_studio.qapp, lambda: chat.job is None)
    assert len(sandboxes) == 2
    chat.agent_mode.setChecked(False)                               # off: the folder goes
    assert sandboxes[1].cleaned and chat.sandbox is None


def test_a_settings_file_of_the_classic_interface_loads_into_the_studio(studio, tmp_path,
                                                                        dialogs):
    base = _tiny_base(tmp_path)
    chats = _dialogs_file(tmp_path / "chats.jsonl")
    settings = FineTuneSettings(method="lora", lora_rank=8, epochs=3, seq_length=128,
                                batch_size=4, lora_targets="attn", prompt_weight=0.2)
    path = write_settings(str(tmp_path / "ft.json"), "finetune", settings.to_dict(), [chats],
                          base=base)
    dialogs["answers"]["open"] = path
    tab = studio.finetune_tab
    tab.load_settings()
    wait(studio.qapp, lambda: tab.data.report is not None)
    assert dialogs["error"] == [], [e.message for e in dialogs["error"]]
    assert tab.source() == "file" and tab.base_spec() == base
    s = tab.settings()
    assert (s.method, s.lora_rank, s.epochs, s.seq_length, s.batch_size, s.lora_targets,
            s.prompt_weight) == ("lora", 8, 3, 128, 4, "attn", 0.2)
    assert tab.data.paths() == [chats]
