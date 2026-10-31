"""JSON mode, JSON schemas and tools on the server (2.17), against a real tiny model.

The model is a few dozen steps old and writes nonsense: every answer must
still parse and follow the schema, because the server decodes it under a JSON
grammar. Tool calls come back in OpenAI's (and Ollama's) shape, streamed too,
and the official OpenAI client reads them (`openai` installed: the test runs).
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import auralite.server.openai_server as srv  # noqa: E402
from auralite.engine import AuraLiteEngine  # noqa: E402
from auralite.engine import structured as S  # noqa: E402

TEXT = ('{"name": "get_weather", "arguments": {"city": "Riga", "unit": "c"}}\n'
        '{"name": "Ann", "age": 34, "tags": ["a", "b"], "email": null, "ok": true}\n'
        '{"name": "Bob", "age": 7, "tags": [], "email": "b@x.lv", "ok": false, "v": -1.5e3}\n'
        'get_time tz UTC noop Person Address city zip street GetWeather\n') * 12
PARAMS = {"tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 4, "n_layers": 2,
          "seq_length": 32, "batch_size": 16, "lr": 5e-3, "epochs": 100, "max_steps": 30,
          "val_split": 0.0, "seed": 0}
USER = [{"role": "user", "content": "Ann"}]
HOT = {"temperature": 1.0, "top_p": 1.0, "top_k": 0}
PERSON = {"type": "object", "title": "Person", "additionalProperties": False,
          "properties": {"name": {"type": "string", "title": "Name"},
                         "age": {"type": "integer"},
                         "email": {"anyOf": [{"type": "string"}, {"type": "null"}],
                                   "default": None},
                         "address": {"$ref": "#/$defs/Address"}},
          "required": ["name", "age", "email", "address"],
          "$defs": {"Address": {"type": "object", "additionalProperties": False,
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"]}}}
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "The weather in a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                    "unit": {"enum": ["c", "f"]}},
                   "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_time", "parameters": {
        "type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]}}}]


@pytest.fixture(scope="module")
def native_engine():
    torch.manual_seed(0)
    engine = AuraLiteEngine()
    engine.train(TEXT, PARAMS)
    return engine


@pytest.fixture()
def client(native_engine, monkeypatch):
    monkeypatch.setattr(srv, "_engine", native_engine)
    monkeypatch.setattr(srv, "_rate_bucket", {})
    with TestClient(srv.app) as test_client:
        yield test_client


def _frames(response) -> list[dict]:
    return [json.loads(line[6:]) for line in response.text.splitlines()
            if line.startswith("data: {")]


def _chat(client, **body):
    r = client.post("/v1/chat/completions", json={"messages": USER, "max_tokens": 80,
                                                   **HOT, **body})
    assert r.status_code == 200, r.text
    return r.json()


# ----------------------------------------------------------------------
#  response_format
# ----------------------------------------------------------------------
def test_json_object_answers_parse(client):
    body = _chat(client, response_format={"type": "json_object"}, n=4)
    assert len(body["choices"]) == 4
    for choice in body["choices"]:
        assert isinstance(json.loads(choice["message"]["content"]), dict)
        assert choice["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] > 0


def test_json_schema_answers_follow_the_schema(client):
    fmt = {"type": "json_schema", "json_schema": {"name": "Person", "schema": PERSON,
                                                  "strict": True}}
    for seed in range(6):
        content = _chat(client, response_format=fmt, seed=seed)["choices"][0]["message"][
            "content"]
        assert S.validate_json(json.loads(content), PERSON) == [], content


def test_streamed_json_joins_to_valid_json(client):
    r = client.post("/v1/chat/completions", json={
        "messages": USER, "max_tokens": 60, **HOT, "stream": True,
        "response_format": {"type": "json_object"},
        "stream_options": {"include_usage": True}})
    frames = _frames(r)
    text = "".join(f["choices"][0]["delta"].get("content") or "" for f in frames
                   if f.get("choices"))
    assert isinstance(json.loads(text), dict)
    assert [f["choices"][0]["finish_reason"] for f in frames if f.get("choices")][-1] == "stop"
    assert frames[-1]["usage"]["completion_tokens"] > 0


def test_logprobs_describe_the_constrained_tokens(client):
    """The tokens of the answer, with their log-probabilities under the raw logits."""
    choice = _chat(client, response_format={"type": "json_object"}, logprobs=True,
                   top_logprobs=2)["choices"][0]
    tokens = [e["token"] for e in choice["logprobs"]["content"]]
    assert "".join(tokens).strip() == choice["message"]["content"]
    assert all(e["logprob"] <= 0.0 for e in choice["logprobs"]["content"])


@pytest.mark.parametrize("fmt,code,text", [
    ({"type": "json_schema", "json_schema": {"name": "x", "schema": {
        "type": "object", "properties": {"a": {"type": "string", "pattern": "^a"}}}}},
     "invalid_json_schema", "pattern"),
    ({"type": "json_schema", "json_schema": {"name": "x"}}, "invalid_json_schema",
     "json_schema.schema"),
    ({"type": "json_schema", "json_schema": {"name": "x", "schema": PERSON}},
     "unsupported_parameter", "too small"),
])
def test_requests_the_grammar_cannot_serve_get_400(client, fmt, code, text):
    r = client.post("/v1/chat/completions", json={"messages": USER, "max_tokens": 10,
                                                   "response_format": fmt})
    assert r.status_code == 400
    error = r.json()["error"]
    assert error["param"] == "response_format" and error["code"] == code
    assert text in error["message"]


def test_a_vocabulary_without_braces_is_refused_before_generating(monkeypatch):
    torch.manual_seed(0)
    engine = AuraLiteEngine()
    engine.train("the quick brown fox jumps over the lazy dog " * 20, {**PARAMS, "max_steps": 2})
    monkeypatch.setattr(srv, "_engine", engine)
    monkeypatch.setattr(srv, "_rate_bucket", {})
    with TestClient(srv.app) as c:
        r = c.post("/v1/chat/completions", json={"messages": USER, "stream": True,
                                                  "response_format": {"type": "json_object"}})
    assert r.status_code == 400 and "no token for" in r.json()["error"]["message"]


def test_the_mask_alone_keeps_answers_valid(client, monkeypatch):
    """No token the model draws under the mask needs replacing (advance's safety net)."""
    replaced: list[int] = []
    original = S.JsonConstraint.advance

    def watching(self, token_id):
        taken = original(self, token_id)
        replaced.append(int(taken != int(token_id)))
        return taken

    monkeypatch.setattr(S.JsonConstraint, "advance", watching)
    _chat(client, response_format={"type": "json_object"}, n=3)
    _chat(client, tools=TOOLS, tool_choice="required", max_tokens=60)
    _chat(client, response_format={"type": "json_object"}, max_tokens=6)    # tight budget
    assert replaced and sum(replaced) == 0


def test_json_mode_never_falls_back_to_plain_text(client, native_engine, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("the prompt is too long")

    monkeypatch.setattr(native_engine, "generate_chat", broken)
    r = client.post("/v1/chat/completions", json={"messages": USER, "max_tokens": 40,
                                                   "response_format": {"type": "json_object"}})
    assert r.status_code == 400 and "too long" in r.json()["error"]["message"]


def test_cli_chat_json_schema(native_engine, tmp_path, capsys):
    from auralite import cli

    model = tmp_path / "m.pt"
    native_engine.save_model(str(model))
    schema = tmp_path / "person.json"
    schema.write_text(json.dumps(PERSON), encoding="utf-8")
    code = cli.main(["chat", "--model", str(model), "--json-schema", str(schema),
                     "--prompt", "Ann", "--length", "120", "--temperature", "1.0"])
    assert code == 0
    out = capsys.readouterr().out
    reply = out[out.index("assistant: ") + len("assistant: "):]     # (JSON may span lines)
    assert S.validate_json(json.loads(reply), PERSON) == []
    assert cli.main(["chat", "--model", str(model), "--json-schema", str(schema),
                     "--prompt", "Ann", "--length", "5"]) == 2      # too small for the schema


def test_finish_reason_says_why_an_answer_ended(client, native_engine):
    stats: dict = {}
    native_engine.generate_chat(USER, 3, temperature=0.0, stats=stats)
    assert stats == {"finish_reason": "length", "completion_tokens": 3}
    stats = {}
    text = native_engine.generate_chat(USER, 60, temperature=0.0, stop_tokens=['"'],
                                       stats=stats)
    assert stats["finish_reason"] == "stop" and '"' not in text
    stats = {}
    pieces = list(native_engine.generate_chat_streaming(USER, 3, temperature=0.0, stats=stats))
    assert pieces and stats["finish_reason"] == "length"
    body = _chat(client, max_tokens=3, temperature=0.0)
    assert body["choices"][0]["finish_reason"] == "length"


# ----------------------------------------------------------------------
#  tools
# ----------------------------------------------------------------------
def _check_call(call: dict) -> dict:
    assert call["type"] == "function" and call["id"].startswith("call_")
    args = json.loads(call["function"]["arguments"])
    params = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
    assert S.validate_json(args, params[call["function"]["name"]]) == []
    return args


def test_required_tool_choice_returns_a_valid_call(client):
    body = _chat(client, tools=TOOLS, tool_choice="required", n=3)
    for choice in body["choices"]:
        assert choice["finish_reason"] == "tool_calls"
        message = choice["message"]
        assert message["content"] is None and len(message["tool_calls"]) == 1
        _check_call(message["tool_calls"][0])


def test_a_named_tool_is_the_one_called(client):
    choice = _chat(client, tools=TOOLS, tool_choice={"type": "function",
                                                      "function": {"name": "get_time"}}
                   )["choices"][0]
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_time"
    assert "tz" in _check_call(choice["message"]["tool_calls"][0])


def test_auto_tool_choice_answers_or_calls(client):
    kinds = set()
    for seed in range(8):
        message = _chat(client, tools=TOOLS, seed=seed)["choices"][0]["message"]
        if message.get("tool_calls"):
            _check_call(message["tool_calls"][0])
            kinds.add("call")
        else:
            assert isinstance(message["content"], str)
            kinds.add("text")
    assert kinds                                      # (which ones depends on the model)


def test_tool_results_in_the_history_are_accepted(client, native_engine, monkeypatch):
    seen: list = []
    original = native_engine.generate_chat

    def spy(messages, *args, **kwargs):
        seen.append(messages)
        return original(messages, *args, **kwargs)

    monkeypatch.setattr(native_engine, "generate_chat", spy)
    history = USER + [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\": \"Riga\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 12}"}]
    r = client.post("/v1/chat/completions", json={"messages": history, "tools": TOOLS,
                                                   "max_tokens": 40, **HOT})
    assert r.status_code == 200, r.text
    turns = seen[0]
    assert turns[0]["role"] == "system" and "get_weather" in turns[0]["content"]
    assert json.loads(turns[2]["content"])["arguments"] == {"city": "Riga"}
    assert turns[3] == {"role": "user", "content": "Result of get_weather: {\"temp\": 12}"}


def test_streamed_tool_call_comes_in_openai_chunks(client):
    r = client.post("/v1/chat/completions", json={
        "messages": USER, "max_tokens": 80, **HOT, "stream": True, "tools": TOOLS,
        "tool_choice": "required"})
    frames = [f for f in _frames(r) if f.get("choices")]
    deltas = [f["choices"][0]["delta"] for f in frames]
    first = deltas[0]
    assert first["role"] == "assistant" and first["content"] is None
    head = first["tool_calls"][0]
    assert head["index"] == 0 and head["id"].startswith("call_") and head["type"] == "function"
    assert head["function"]["arguments"] == ""
    args = "".join(d["tool_calls"][0]["function"].get("arguments", "")
                   for d in deltas if d.get("tool_calls"))
    _check_call({"id": head["id"], "type": "function",
                 "function": {"name": head["function"]["name"], "arguments": args}})
    assert frames[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_legacy_functions_answer_with_function_call(client):
    functions = [t["function"] for t in TOOLS]
    choice = _chat(client, functions=functions, function_call={"name": "get_weather"}
                   )["choices"][0]
    assert choice["finish_reason"] == "function_call"
    call = choice["message"]["function_call"]
    assert call["name"] == "get_weather" and "city" in json.loads(call["arguments"])


@pytest.mark.parametrize("extra,param,code", [
    ({"tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "nope"}}},
     "tools", "invalid_value"),
    ({"tools": [{"type": "function", "function": {"name": "bad name"}}]}, "tools",
     "invalid_value"),
    ({"tools": [{"type": "function", "function": {"name": "f", "parameters": {
        "type": "object", "properties": {"x": {"type": "integer", "minimum": 0}}}}}]},
     "tools", "invalid_json_schema"),
])
def test_bad_tools_get_400(client, extra, param, code):
    r = client.post("/v1/chat/completions", json={"messages": USER, "max_tokens": 40, **extra})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == param and r.json()["error"]["code"] == code


# ----------------------------------------------------------------------
#  Ollama
# ----------------------------------------------------------------------
def test_ollama_format_json_and_schema(client):
    for fmt in ("json", PERSON):
        r = client.post("/api/chat", json={"messages": USER, "stream": False, "format": fmt,
                                           "options": {"temperature": 1.0, "num_predict": 80}})
        assert r.status_code == 200, r.text
        value = json.loads(r.json()["message"]["content"])
        if fmt is PERSON:
            assert S.validate_json(value, PERSON) == []
    r = client.post("/api/generate", json={"prompt": "{", "stream": False, "format": "json",
                                           "options": {"num_predict": 40}})
    assert r.status_code == 200 and isinstance(json.loads(r.json()["response"]), dict)
    r = client.post("/api/chat", json={"messages": USER, "stream": True, "format": "json",
                                       "options": {"num_predict": 40}})
    lines = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    assert isinstance(json.loads("".join(x["message"]["content"] for x in lines)), dict)


def test_ollama_tools_come_back_as_tool_calls(client):
    named = {"messages": USER, "tools": TOOLS[:1], "options": {"num_predict": 80}}
    for stream in (False, True):
        r = client.post("/api/chat", json={**named, "stream": stream})
        assert r.status_code == 200, r.text
        lines = [json.loads(line) for line in r.text.splitlines() if line.strip()]
        calls = [x["message"].get("tool_calls") for x in lines if x["message"].get("tool_calls")]
        texts = "".join(x["message"]["content"] for x in lines)
        if calls:                                         # auto: a call or an answer
            call = calls[0][0]["function"]
            assert call["name"] == "get_weather" and isinstance(call["arguments"], dict)
            assert texts == ""
        else:
            assert isinstance(texts, str)


# ----------------------------------------------------------------------
#  The official OpenAI client
# ----------------------------------------------------------------------
def test_the_official_openai_client_reads_json_and_tool_calls(client):
    openai = pytest.importorskip("openai")
    from pydantic import BaseModel

    api = openai.OpenAI(base_url="http://testserver/v1", api_key="sk-test", http_client=client)

    class Address(BaseModel):
        city: str

    class Person(BaseModel):
        name: str
        age: int
        email: str | None
        address: Address

    reply = api.chat.completions.create(model="auralite", messages=USER, max_tokens=60,
                                        temperature=1.0,
                                        response_format={"type": "json_object"})
    assert isinstance(json.loads(reply.choices[0].message.content or ""), dict)

    parsed = api.chat.completions.parse(model="auralite", messages=USER, max_tokens=120,
                                        temperature=1.0, response_format=Person)
    person = parsed.choices[0].message.parsed
    assert isinstance(person, Person) and isinstance(person.address, Address)

    class GetWeather(BaseModel):
        city: str

    called = api.chat.completions.parse(
        model="auralite", messages=USER, max_tokens=80, temperature=1.0,
        tools=[openai.pydantic_function_tool(GetWeather)], tool_choice="required")
    call = called.choices[0].message.tool_calls[0]
    assert call.function.name == "GetWeather"
    assert isinstance(call.function.parsed_arguments, GetWeather)
    assert called.choices[0].finish_reason == "tool_calls"

    chunks = list(api.chat.completions.create(
        model="auralite", messages=USER, max_tokens=80, temperature=1.0, stream=True,
        tools=TOOLS, tool_choice="required"))
    args = "".join(c.choices[0].delta.tool_calls[0].function.arguments or ""
                   for c in chunks if c.choices and c.choices[0].delta.tool_calls)
    assert isinstance(json.loads(args), dict)


# ----------------------------------------------------------------------
#  GGUF: llama.cpp gets the grammar
# ----------------------------------------------------------------------
class _FakeLlama:
    def __init__(self, **kwargs):
        self.calls: list[dict] = []
        self.metadata: dict = {}
        self.reply = '{"name": "get_weather", "arguments": {"city": "Riga"}}'

    def n_vocab(self):
        return 32

    def tokenize(self, data, add_bos=False, special=True):
        return list(range(len(data) // 4 + 1))

    def detokenize(self, ids):
        return b"x" * len(ids)

    def create_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"text": self.reply, "finish_reason": "stop"}]}

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": self.reply}, "finish_reason": "stop"}]}


@pytest.fixture()
def gguf_engine(monkeypatch):
    import sys
    import types

    module = types.ModuleType("llama_cpp")
    module.Llama = _FakeLlama            # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    monkeypatch.setattr(S, "llama_cpp_grammar", lambda g: ("grammar", g.key, g.allow_text))
    engine = AuraLiteEngine()
    engine.load_gguf_model("model.gguf")
    return engine


def test_gguf_models_get_a_llama_cpp_grammar(gguf_engine, monkeypatch):
    llama = gguf_engine.model.llama
    llama.reply = '{"a": 1}'
    text = gguf_engine.generate_chat(USER, 20, response_format={"type": "json_object"})
    assert json.loads(text) == {"a": 1}
    grammar = llama.calls[-1]["grammar"]
    assert grammar[0] == "grammar" and grammar[2] is False
    llama.reply = '{"name": "get_weather", "arguments": {"city": "Riga"}}'
    out = gguf_engine.chat_with_tools(USER, TOOLS, "auto", max_new_tokens=40)
    assert out["content"] is None
    assert json.loads(out["tool_calls"][0]["function"]["arguments"]) == {"city": "Riga"}
    assert llama.calls[-1]["grammar"][2] is True          # auto: a call or text
    monkeypatch.setattr(srv, "_engine", gguf_engine)
    monkeypatch.setattr(srv, "_rate_bucket", {})
    with TestClient(srv.app) as c:
        r = c.post("/v1/chat/completions", json={"messages": USER, "tools": TOOLS,
                                                  "tool_choice": "required", "max_tokens": 40})
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == \
            "get_weather"
        r = c.post("/api/chat", json={"messages": USER, "tools": TOOLS, "stream": False})
        assert r.json()["message"]["tool_calls"] == [
            {"function": {"name": "get_weather", "arguments": {"city": "Riga"}}}]
        llama.reply = '{"a": "a string llama.cpp did not get to close'
        r = c.post("/v1/chat/completions", json={"messages": USER, "max_tokens": 8,
                                                  "response_format": {"type": "json_object"}})
        assert r.json()["choices"][0]["finish_reason"] == "length"
        llama.reply = "Hello there."
        r = c.post("/v1/chat/completions", json={"messages": USER, "max_tokens": 8})
        assert r.json()["choices"][0]["finish_reason"] == "stop"     # llama.cpp said so


def test_llama_server_gets_the_json_schema():
    from auralite.engine.gguf_proxy import LlamaServerProxy

    proxy = LlamaServerProxy.__new__(LlamaServerProxy)
    proxy._props = {}
    sent: list = []

    def request(method, path, payload=None):
        sent.append((path, payload))
        return {"choices": [{"message": {"content": "{}"}}], "content": "{}"}

    proxy._request = request
    proxy.create_chat_completion(USER, json_schema={"type": "object"})
    proxy.create_completion("x", json_schema={"type": "object"})
    assert sent[0][1]["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert sent[1][1]["json_schema"] == {"type": "object"}


def test_hugging_face_models_are_refused(monkeypatch):
    engine = AuraLiteEngine()
    monkeypatch.setattr(engine, "model", object(), raising=False)
    monkeypatch.setattr(engine, "is_hf_model", lambda: True)
    with pytest.raises(ValueError, match="Hugging Face"):
        engine.generate_chat(USER, 10, response_format={"type": "json_object"})
