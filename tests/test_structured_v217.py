"""JSON mode, JSON schemas and tool calls: the grammar and the masks (2.17).

`auralite/engine/structured.py` compiles a schema into a byte-level automaton
and masks the tokens a model may draw. These tests drive the automaton byte by
byte and token by token, with random choices instead of a model: whatever is
drawn, the answer must parse, follow the schema and fit into its budget.
"""
from __future__ import annotations

import json
import random

import pytest

from auralite.engine import structured as S

PERSON = {
    "type": "object",
    "title": "Person", "description": "annotations are ignored",
    "properties": {
        "name": {"type": "string", "description": "the name"},
        "age": {"type": "integer"},
        "email": {"type": ["string", "null"], "format": "email"},
        "tags": {"type": "array", "items": {"enum": ["a", "b", "c"]}, "maxItems": 3},
        "role": {"enum": ["admin", "user", 3, None, True]},
        "score": {"type": "number", "default": 1.5},
        "address": {"type": "object", "required": ["city"], "properties": {
            "city": {"type": "string", "minLength": 2, "maxLength": 5},
            "zip": {"type": "string"}}},
    },
    "required": ["name", "age", "address"],
    "additionalProperties": False,
}
TREE = {"$defs": {"node": {"type": "object", "required": ["v"], "properties": {
    "v": {"type": "integer"},
    "kids": {"type": "array", "items": {"$ref": "#/$defs/node"}}}}},
    "$ref": "#/$defs/node"}
LIST = {"type": "array", "minItems": 2, "maxItems": 4,
        "items": {"type": "object", "properties": {"x": {"type": "boolean"}}, "required": ["x"]}}
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather now",
                                      "parameters": {"type": "object", "required": ["city"],
                                                     "properties": {
                                                         "city": {"type": "string"},
                                                         "unit": {"enum": ["c", "f"]}}}}},
    {"type": "function", "function": {"name": "get_time", "parameters": {
        "type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]}}},
    {"type": "function", "function": {"name": "noop"}},
]
POOL = list(b' \n\t{}[]":,-.0123456789eE+abcdefghijklmnopqrstuvwxyzABCXYZ_\\/') \
    + [0xC3, 0xA9, 0xE2, 0x82, 0xAC]


def _walk(grammar: S.JsonGrammar, budget: int, rng: random.Random) -> bytes:
    """Random bytes the grammar allows, as a model under the mask would draw them."""
    st, out, left = grammar.initial, bytearray(), budget
    assert grammar.cost(st) <= budget
    while not grammar.done(st) and not grammar.free(st):
        if rng.random() < 0.05:                       # the cost is the shortest ending
            assert len(grammar.ending(st)) == grammar.cost(st), bytes(out)
        options = []
        for b in POOL:
            nxt = grammar.feed(st, b)
            if nxt is not None and grammar.cost(nxt) <= left - 1:
                options.append((b, nxt))
        assert options, bytes(out)
        b, st = rng.choice(options)
        out.append(b)
        left -= 1
    return bytes(out)


# ----------------------------------------------------------------------
#  Schemas
# ----------------------------------------------------------------------
def test_schema_grammar_accepts_valid_and_refuses_invalid_answers():
    g = S.compile_schema(PERSON)
    assert g.accepts('{"name": "Al", "age": 34, "address": {"city": "Riga"}}')
    assert g.accepts('{"name":"Al","age":-3,"email":null,"tags":["a","c"],"role":3,'
                     '"score":1.5e-3,"address":{"city":"Os","zip":"0150"}}')
    assert g.accepts('{"name":"\\u00e9\\n","age":0,"role":null,"address":{"city":"Ri"}}')
    for wrong in [
        '{"age": 3, "name": "x", "address": {"city": "Riga"}}',    # keys out of order
        '{"name": "x", "age": 3.5, "address": {"city": "Riga"}}',  # integer
        '{"name": "x", "age": 1, "address": {"city": "R"}}',       # minLength
        '{"name": "x", "age": 1, "address": {"city": "Riga!!"}}',  # maxLength
        '{"name": "x", "age": 1}',                                 # required
        '{"name": "x", "age": 1, "address": {"city": "Ri"}, "zzz": 1}',
        '{"name": "x", "age": 1, "role": "root", "address": {"city": "Ri"}}',
        '{"name": "x", "age": 1, "tags": ["a","a","a","a"], "address": {"city": "Ri"}}',
        '{"name": "x", "age": 01, "address": {"city": "Ri"}}',
        '{"name": "x\x01", "age": 1, "address": {"city": "Ri"}}',
        '{"name": "x", "name": "y", "age": 1, "address": {"city": "Ri"}}',   # twice
    ]:
        assert not g.accepts(wrong), wrong


def test_numbers_strings_and_utf8():
    g = S.compile_schema({"type": "array", "items": {"type": "number"}})
    assert g.accepts("[-0, 0.5, 1e10, -12.5E-3, 7]")
    for wrong in ("[01]", "[1.]", "[.5]", "[+1]", "[1e]", "[--1]"):
        assert not g.accepts(wrong), wrong
    ints = S.compile_schema({"type": "array", "items": {"type": "integer"}})
    assert ints.accepts("[1, -20]") and not ints.accepts("[1.5]") and not ints.accepts("[1e3]")
    s = S.compile_schema({"type": "array", "items": {"type": "string"}})
    assert s.accepts('["é€", "a\\"b", "\\u20AC", "\\/"]')
    st = s.feed_bytes(s.initial, b'["')
    assert st is not None
    assert s.feed(st, 0xFF) is None and s.feed(st, 0x80) is None    # not a lead byte
    lead = s.feed(st, 0xC3)
    assert lead is not None and s.feed(lead, 0x41) is None           # continuation needed
    assert s.feed(st, 0x0A) is None                                  # raw newline


def test_unsupported_keywords_are_refused_not_ignored():
    for keyword, value in [("pattern", "^a"), ("minimum", 0), ("uniqueItems", True),
                           ("allOf", [{}]), ("not", {})]:
        with pytest.raises(S.SchemaError, match=keyword):
            S.compile_schema({"type": "object", "properties": {"x": {"type": "string",
                                                                    keyword: value}}})
    with pytest.raises(S.SchemaError, match="root"):
        S.compile_schema({"type": "string"})
    with pytest.raises(S.SchemaError, match="start alike"):
        S.compile_schema({"type": "object", "properties": {"x": {"anyOf": [
            {"type": "object"}, {"type": "object", "properties": {"y": {}}}]}}})
    with pytest.raises(S.SchemaError, match="finite"):
        S.compile_schema({"$defs": {"n": {"type": "object", "required": ["n"],
                                          "properties": {"n": {"$ref": "#/$defs/n"}}}},
                          "$ref": "#/$defs/n"})
    with pytest.raises(S.SchemaError, match="points nowhere"):
        S.compile_schema({"type": "object", "properties": {"x": {"$ref": "#/$defs/no"}}})


def test_nullable_unions_refs_and_free_objects():
    g = S.compile_schema({"type": "object", "properties": {
        "a": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "b": {"type": "string", "nullable": True},
        "c": {"type": "object", "additionalProperties": {"type": "integer"}},
        "d": {"enum": ["x", {"k": [1]}]}}, "required": ["a", "b", "c", "d"]})
    assert g.accepts('{"a": null, "b": "q", "c": {"k1": 1, "k2": -2}, "d": {"k":[1]}}')
    assert not g.accepts('{"a": 1, "b": "q", "c": {}, "d": "x"}')
    assert not g.accepts('{"a": null, "b": "q", "c": {"k": "s"}, "d": "x"}')
    tree = S.compile_schema(TREE)
    assert tree.accepts('{"v": 1, "kids": [{"v": 2}, {"v": 3, "kids": [{"v": 4}]}]}')
    assert not tree.accepts('{"kids": []}')


def test_shortest_endings_complete_any_prefix():
    g = S.compile_schema(PERSON)
    assert g.ending(g.initial) == b'{"name":"","age":0,"address":{"city":"aa"}}'
    for prefix in (b'{"name": "Ann', b'{"name":"A","age":-', b'{"name":"A","age":1,"ta',
                   b'{"name":"A","age":1,"tags":["b",', b'{"name":"\\u00'):
        st = g.feed_bytes(g.initial, prefix)
        assert st is not None, prefix
        text = (prefix + g.ending(st)).decode()
        assert not S.validate_json(json.loads(text), PERSON), text
        assert len(g.ending(st)) == g.cost(st)
    assert S.compile_schema(LIST).ending(S.compile_schema(LIST).initial) == \
        b'[{"x":true},{"x":true}]'
    assert S.json_object_grammar().cost(S.json_object_grammar().initial) == 2


@pytest.mark.parametrize("name", ["person", "tree", "list", "json_object", "required",
                                  "named", "auto"])
def test_random_answers_always_parse_follow_the_schema_and_fit(name):
    tools = S.normalize_tools(TOOLS)
    grammar, schema = {
        "person": (S.compile_schema(PERSON), PERSON),
        "tree": (S.compile_schema(TREE), TREE),
        "list": (S.compile_schema(LIST), LIST),
        "json_object": (S.json_object_grammar(), {"type": "object"}),
        "required": (S.tool_grammar(tools, "required"), None),
        "named": (S.tool_grammar(tools, "required", "get_time"), None),
        "auto": (S.tool_grammar(tools, "auto"), None),
    }[name]
    rng = random.Random(7)
    least = grammar.cost(grammar.initial)
    for trial in range(120):
        budget = least + rng.choice([0, 1, 5, 40, 150])
        data = _walk(grammar, budget, rng)
        assert len(data) <= budget
        if name == "auto" and not data.lstrip().startswith(b"{"):
            continue                                    # an ordinary answer
        value = json.loads(data.decode("utf-8"))
        if schema is not None:
            assert not S.validate_json(value, schema), data
            continue
        assert list(value) == ["name", "arguments"], data
        tool = {t["name"]: t for t in tools}[value["name"]]
        assert not S.validate_json(value["arguments"], S._as_object_schema(tool["parameters"]))
        if name == "named":
            assert value["name"] == "get_time"
        del trial


# ----------------------------------------------------------------------
#  response_format and tools
# ----------------------------------------------------------------------
def test_response_format_variants():
    assert S.grammar_for_response_format(None) is None
    assert S.grammar_for_response_format({"type": "text"}) is None
    assert S.grammar_for_response_format({"type": "json_object"}) is S.json_object_grammar()
    assert S.grammar_for_response_format("json") is S.json_object_grammar()
    g = S.grammar_for_response_format({"type": "json_schema",
                                       "json_schema": {"name": "p", "schema": PERSON}})
    assert g is S.compile_schema(PERSON)                 # compiled once (cache)
    with pytest.raises(S.SchemaError, match="json_schema.schema"):
        S.grammar_for_response_format({"type": "json_schema", "json_schema": {"name": "x"}})
    with pytest.raises(S.SchemaError, match="unknown response_format"):
        S.grammar_for_response_format({"type": "yaml"})


def test_tools_are_checked_and_tool_choice_resolved():
    tools = S.normalize_tools(TOOLS)
    assert [t["name"] for t in tools] == ["get_weather", "get_time", "noop"]
    assert tools[2]["parameters"] == {"type": "object", "properties": {}}
    assert S.resolve_tool_choice(None, tools) == ("auto", None)
    assert S.resolve_tool_choice("required", tools) == ("required", None)
    assert S.resolve_tool_choice({"type": "function", "function": {"name": "noop"}},
                                 tools) == ("required", "noop")
    assert S.resolve_tool_choice({"name": "noop"}, tools) == ("required", "noop")  # legacy
    assert S.resolve_tool_choice("auto", []) == ("none", None)
    with pytest.raises(S.SchemaError, match="not among"):
        S.resolve_tool_choice({"type": "function", "function": {"name": "x"}}, tools)
    with pytest.raises(S.SchemaError, match="twice"):
        S.normalize_tools(TOOLS + TOOLS[:1])
    with pytest.raises(S.SchemaError, match="tool name"):
        S.normalize_tools([{"type": "function", "function": {"name": "a b"}}])
    assert S.normalize_tools(None, [{"name": "f", "parameters": {"type": "object"}}])[0][
        "name"] == "f"


def test_tool_grammar_lets_the_name_choose_the_arguments():
    tools = S.normalize_tools(TOOLS)
    g = S.tool_grammar(tools)
    assert g.accepts('{"name": "get_time", "arguments": {"tz": "UTC"}}')
    assert g.accepts('{"name":"noop","arguments":{}}')
    assert not g.accepts('{"name": "get_time", "arguments": {"city": "Riga"}}')
    assert not g.accepts('{"name": "rm_rf", "arguments": {}}')
    assert not g.accepts('Sure!')
    auto = S.tool_grammar(tools, "auto")
    assert auto.accepts("Sure! It is sunny.") and auto.allow_text
    st = auto.feed_bytes(auto.initial, b"  {")
    assert st is not None and not auto.free(st)
    for prefix in (b'{"name":"get_', b'{"name": "get_t', b'{"name":"noop","argu'):
        st = g.feed_bytes(g.initial, prefix)
        assert st is not None and len(g.ending(st)) == g.cost(st), prefix
        assert g.accepts((prefix + g.ending(st)).decode())
    named = S.tool_grammar(tools, "required", "get_weather")
    assert not named.accepts('{"name":"noop","arguments":{}}')
    assert named.ending(named.initial) == b'{"name":"get_weather","arguments":{"city":""}}'


def test_messages_with_tool_calls_and_results_become_plain_turns():
    tools = S.normalize_tools(TOOLS)
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": [{"type": "text", "text": "Weather "},
                                     {"type": "text", "text": "in Riga?"}]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city": "Riga"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 12}'},
        {"role": "developer", "content": "Answer in Celsius."},
    ]
    plain = S.plain_messages(messages, tools, "auto")
    assert [m["role"] for m in plain] == ["system", "user", "assistant", "user", "system"]
    assert plain[0]["content"].startswith("Be brief.\n\n# Tools")
    assert '"get_weather"' in plain[0]["content"] and "answer normally" in plain[0]["content"]
    assert plain[1]["content"] == "Weather in Riga?"
    assert json.loads(plain[2]["content"]) == {"name": "get_weather",
                                               "arguments": {"city": "Riga"}}
    assert plain[3]["content"] == 'Result of get_weather: {"temp": 12}'
    assert S.plain_messages([{"role": "user", "content": "hi"}]) == [
        {"role": "user", "content": "hi"}]
    named = S.plain_messages([{"role": "user", "content": "x"}], tools, "required", "noop")
    assert named[0]["role"] == "system" and "Call the function noop now." in named[0]["content"]
    with pytest.raises(S.SchemaError, match="image_url"):
        S.plain_messages([{"role": "user", "content": [{"type": "image_url"}]}])


def test_parse_tool_call_is_lenient_with_free_text_but_checks_the_name():
    tools = S.normalize_tools(TOOLS)
    call = {"name": "get_time", "arguments": {"tz": "UTC"}}
    for text in ('{"name": "get_time", "arguments": {"tz": "UTC"}}',
                 '<tool_call>\n{"name": "get_time", "arguments": {"tz": "UTC"}}\n</tool_call>',
                 '```json\n{"name": "get_time", "parameters": {"tz": "UTC"}}\n```',
                 '{"name": "get_time", "arguments": "{\\"tz\\": \\"UTC\\"}"} trailing'):
        assert S.parse_tool_call(text, tools) == call, text
    for text in ("It is noon.", '{"name": "rm", "arguments": {}}', "{broken", "[1]"):
        assert S.parse_tool_call(text, tools) is None, text
    out = S.openai_tool_calls(call)
    assert out[0]["type"] == "function" and out[0]["id"].startswith("call_")
    assert json.loads(out[0]["function"]["arguments"]) == {"tz": "UTC"}


def test_tool_call_schema_for_llama_cpp_hoists_defs():
    tools = S.normalize_tools([{"type": "function", "function": {"name": "add", "parameters": {
        "$defs": {"Item": {"type": "object", "properties": {"n": {"type": "integer"}}}},
        "type": "object", "properties": {"items": {"type": "array",
                                                   "items": {"$ref": "#/$defs/Item"}}}}}}])
    schema = S.tool_call_schema(tools)
    assert schema["properties"]["name"] == {"const": "add"}
    assert schema["properties"]["arguments"]["properties"]["items"]["items"] == {
        "$ref": "#/$defs/t0_Item"}
    assert "t0_Item" in schema["$defs"] and "$defs" not in schema["properties"]["arguments"]
    both = S.tool_call_schema(S.normalize_tools(TOOLS))
    assert [a["properties"]["name"]["const"] for a in both["anyOf"]] == [
        "get_weather", "get_time", "noop"]


def test_validate_json_reports_problems():
    assert S.validate_json({"name": "x", "age": 1, "address": {"city": "Ri"}}, PERSON) == []
    errors = S.validate_json({"name": 1, "age": True, "address": {}}, PERSON)
    assert any("name" in e for e in errors) and any("age" in e for e in errors)
    assert any("missing 'city'" in e for e in errors)


# ----------------------------------------------------------------------
#  Tokens: tables, masks, constraints
# ----------------------------------------------------------------------
torch = pytest.importorskip("torch")


def _char_tokenizer(text: str, specials=("<|im_start|>", "<|im_end|>")):
    from auralite.engine.tokenizers import CharTokenizer

    tok = CharTokenizer()
    tok.train(text)
    tok.add_special_tokens(list(specials))
    return tok


def _bpe_tokenizer(text: str):
    from auralite.engine.tokenizers import BPETokenizer

    tok = BPETokenizer()
    tok.train(text, vocab_size=300)
    tok.add_special_tokens(["<|im_end|>"])
    return tok


JSONISH = ('{"name": "get_weather", "arguments": {"city": "Riga", "unit": "c"}} '
           '{"name": "Ann", "age": 34, "tags": ["a", "b", "c"], "x": null, "y": true, '
           '"z": false, "q": -1.5e3, "address": {"city": "Oslo", "zip": "0150"}, '
           '"get_time": "tz", "role": "admin", "email": "w@x.y", "kids": [], "v": 9} '
           'ABCXYZ_\\/ noop user score') * 3


def test_token_tables_leave_out_special_and_unknown_tokens():
    tok = _char_tokenizer(JSONISH)
    table = S.TokenTable.for_tokenizer(tok)
    ids = tok.encode("<|im_end|>")
    assert len(ids) == 1 and table.token_bytes[ids[0]] is None
    assert table.single[ord("{")] == tok.encode("{")[0]
    assert table.missing(S.compile_schema(PERSON)) == []
    plain = S.TokenTable.for_tokenizer(_char_tokenizer("abc def"))
    missing = plain.missing(S.compile_schema(PERSON))
    assert "{" in missing and '"' in missing and any(m.startswith("one of") for m in missing)
    bpe = S.TokenTable.for_tokenizer(_bpe_tokenizer(JSONISH))
    assert any(data and len(data) > 1 for data in bpe.token_bytes)   # merged tokens


def test_costs_vector_masks_refused_tokens_and_is_cached():
    tok = _char_tokenizer(JSONISH)
    table = S.TokenTable.for_tokenizer(tok)
    g = S.compile_schema(PERSON)
    costs = table.costs(g, g.initial)
    allowed = {tok.decode([i]) for i in range(len(costs)) if int(costs[i]) < S.COST_REFUSED}
    assert allowed == {"{", " "}                       # (no newline in this vocabulary)
    assert table.costs(g, g.initial) is costs and table.hits == 1
    st = g.feed_bytes(g.initial, b'{"name": "A')
    assert st is not None
    costs = table.costs(g, st)
    after_quote = g.feed(st, 0x22)
    assert after_quote is not None
    assert int(costs[tok.encode('"')[0]]) == g.cost(after_quote)
    assert int(costs[tok.encode("<|im_end|>")[0]]) == S.COST_REFUSED


@pytest.mark.parametrize("which", ["char", "bpe"])
def test_random_logits_under_the_constraint_always_give_valid_json(which):
    tok = _char_tokenizer(JSONISH) if which == "char" else _bpe_tokenizer(JSONISH)
    table = S.TokenTable.for_tokenizer(tok)
    gen = torch.Generator().manual_seed(3)
    tools = S.normalize_tools(TOOLS)
    for grammar, schema in [(S.compile_schema(PERSON), PERSON), (S.compile_schema(TREE), TREE),
                            (S.json_object_grammar(), {"type": "object"}),
                            (S.tool_grammar(tools, "required"), None)]:
        least = grammar.cost(grammar.initial)
        for budget in (least, least + 3, least + 25, 200):
            c = S.JsonConstraint(grammar, table, budget)
            ids: list[int] = []
            while not c.done:
                assert c.remaining > 0
                logits = torch.randn(table.size + 3, generator=gen) * 4
                masked = c.mask(logits)
                probs = torch.softmax(masked, dim=-1)
                drawn = int(torch.multinomial(probs, 1, generator=gen))
                ids.append(c.advance(drawn))
                assert ids[-1] == drawn                # the mask alone kept it valid
            text = tok.decode(ids)
            assert len(ids) <= budget
            value = json.loads(text)
            if schema is not None:
                assert not S.validate_json(value, schema), text
            else:
                assert value["name"] in {"get_weather", "get_time", "noop"}


def test_a_refused_token_is_replaced_by_one_of_the_shortest_ending():
    tok = _char_tokenizer(JSONISH)
    table = S.TokenTable.for_tokenizer(tok)
    g = S.compile_schema(PERSON)
    c = S.JsonConstraint(g, table, 200)
    taken = c.advance(tok.encode("x")[0])              # not allowed before "{"
    assert tok.decode([taken]) == "{"
    c = S.JsonConstraint(g, table, g.cost(g.initial))   # no room to spare at all
    out = [c.advance(tok.encode("{")[0])]
    while not c.done:
        out.append(c.advance(tok.encode(" ")[0]))     # spaces fit only inside the city
    assert tok.decode(out) == '{"name":"","age":0,"address":{"city":"  "}}'


def test_llama_cpp_grammar_keeps_control_characters_out_and_allows_text():
    pytest.importorskip("llama_cpp")
    gbnf = S.llama_cpp_gbnf(S.json_object_grammar())
    assert r'char ::= [^"\\\x7F\x00-\x1F]' in gbnf
    auto = S.llama_cpp_gbnf(S.tool_grammar(S.normalize_tools(TOOLS), "auto"))
    assert "root ::= auralite-call | auralite-text" in auto and "auralite-call ::=" in auto
