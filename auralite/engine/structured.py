"""Structured output: JSON mode, JSON schemas and tool calls (2.17).

A native AuraLite model writes JSON through *constrained decoding*. A JSON
schema (or plain JSON mode, or a tool call) compiles into a
:class:`JsonGrammar`: a byte-level pushdown automaton. A :class:`TokenTable`
holds the bytes of every token of the model's tokenizer in a prefix tree, and
at every step a :class:`JsonConstraint` gives ``-inf`` to the tokens whose
bytes the automaton refuses.

The automaton also knows how many bytes the shortest valid ending of the
answer still needs (:meth:`JsonGrammar.cost`). A token is only allowed while
that ending still fits into the tokens that remain, so the answer is complete
JSON, valid against the schema, whatever the model, the sampling settings and
``max_tokens`` (as long as the shortest valid answer fits at all). The masks
are cached per automaton state: inside a long string the state repeats, and
so does the work.

Supported schema keywords: ``type`` (object, array, string, number, integer,
boolean, null, or a list of them), ``properties``, ``required``,
``additionalProperties``, ``items``, ``enum``, ``const``, ``anyOf`` /
``oneOf`` (alternatives that start differently: an object, a string, a
number, ...), ``minLength`` / ``maxLength``, ``minItems`` / ``maxItems``,
``nullable`` and local ``$ref`` (``#/$defs/...``, recursion included).
Annotations (``title``, ``description``, ``default``, ``format``, ...) are
ignored, as JSON Schema says. Keywords that would need more than a grammar
(``pattern``, ``minimum``, ``uniqueItems``, ...) are refused with a
:class:`SchemaError` instead of being ignored. Object keys come in the
schema's order, optional ones may be left out (OpenAI's structured outputs
order them the same way).

Tools (OpenAI ``tools`` / ``tool_choice``) become a system message that lists
them (:func:`tools_system_prompt`) and a grammar for one call
``{"name": ..., "arguments": {...}}``: the name is one of the tools and the
arguments follow that tool's ``parameters``. With ``tool_choice="auto"`` the
model decides with its first character: ``{`` starts a call, anything else is
an ordinary answer.

GGUF models get the same JSON schema as a llama.cpp grammar
(:func:`llama_cpp_grammar`).
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from collections import OrderedDict
from typing import Any, Iterator

__all__ = [
    "SchemaError", "JsonGrammar", "TokenTable", "JsonConstraint", "StructuredMixin",
    "compile_schema", "json_object_grammar", "grammar_for_response_format",
    "tool_grammar", "normalize_tools", "resolve_tool_choice", "tools_system_prompt",
    "plain_messages", "tool_call_schema", "parse_tool_call", "openai_tool_calls",
    "llama_cpp_grammar", "validate_json", "token_byte_table",
]

#: Whitespace bytes allowed between JSON tokens, and how many in a row.
WS_MAX = 8
#: Nesting depth of free-form JSON (JSON mode); deeper values are scalars.
ANY_DEPTH = 8
#: Longest number (bytes) before only its end is allowed.
NUM_MAX = 24
#: Cost of a token the automaton refuses.
INF = 1 << 30
#: Masks store costs as int16: refused tokens get COST_REFUSED, others at most one less.
COST_REFUSED = 32767
#: Automaton states whose masks a token table keeps (LRU), and the most mask
#: entries in all (a large vocabulary keeps fewer states).
CACHE_STATES = 2048
CACHE_ENTRIES = 16_000_000

_WS = frozenset(b" \n\t")
_HEX = frozenset(b"0123456789abcdefABCDEF")
_ESCAPES = frozenset(b'"\\/bfnrt')

# frame tags of the automaton's stack
_VAL, _OBJ, _KEY, _ARR, _STR, _NUM, _LIT, _DONE, _TEXT = range(9)
# object phases
_P_START, _P_KEY, _P_COLON, _P_VALUE, _P_AFTER, _P_COMMA = range(6)
# array phases
_A_START, _A_VALUE, _A_AFTER, _A_COMMA = range(4)
# number states
_N_SIGN, _N_ZERO, _N_INT, _N_DOT, _N_FRAC, _N_E, _N_ESIGN, _N_EXP = range(8)
_NUM_ACCEPT = frozenset((_N_ZERO, _N_INT, _N_FRAC, _N_EXP))
# node kinds
_K_OBJ, _K_ARR, _K_STR, _K_NUM, _K_LIT, _K_UNION, _K_AUTO, _K_REF = range(1, 9)

_DONE_STACK: tuple = ((_DONE,),)
_TEXT_STACK: tuple = ((_TEXT,),)

_ANNOTATIONS = frozenset({
    "title", "description", "default", "examples", "$schema", "$id", "$comment",
    "deprecated", "readOnly", "writeOnly", "format", "contentMediaType",
    "contentEncoding", "discriminator", "$defs", "definitions", "$anchor", "strict",
})
_UNSUPPORTED = frozenset({
    "pattern", "patternProperties", "propertyNames", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "uniqueItems", "contains",
    "minContains", "maxContains", "prefixItems", "unevaluatedProperties",
    "unevaluatedItems", "dependentRequired", "dependentSchemas", "dependencies", "if",
    "then", "else", "not", "allOf", "minProperties", "maxProperties", "$dynamicRef",
    "$recursiveRef",
})


class SchemaError(ValueError):
    """A JSON schema (or tool definition) that structured output cannot follow."""


def _bits(mask: int) -> Iterator[int]:
    i = 0
    while mask:
        if mask & 1:
            yield i
        mask >>= 1
        i += 1


def _literal_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class _Trie:
    """Prefix tree of a few byte strings (object keys, enum values)."""

    __slots__ = ("children", "end", "mask", "depth", "lengths", "minrem")

    def __init__(self, words: list[bytes]):
        self.children: list[dict[int, int]] = [{}]
        self.end = [-1]
        self.mask = [0]
        self.depth = [0]
        self.lengths = [len(w) for w in words]
        for i, word in enumerate(words):
            node = 0
            self.mask[0] |= 1 << i
            for b in word:
                nxt = self.children[node].get(b)
                if nxt is None:
                    nxt = len(self.children)
                    self.children.append({})
                    self.end.append(-1)
                    self.mask.append(0)
                    self.depth.append(self.depth[node] + 1)
                    self.children[node][b] = nxt
                node = nxt
                self.mask[node] |= 1 << i
            self.end[node] = i
        self.minrem = [min(self.lengths[j] for j in _bits(m)) - d if m else 0
                       for m, d in zip(self.mask, self.depth, strict=True)]


class _Node:
    """One compiled schema node (see the ``_K_*`` kinds)."""

    __slots__ = ("kind", "keys", "vals", "req", "free", "trie", "dyn", "item", "min_items",
                 "max_items", "min_len", "max_len", "track", "integer", "cands", "alts",
                 "dispatch", "call", "target", "allowed", "need", "rest", "comma",
                 "key_total", "start_cost", "na_min")

    def __init__(self, kind: int):
        self.kind = kind
        self.keys: list[bytes] = []
        self.vals: list[int] = []
        self.req: list[bool] = []
        self.free = -1
        self.trie: _Trie | None = None
        self.dyn: list[int] | None = None
        self.item = -1
        self.min_items = 0
        self.max_items = INF
        self.min_len = 0
        self.max_len = INF
        self.track = 0
        self.integer = False
        self.cands: list[bytes] = []
        self.alts: list[int] = []
        self.dispatch: dict[int, int] = {}
        self.call = -1
        self.target = -1
        self.allowed: list[int] = []
        self.need: list[bool] = []
        self.rest: list[int] = []
        self.comma: list[int] = []
        self.key_total: list[int] = []
        self.start_cost = 0
        self.na_min = 0


# ----------------------------------------------------------------------
#  Compiling schemas
# ----------------------------------------------------------------------
class _Compiler:
    def __init__(self, root_schema: Any):
        self.root_schema = root_schema
        self.nodes: list[_Node] = []
        self.refs: dict[str, int] = {}
        self.any_nodes: dict[int, int] = {}
        self.key_string = -1

    def new(self, kind: int) -> int:
        self.nodes.append(_Node(kind))
        return len(self.nodes) - 1

    # -- schema -> node ---------------------------------------------------
    def compile(self, schema: Any, path: str, depth: int = 0) -> int:
        if schema is True or schema == {}:
            return self.any_value(depth)
        if schema is False:
            raise SchemaError(f"{path}: the schema false accepts no value")
        if not isinstance(schema, dict):
            raise SchemaError(f"{path}: a schema must be an object, not {type(schema).__name__}")
        bad = sorted(k for k in schema if k in _UNSUPPORTED)
        if bad:
            raise SchemaError(f"{path}: the JSON schema keyword {bad[0]!r} is not supported "
                              "by structured output (supported: type, properties, required, "
                              "additionalProperties, items, enum, const, anyOf, oneOf, "
                              "minLength, maxLength, minItems, maxItems, $ref)")
        if "$ref" in schema:
            others = [k for k in schema if k != "$ref" and k not in _ANNOTATIONS
                      and not k.startswith("x-")]
            if others:
                raise SchemaError(f"{path}: $ref next to {others[0]!r} is not supported")
            node = self.ref(str(schema["$ref"]), path)
        elif "const" in schema:
            node = self.literal([schema["const"]], path)
        elif "enum" in schema:
            values = schema["enum"]
            if not isinstance(values, list) or not values:
                raise SchemaError(f"{path}: enum must be a non-empty list")
            node = self.literal(list(values), path)
        elif "anyOf" in schema or "oneOf" in schema:
            word = "anyOf" if "anyOf" in schema else "oneOf"
            alts = schema[word]
            if not isinstance(alts, list) or not alts:
                raise SchemaError(f"{path}: {word} must be a non-empty list")
            others = [k for k in schema if k not in (word, "nullable") and k not in _ANNOTATIONS
                      and not k.startswith("x-")]
            if others:
                raise SchemaError(f"{path}: {word} next to {others[0]!r} is not supported")
            node = self.union([self.compile(a, f"{path}/{word}/{i}", depth)
                               for i, a in enumerate(alts)])
        else:
            kind = schema.get("type")
            if kind is None:
                if "properties" in schema or "additionalProperties" in schema \
                        or "required" in schema:
                    kind = "object"
                elif "items" in schema or "minItems" in schema or "maxItems" in schema:
                    kind = "array"
                elif "minLength" in schema or "maxLength" in schema:
                    kind = "string"
            if kind is None:
                node = self.any_value(depth)
            elif isinstance(kind, list):
                if not kind:
                    raise SchemaError(f"{path}: type must not be an empty list")
                node = self.union([self.typed(str(k), schema, path, depth) for k in kind])
            else:
                node = self.typed(str(kind), schema, path, depth)
        if schema.get("nullable") is True:
            node = self.union([node, self.literal([None], path)])
        return node

    def typed(self, kind: str, schema: dict, path: str, depth: int) -> int:
        if kind == "object":
            return self.object(schema, path, depth)
        if kind == "array":
            items = schema.get("items", True)
            if isinstance(items, list):
                raise SchemaError(f"{path}: items as a list (tuples) is not supported")
            nid = self.new(_K_ARR)
            self.nodes[nid].item = self.compile(items, f"{path}/items", depth + 1)
            self.nodes[nid].min_items = self._count(schema, "minItems", 0, path)
            self.nodes[nid].max_items = self._count(schema, "maxItems", INF, path)
            if self.nodes[nid].min_items > self.nodes[nid].max_items:
                raise SchemaError(f"{path}: minItems is larger than maxItems")
            return nid
        if kind == "string":
            nid = self.new(_K_STR)
            node = self.nodes[nid]
            node.min_len = self._count(schema, "minLength", 0, path)
            node.max_len = self._count(schema, "maxLength", INF, path)
            if node.min_len > node.max_len:
                raise SchemaError(f"{path}: minLength is larger than maxLength")
            node.track = 1 if (node.min_len > 0 or node.max_len < INF) else 0
            return nid
        if kind in ("number", "integer"):
            nid = self.new(_K_NUM)
            self.nodes[nid].integer = kind == "integer"
            return nid
        if kind == "boolean":
            return self.literal([True, False], path)
        if kind == "null":
            return self.literal([None], path)
        raise SchemaError(f"{path}: unknown type {kind!r}")

    @staticmethod
    def _count(schema: dict, key: str, default: int, path: str) -> int:
        value = schema.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SchemaError(f"{path}: {key} must be a non-negative integer")
        return value

    def object(self, schema: dict, path: str, depth: int) -> int:
        props = schema.get("properties") or {}
        if not isinstance(props, dict):
            raise SchemaError(f"{path}: properties must be an object")
        required = schema.get("required") or []
        if not isinstance(required, list) or not all(isinstance(r, str) for r in required):
            raise SchemaError(f"{path}: required must be a list of property names")
        extra = schema.get("additionalProperties", True)
        nid = self.new(_K_OBJ)
        if not props and not required:
            if extra is False:
                return nid                                  # only {}
            value = extra if isinstance(extra, dict) else True
            self.nodes[nid].free = self.compile(value, f"{path}/additionalProperties",
                                                depth + 1)
            if self.key_string < 0:
                self.key_string = self.new(_K_STR)
            return nid
        names = list(props)
        for name in required:
            if name not in props:
                if extra is False:
                    raise SchemaError(f"{path}: required property {name!r} is not in "
                                      "properties and additionalProperties is false")
                names.append(name)
        keys, vals, req = [], [], []
        for name in names:
            sub = props.get(name, extra if isinstance(extra, dict) else True)
            keys.append(_literal_bytes(str(name)))
            vals.append(self.compile(sub, f"{path}/properties/{name}", depth + 1))
            req.append(name in required)
        node = self.nodes[nid]
        node.keys, node.vals, node.req = keys, vals, req
        return nid

    def literal(self, values: list, path: str) -> int:
        cands: list[bytes] = []
        for value in values:
            try:
                data = _literal_bytes(value)
            except (TypeError, ValueError) as err:
                raise SchemaError(f"{path}: enum value {value!r} is not JSON") from err
            if data not in cands:
                cands.append(data)
        nid = self.new(_K_LIT)
        self.nodes[nid].cands = cands
        return nid

    def union(self, alts: list[int]) -> int:
        if len(alts) == 1:
            return alts[0]
        nid = self.new(_K_UNION)
        self.nodes[nid].alts = list(alts)
        return nid

    def any_value(self, depth: int) -> int:
        depth = min(depth, ANY_DEPTH)
        found = self.any_nodes.get(depth)
        if found is not None:
            return found
        nid = self.new(_K_UNION)
        self.any_nodes[depth] = nid
        alts = [self.new(_K_STR), self.new(_K_NUM), self.literal([True, False, None], "")]
        if depth < ANY_DEPTH:
            obj = self.new(_K_OBJ)
            self.nodes[obj].free = self.any_value(depth + 1)
            if self.key_string < 0:
                self.key_string = self.new(_K_STR)
            arr = self.new(_K_ARR)
            self.nodes[arr].item = self.any_value(depth + 1)
            alts = [obj, arr] + alts
        self.nodes[nid].alts = alts
        return nid

    def ref(self, ref: str, path: str) -> int:
        found = self.refs.get(ref)
        if found is not None:
            return found
        if not ref.startswith("#"):
            raise SchemaError(f"{path}: only local $ref (#/...) is supported, not {ref!r}")
        target: Any = self.root_schema
        for part in [p for p in ref[1:].split("/") if p]:
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(target, dict) and part in target:
                target = target[part]
            elif isinstance(target, list) and part.isdigit() and int(part) < len(target):
                target = target[int(part)]
            else:
                raise SchemaError(f"{path}: $ref {ref!r} points nowhere")
        placeholder = self.new(_K_REF)
        self.refs[ref] = placeholder
        self.nodes[placeholder].target = self.compile(target, ref or "#")
        return placeholder

    # -- after compiling ----------------------------------------------------
    def finish(self, root: int) -> tuple[list[_Node], int, int]:
        nodes = self.nodes

        def real(nid: int) -> int:
            seen = 0
            while nid >= 0 and nodes[nid].kind == _K_REF:
                nid = nodes[nid].target
                seen += 1
                if seen > len(nodes):
                    raise SchemaError("$ref refers only to itself")
            return nid

        for node in nodes:
            node.vals = [real(v) if v >= 0 else v for v in node.vals]
            if node.free >= 0:
                node.free = real(node.free)
            if node.item >= 0:
                node.item = real(node.item)
            if node.dyn is not None:
                node.dyn = [real(d) for d in node.dyn]
            if node.call >= 0:
                node.call = real(node.call)
            node.alts = [real(a) for a in node.alts]
        root = real(root)
        for nid, node in enumerate(nodes):
            if node.kind == _K_UNION:
                self._build_union(nid)
        for node in nodes:
            if node.kind == _K_LIT or (node.kind == _K_OBJ and node.keys):
                node.trie = _Trie(node.cands if node.kind == _K_LIT else node.keys)
        return nodes, root, self.key_string

    def _flatten(self, nid: int, out: list[int], seen: set[int]) -> None:
        node = self.nodes[nid]
        if node.kind == _K_UNION:
            if nid in seen:
                raise SchemaError("anyOf refers back to itself")
            seen.add(nid)
            for a in node.alts:
                self._flatten(a, out, seen)
        elif nid not in out:
            out.append(nid)

    def _first_bytes(self, nid: int) -> set[int]:
        node = self.nodes[nid]
        if node.kind == _K_OBJ:
            return {0x7B}
        if node.kind == _K_ARR:
            return {0x5B}
        if node.kind == _K_STR:
            return {0x22}
        if node.kind == _K_NUM:
            return set(b"-0123456789")
        if node.kind == _K_LIT:
            return {c[0] for c in node.cands}
        raise SchemaError("a value of this schema cannot be told apart from another one")

    def _build_union(self, nid: int) -> None:
        nodes = self.nodes
        flat: list[int] = []
        self._flatten(nid, flat, set())
        lits = [a for a in flat if nodes[a].kind == _K_LIT]
        strs = [a for a in flat if nodes[a].kind == _K_STR]
        nums = [a for a in flat if nodes[a].kind == _K_NUM]
        others = [a for a in flat if nodes[a].kind not in (_K_LIT, _K_STR, _K_NUM)]
        cands: list[bytes] = []
        for a in lits:
            for c in nodes[a].cands:
                if c not in cands:
                    cands.append(c)
        # a free string (or number) already accepts the matching enum values
        if strs and all(not nodes[s].track for s in strs):
            cands = [c for c in cands if not c.startswith(b'"')]
        if nums and not any(nodes[n].integer for n in nums):
            cands = [c for c in cands if c[:1] not in b"-0123456789"]
        alts = others + strs[:1] + nums[:1] + strs[1:] + nums[1:]
        if cands:
            lit = self.new(_K_LIT)
            nodes[lit].cands = cands
            nodes[lit].trie = _Trie(cands)
            alts.append(lit)
        dispatch: dict[int, int] = {}
        for a in alts:
            for b in self._first_bytes(a):
                if b in dispatch and dispatch[b] != a:
                    raise SchemaError("anyOf / oneOf / a type list with alternatives that "
                                      "start alike (two objects, two strings, ...) is not "
                                      "supported")
                dispatch[b] = a
        nodes[nid].alts = alts
        nodes[nid].dispatch = dispatch


# ----------------------------------------------------------------------
#  The grammar: compiled nodes + automaton
# ----------------------------------------------------------------------
class JsonGrammar:
    """A compiled JSON grammar (schema, JSON mode or tool calls) and its automaton.

    States are tuples of frames (immutable, hashable): :attr:`initial`,
    :meth:`feed` one byte at a time (``None``: refused), :meth:`cost` (bytes
    the shortest valid ending still needs), :meth:`done`, :meth:`free`
    (an ordinary answer after ``tool_choice="auto"``).
    """

    def __init__(self, nodes: list[_Node], root: int, key_string: int, *,
                 schema: dict | None, allow_text: bool = False, key: str = ""):
        self.nodes = nodes
        self.root = root
        self.key_string = key_string
        #: The JSON schema llama.cpp gets for GGUF models.
        self.schema = schema
        #: ``tool_choice="auto"``: an ordinary answer is allowed too.
        self.allow_text = allow_text
        #: Identifies the grammar in mask caches (same schema, same key).
        self.key = key
        self.mc: list[int] = [INF] * len(nodes)
        self._llama: Any = None
        self._vocab_checked: set[int] = set()
        self._min_costs()
        if self.mc[root] >= INF:
            raise SchemaError("the schema has no finite value (a required property refers "
                              "back to its own object)")
        self._tables()
        self.initial: tuple = ((_VAL, root, 0),)

    # -- precomputation -----------------------------------------------------
    def _min_costs(self) -> None:
        nodes, mc = self.nodes, self.mc
        for _ in range(len(nodes) + 2):
            changed = False
            for nid, node in enumerate(nodes):
                value = self._node_cost(node)
                if value < mc[nid]:
                    mc[nid] = value
                    changed = True
            if not changed:
                return

    def _node_cost(self, node: _Node) -> int:
        mc = self.mc
        kind = node.kind
        if kind == _K_STR:
            return 2 + node.min_len
        if kind == _K_NUM:
            return 1
        if kind == _K_LIT:
            return min(len(c) for c in node.cands)
        if kind == _K_ARR:
            if node.min_items == 0:
                return 2
            item = mc[node.item]
            return INF if item >= INF else 2 + node.min_items * item + node.min_items - 1
        if kind == _K_UNION:
            return min((mc[a] for a in node.alts), default=INF)
        if kind == _K_AUTO:
            return 1
        if kind == _K_OBJ:
            if node.free >= 0 or not node.keys:
                return 2
            if node.dyn is not None:
                best = min((len(c) + mc[d] for c, d in
                            zip(self.nodes[node.vals[0]].cands, node.dyn, strict=True)),
                           default=INF)
                if best >= INF:
                    return INF
                return 2 + len(node.keys[0]) + 1 + best + 1 + len(node.keys[1]) + 1
            total, count = 2, 0
            for key, val, req in zip(node.keys, node.vals, node.req, strict=True):
                if req:
                    if mc[val] >= INF:
                        return INF
                    total += len(key) + 1 + mc[val]
                    count += 1
            return total + max(0, count - 1)
        return INF

    def _tables(self) -> None:
        mc = self.mc
        for node in self.nodes:
            if node.kind != _K_OBJ or not node.keys:
                continue
            n = len(node.keys)
            if node.dyn is not None:
                names = self.nodes[node.vals[0]].cands
                node.na_min = min(len(c) + mc[d] for c, d in zip(names, node.dyn, strict=True))
            first_req_after = [n] * (n + 1)
            nxt = n
            for j in range(n - 1, -1, -1):
                if node.req[j]:
                    nxt = j
                first_req_after[j] = nxt
            node.allowed, node.need, node.rest, node.comma = [], [], [], []
            rest_after = [0] * (n + 1)
            for j in range(n - 1, -1, -1):
                add = 0
                if node.req[j]:
                    value = mc[node.vals[j]] if node.vals[j] >= 0 else 0
                    add = 1 + len(node.keys[j]) + 1 + value
                rest_after[j] = rest_after[j + 1] + add
            for last in range(-1, n):
                lo = last + 1
                hi = first_req_after[lo] if lo < n else n
                allowed = 0
                for j in range(lo, min(hi, n - 1) + 1):
                    allowed |= 1 << j
                node.allowed.append(allowed)
                node.need.append(hi < n)
                node.rest.append(rest_after[lo] if lo <= n else 0)
                if hi < n:
                    node.comma.append(rest_after[lo])          # (rest - 1 comma) + '}'
                elif allowed:
                    node.comma.append(min(len(node.keys[j]) + 1
                                          + (mc[node.vals[j]] if node.vals[j] >= 0 else 0)
                                          for j in _bits(allowed)) + 1)
                else:
                    node.comma.append(INF)
            node.key_total = [len(node.keys[j]) + 1 + (mc[node.vals[j]] if node.vals[j] >= 0
                                                       else 0) + rest_after[j + 1]
                              for j in range(n)]
            node.start_cost = (rest_after[0] - 1 if rest_after[0] else 0) + 1

    # -- the automaton --------------------------------------------------------
    def done(self, state: tuple) -> bool:
        return state[-1][0] == _DONE

    def free(self, state: tuple) -> bool:
        return state[-1][0] == _TEXT

    def _complete(self, rest: tuple, info: int):
        if not rest:
            return _DONE_STACK
        parent = rest[-1]
        tag = parent[0]
        if tag == _OBJ:
            _, nid, ph, last, _ws, cur, sel = parent
            if ph == _P_VALUE:
                if cur == 0 and self.nodes[nid].dyn is not None:
                    sel = info
                return rest[:-1] + ((_OBJ, nid, _P_AFTER, last, 0, cur, sel),)
            if ph == _P_KEY:                        # a free-form key ended
                return rest[:-1] + ((_OBJ, nid, _P_COLON, last, 0, cur, sel),)
            return None
        if tag == _ARR:
            _, nid, _ph, count, _ws = parent
            return rest[:-1] + ((_ARR, nid, _A_AFTER, count + 1, 0),)
        return None

    def _start(self, rest: tuple, nid: int, b: int):
        node = self.nodes[nid]
        kind = node.kind
        if kind == _K_UNION:
            alt = node.dispatch.get(b)
            return None if alt is None else self._start(rest, alt, b)
        if kind == _K_OBJ:
            return rest + ((_OBJ, nid, _P_START, -1, 0, -1, -1),) if b == 0x7B else None
        if kind == _K_STR:
            return rest + ((_STR, nid, 0, 0, 0),) if b == 0x22 else None
        if kind == _K_ARR:
            return rest + ((_ARR, nid, _A_START, 0, 0),) if b == 0x5B else None
        if kind == _K_NUM:
            if b == 0x2D:
                return rest + ((_NUM, nid, _N_SIGN, 1),)
            if b == 0x30:
                return rest + ((_NUM, nid, _N_ZERO, 1),)
            if 0x31 <= b <= 0x39:
                return rest + ((_NUM, nid, _N_INT, 1),)
            return None
        if kind == _K_LIT:
            trie = node.trie
            assert trie is not None
            child = trie.children[0].get(b)
            if child is None:
                return None
            if trie.end[child] >= 0 and not trie.children[child]:
                return self._complete(rest, trie.end[child])
            return rest + ((_LIT, nid, child),)
        if kind == _K_AUTO:
            if b == 0x7B:
                return rest + ((_OBJ, node.call, _P_START, -1, 0, -1, -1),)
            return _TEXT_STACK
        return None

    def feed(self, st: tuple, b: int) -> tuple | None:
        """The state after byte `b`, or None when `b` would break the JSON."""
        top = st[-1]
        tag = top[0]
        if tag == _STR:
            _, nid, length, esc, utf = top
            node = self.nodes[nid]
            if utf:
                if 0x80 <= b <= 0xBF:
                    utf -= 1
                    if utf == 0 and node.track:
                        length += 1
                    return st[:-1] + ((_STR, nid, length, 0, utf),)
                return None
            if esc:
                if esc == 1:
                    if b in _ESCAPES:
                        return st[:-1] + ((_STR, nid, length + node.track, 0, 0),)
                    if b == 0x75:                                    # \uXXXX
                        return st[:-1] + ((_STR, nid, length, 14, 0),)
                    return None
                if b in _HEX:
                    if esc == 11:
                        return st[:-1] + ((_STR, nid, length + node.track, 0, 0),)
                    return st[:-1] + ((_STR, nid, length, esc - 1, 0),)
                return None
            if b == 0x22:
                if length < node.min_len:
                    return None
                return self._complete(st[:-1], -1)
            if node.track and length >= node.max_len:
                return None
            if b == 0x5C:
                return st[:-1] + ((_STR, nid, length, 1, 0),)
            if b < 0x20:
                return None
            if b < 0x80:
                return st[:-1] + ((_STR, nid, length + 1, 0, 0),) if node.track else st
            if 0xC2 <= b <= 0xDF:
                utf = 1
            elif 0xE0 <= b <= 0xEF:
                utf = 2
            elif 0xF0 <= b <= 0xF4:
                utf = 3
            else:
                return None
            return st[:-1] + ((_STR, nid, length, 0, utf),)
        if tag == _VAL:
            if b in _WS:
                if top[2] >= WS_MAX:
                    return None
                return st[:-1] + ((_VAL, top[1], top[2] + 1),)
            return self._start(st[:-1], top[1], b)
        if tag == _OBJ:
            _, nid, ph, last, ws, cur, sel = top
            node = self.nodes[nid]
            if b in _WS:
                if ws >= WS_MAX:
                    return None
                return st[:-1] + ((_OBJ, nid, ph, last, ws + 1, cur, sel),)
            if node.free >= 0:                                   # free-form object
                if ph == _P_START or ph == _P_COMMA:
                    if b == 0x22:
                        return st[:-1] + ((_OBJ, nid, _P_KEY, last, 0, cur, sel),
                                          (_STR, self.key_string, 0, 0, 0))
                    if b == 0x7D and ph == _P_START:
                        return self._complete(st[:-1], -1)
                    return None
                if ph == _P_COLON:
                    if b == 0x3A:
                        return st[:-1] + ((_OBJ, nid, _P_VALUE, last, 0, cur, sel),
                                          (_VAL, node.free, 0))
                    return None
                if ph == _P_AFTER:
                    if b == 0x2C:
                        return st[:-1] + ((_OBJ, nid, _P_COMMA, last, 0, cur, sel),)
                    if b == 0x7D:
                        return self._complete(st[:-1], -1)
                return None
            if not node.keys:                                    # only {}
                return self._complete(st[:-1], -1) if b == 0x7D and ph == _P_START else None
            if ph == _P_START or ph == _P_COMMA:
                if b == 0x22:
                    allowed = node.allowed[last + 1]
                    trie = node.trie
                    assert trie is not None
                    child = trie.children[0].get(0x22)
                    if not allowed or child is None:
                        return None
                    return st[:-1] + ((_OBJ, nid, _P_KEY, last, 0, cur, sel),
                                      (_KEY, nid, child, allowed))
                if b == 0x7D and ph == _P_START and not node.need[0]:
                    return self._complete(st[:-1], -1)
                return None
            if ph == _P_COLON:
                if b == 0x3A:
                    value = node.vals[cur]
                    if value < 0:
                        assert node.dyn is not None
                        value = node.dyn[sel]
                    return st[:-1] + ((_OBJ, nid, _P_VALUE, last, 0, cur, sel),
                                      (_VAL, value, 0))
                return None
            if ph == _P_AFTER:
                if b == 0x2C and node.allowed[last + 1]:
                    return st[:-1] + ((_OBJ, nid, _P_COMMA, last, 0, cur, sel),)
                if b == 0x7D and not node.need[last + 1]:
                    return self._complete(st[:-1], -1)
            return None
        if tag == _KEY:
            _, nid, pos, allowed = top
            trie = self.nodes[nid].trie
            assert trie is not None
            child = trie.children[pos].get(b)
            if child is None or not trie.mask[child] & allowed:
                return None
            end = trie.end[child]
            if end >= 0 and (allowed >> end) & 1:
                parent = st[-2]
                return st[:-2] + ((_OBJ, nid, _P_COLON, end, 0, end, parent[6]),)
            return st[:-1] + ((_KEY, nid, child, allowed),)
        if tag == _ARR:
            _, nid, ph, count, ws = top
            node = self.nodes[nid]
            if b in _WS:
                if ws >= WS_MAX:
                    return None
                return st[:-1] + ((_ARR, nid, ph, count, ws + 1),)
            if ph == _A_START:
                if b == 0x5D:
                    return self._complete(st[:-1], -1) if node.min_items == 0 else None
                if node.max_items <= 0:
                    return None
                return self._start(st[:-1] + ((_ARR, nid, _A_VALUE, 0, 0),), node.item, b)
            if ph == _A_AFTER:
                if b == 0x2C:
                    if count >= node.max_items:
                        return None
                    return st[:-1] + ((_ARR, nid, _A_COMMA, count, 0),)
                if b == 0x5D:
                    return self._complete(st[:-1], -1) if count >= node.min_items else None
                return None
            if ph == _A_COMMA:
                return self._start(st[:-1] + ((_ARR, nid, _A_VALUE, count, 0),), node.item, b)
            return None
        if tag == _NUM:
            _, nid, ns, n = top
            room = n < NUM_MAX
            if 0x30 <= b <= 0x39:
                if ns == _N_SIGN:
                    return st[:-1] + ((_NUM, nid, _N_ZERO if b == 0x30 else _N_INT, n + 1),)
                if ns == _N_INT or ns == _N_FRAC or ns == _N_EXP:
                    return st[:-1] + ((_NUM, nid, ns, n + 1),) if room else None
                if ns == _N_DOT:
                    return st[:-1] + ((_NUM, nid, _N_FRAC, n + 1),)
                if ns == _N_E or ns == _N_ESIGN:
                    return st[:-1] + ((_NUM, nid, _N_EXP, n + 1),)
                return None                                       # 0 then a digit
            if not self.nodes[nid].integer and room:
                if b == 0x2E and (ns == _N_ZERO or ns == _N_INT):
                    return st[:-1] + ((_NUM, nid, _N_DOT, n + 1),)
                if (b == 0x65 or b == 0x45) and ns in (_N_ZERO, _N_INT, _N_FRAC):
                    return st[:-1] + ((_NUM, nid, _N_E, n + 1),)
            if (b == 0x2B or b == 0x2D) and ns == _N_E:
                return st[:-1] + ((_NUM, nid, _N_ESIGN, n + 1),)
            if ns in _NUM_ACCEPT:
                after = self._complete(st[:-1], -1)
                return None if after is None else self.feed(after, b)
            return None
        if tag == _LIT:
            _, nid, pos = top
            trie = self.nodes[nid].trie
            assert trie is not None
            child = trie.children[pos].get(b)
            if child is not None:
                if trie.end[child] >= 0 and not trie.children[child]:
                    return self._complete(st[:-1], trie.end[child])
                return st[:-1] + ((_LIT, nid, child),)
            end = trie.end[pos]
            if end >= 0:
                after = self._complete(st[:-1], end)
                return None if after is None else self.feed(after, b)
            return None
        if tag == _TEXT:
            return st
        return None                                               # done: nothing more

    def feed_bytes(self, st: tuple, data: bytes) -> tuple | None:
        cur: tuple | None = st
        for b in data:
            assert cur is not None
            cur = self.feed(cur, b)
            if cur is None:
                return None
        return cur

    # -- costs ------------------------------------------------------------------
    def cost(self, st: tuple) -> int:
        """Bytes the shortest valid ending of the answer still needs from `st`."""
        nodes, mc = self.nodes, self.mc
        total = 0
        n = len(st)
        i = 0
        while i < n:
            f = st[i]
            tag = f[0]
            if tag == _STR:
                node = nodes[f[1]]
                esc, utf = f[3], f[4]
                rem = 1 + utf
                if esc == 1:
                    rem += 1
                elif esc > 10:
                    rem += esc - 10
                if node.min_len:
                    rem += max(0, node.min_len - f[2] - (1 if (esc or utf) else 0))
                total += rem
            elif tag == _OBJ:
                node = nodes[f[1]]
                if node.dyn is not None:
                    value, skip = self._call_cost(node, f, st[i + 1] if i + 1 < n else None)
                    total += value
                    i += 1 + skip
                    continue
                total += self._obj_cost(node, f)
            elif tag == _VAL:
                total += mc[f[1]]
            elif tag == _KEY:
                node = nodes[f[1]]
                trie = node.trie
                assert trie is not None
                mask = trie.mask[f[2]] & f[3]
                total += min(node.key_total[j] for j in _bits(mask)) - trie.depth[f[2]]
            elif tag == _ARR:
                node = nodes[f[1]]
                ph, count = f[2], f[3]
                m, item = node.min_items, mc[node.item]
                if ph == _A_START:
                    total += 1 if m == 0 else m * item + m
                elif ph == _A_VALUE:
                    total += max(0, m - count - 1) * (1 + item) + 1
                elif ph == _A_AFTER:
                    total += max(0, m - count) * (1 + item) + 1
                else:
                    total += item + max(0, m - count - 1) * (1 + item) + 1
            elif tag == _NUM:
                total += 0 if f[2] in _NUM_ACCEPT else 1
            elif tag == _LIT:
                trie = nodes[f[1]].trie
                assert trie is not None
                total += trie.minrem[f[2]]
            i += 1
        return total

    def _obj_cost(self, node: _Node, f: tuple) -> int:
        ph, last, cur = f[2], f[3], f[5]
        if node.free >= 0:
            value = self.mc[node.free]
            if ph in (_P_START, _P_VALUE, _P_AFTER):
                return 1
            if ph in (_P_KEY, _P_COLON):
                return 2 + value
            return 4 + value
        if not node.keys:
            return 1
        if ph == _P_START:
            return node.start_cost
        if ph == _P_KEY:
            return 1
        if ph == _P_COLON:
            return 1 + self.mc[node.vals[cur]] + node.rest[cur + 1] + 1
        if ph == _P_VALUE:
            return node.rest[cur + 1] + 1
        if ph == _P_AFTER:
            return node.rest[last + 1] + 1
        return node.comma[last + 1]

    def _call_cost(self, node: _Node, f: tuple, child: tuple | None) -> tuple[int, int]:
        """Cost of a tool call object (name first, then the name's arguments)."""
        assert node.dyn is not None
        mc = self.mc
        _, _nid, ph, last, _ws, cur, sel = f
        len0, len1 = len(node.keys[0]), len(node.keys[1])
        trie = node.trie
        assert trie is not None
        tail = 1 + len1 + 1 + 1                     # , "arguments" : ... }
        if sel < 0:
            if ph == _P_START:
                return len0 + 1 + node.na_min + tail, 0
            if ph == _P_KEY and child is not None:
                return len0 - trie.depth[child[2]] + 1 + node.na_min + tail, 1
            if ph == _P_COLON:
                return 1 + node.na_min + tail, 0
            if ph == _P_VALUE and child is not None:
                if child[0] == _VAL:
                    return node.na_min + tail, 1
                names = self.nodes[node.vals[0]].trie
                assert names is not None
                pos = child[2]
                depth = names.depth[pos]
                best = min(names.lengths[j] - depth + mc[node.dyn[j]]
                           for j in _bits(names.mask[pos]))
                return best + tail, 1
            raise AssertionError("tool call state")
        args = mc[node.dyn[sel]]
        if ph == _P_AFTER and last == 0:
            return 1 + len1 + 1 + args + 1, 0
        if ph == _P_COMMA:
            return len1 + 1 + args + 1, 0
        if ph == _P_KEY and child is not None:
            return len1 - trie.depth[child[2]] + 1 + args + 1, 1
        if ph == _P_COLON:
            return 1 + args + 1, 0
        del cur
        return 1, 0                                 # value in progress, or after it

    # -- shortest endings ---------------------------------------------------
    _PREFERRED = (b'"}]:,0' + bytes(range(0x61, 0x7B)) + bytes(range(0x20, 0x7F))
                  + bytes(range(0x80, 0xC0)))
    _IN_STRING = b'"' + bytes(range(0x61, 0x7B)) + _PREFERRED

    def ending_byte(self, st: tuple) -> int | None:
        """The first byte of a shortest valid ending of `st` (None: complete)."""
        if self.done(st) or self.free(st):
            return None
        target = self.cost(st) - 1
        order = self._IN_STRING if st[-1][0] == _STR else self._PREFERRED
        for b in order:
            nxt = self.feed(st, b)
            if nxt is not None and self.cost(nxt) <= target:
                return b
        raise AssertionError("no byte shortens the answer")    # pragma: no cover

    def ending(self, st: tuple) -> bytes:
        """A shortest valid ending of `st` (``b""`` when it is complete)."""
        out = bytearray()
        cur: tuple | None = st
        while True:
            assert cur is not None
            b = self.ending_byte(cur)
            if b is None:
                return bytes(out)
            out.append(b)
            cur = self.feed(cur, b)

    def needed_bytes(self) -> tuple[set[int], list[bytes]]:
        """Bytes a model needs as tokens of their own to always finish an answer:
        ``(every one of these, [at least one of each of these])``."""
        need: set[int] = set()
        one_of: list[bytes] = []
        for node in self.nodes:
            kind = node.kind
            if kind == _K_OBJ or kind == _K_AUTO:
                need.update(b'{}:,"')
            elif kind == _K_ARR:
                need.update(b"[],")
            elif kind == _K_STR:
                need.add(0x22)
                if node.min_len:
                    one_of.append(bytes(range(0x61, 0x7B)) + bytes(range(0x41, 0x5B)))
            elif kind == _K_NUM:
                one_of.append(b"0123456789")
            for word in node.keys + node.cands:
                need.update(word)
        return need, one_of

    def accepts(self, text: str) -> bool:
        """Is `text` a complete answer of this grammar (tests, checks)?"""
        st = self.feed_bytes(self.initial, text.strip().encode("utf-8"))
        return st is not None and (self.done(st) or self.free(st))


# ----------------------------------------------------------------------
#  Building grammars
# ----------------------------------------------------------------------
_GRAMMARS: OrderedDict[str, JsonGrammar] = OrderedDict()
_GRAMMARS_LOCK = threading.Lock()


def _cached(key_data: Any, build) -> JsonGrammar:
    key = hashlib.sha256(json.dumps(key_data, sort_keys=False, default=str)
                         .encode("utf-8")).hexdigest()[:24]
    with _GRAMMARS_LOCK:
        found = _GRAMMARS.get(key)
        if found is not None:
            _GRAMMARS.move_to_end(key)
            return found
    grammar = build(key)
    with _GRAMMARS_LOCK:
        _GRAMMARS[key] = grammar
        while len(_GRAMMARS) > 64:
            _GRAMMARS.popitem(last=False)
    return grammar


def compile_schema(schema: Any) -> JsonGrammar:
    """The grammar of the values a JSON schema accepts (its root: an object or array)."""
    def build(key: str) -> JsonGrammar:
        comp = _Compiler(schema)
        root = comp.compile(schema, "#")
        nodes, root, key_string = comp.finish(root)
        if nodes[root].kind not in (_K_OBJ, _K_ARR):
            raise SchemaError("the root of the schema must be an object (or an array)")
        return JsonGrammar(nodes, root, key_string, schema=schema, key=key)
    if not isinstance(schema, dict):
        raise SchemaError("the schema must be a JSON object")
    return _cached(["schema", schema], build)


def json_object_grammar() -> JsonGrammar:
    """JSON mode (``response_format={"type": "json_object"}``): any JSON object."""
    def build(key: str) -> JsonGrammar:
        comp = _Compiler({})
        root = comp.new(_K_OBJ)
        comp.nodes[root].free = comp.any_value(1)
        if comp.key_string < 0:
            comp.key_string = comp.new(_K_STR)
        nodes, root, key_string = comp.finish(root)
        return JsonGrammar(nodes, root, key_string, schema={"type": "object"}, key=key)
    return _cached(["json_object"], build)


def grammar_for_response_format(response_format: Any) -> JsonGrammar | None:
    """The grammar of an OpenAI ``response_format`` (None for plain text).

    ``{"type": "json_object"}``, ``{"type": "json_schema", "json_schema":
    {"schema": {...}}}``, and as shortcuts ``"json"`` or a bare schema dict.
    """
    if response_format is None:
        return None
    if isinstance(response_format, str):
        if response_format in ("json", "json_object"):
            return json_object_grammar()
        if response_format == "text":
            return None
        raise SchemaError(f"unknown response_format {response_format!r}")
    if not isinstance(response_format, dict):
        raise SchemaError("response_format must be an object")
    kind = response_format.get("type")
    if kind in (None, "text"):
        if kind is None and response_format:
            return compile_schema(response_format)       # a bare schema
        return None
    if kind == "json_object":
        schema = response_format.get("schema")             # llama.cpp's extension
        return compile_schema(schema) if isinstance(schema, dict) else json_object_grammar()
    if kind == "json_schema":
        spec = response_format.get("json_schema")
        if not isinstance(spec, dict) or not isinstance(spec.get("schema"), dict):
            raise SchemaError("response_format json_schema needs json_schema.schema "
                              "(an object)")
        return compile_schema(spec["schema"])
    if kind in ("object", "array"):
        return compile_schema(response_format)            # a bare schema
    raise SchemaError(f"unknown response_format type {kind!r} (text, json_object or "
                      "json_schema)")


# ----------------------------------------------------------------------
#  Tools
# ----------------------------------------------------------------------
_TOOL_NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


def normalize_tools(tools: Any, functions: Any = None) -> list[dict]:
    """``[{"name", "description", "parameters"}]`` of OpenAI ``tools`` (or legacy
    ``functions``), checked: names, unique, parameters an object schema."""
    items: list[dict] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            raise SchemaError("every tool must be an object")
        if tool.get("type", "function") != "function":
            raise SchemaError(f"tool type {tool.get('type')!r} is not supported (function)")
        fn = tool.get("function")
        if not isinstance(fn, dict):
            raise SchemaError("a function tool needs a 'function' object")
        items.append(fn)
    for fn in functions or []:
        if not isinstance(fn, dict):
            raise SchemaError("every function must be an object")
        items.append(fn)
    out: list[dict] = []
    seen: set[str] = set()
    for fn in items:
        name = fn.get("name")
        if not isinstance(name, str) or not _TOOL_NAME.match(name):
            raise SchemaError(f"tool name {name!r}: 1-64 letters, digits, _ . or -")
        if name in seen:
            raise SchemaError(f"tool {name!r} is defined twice")
        seen.add(name)
        params = fn.get("parameters")
        if params is None:
            params = {"type": "object", "properties": {}}
        if not isinstance(params, dict):
            raise SchemaError(f"tool {name!r}: parameters must be a JSON schema object")
        out.append({"name": name, "description": str(fn.get("description") or ""),
                    "parameters": params})
    return out


def resolve_tool_choice(tool_choice: Any, tools: list[dict]) -> tuple[str, str | None]:
    """``("none" | "auto" | "required", name or None)`` of an OpenAI ``tool_choice``
    (or legacy ``function_call``)."""
    if not tools:
        return "none", None
    if tool_choice is None or tool_choice == "auto":
        return "auto", None
    if tool_choice in ("none", "required"):
        return str(tool_choice), None
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") if tool_choice.get("type", "function") == "function" \
            else None
        name = fn.get("name") if isinstance(fn, dict) else tool_choice.get("name")
        if isinstance(name, str):
            if name not in {t["name"] for t in tools}:
                raise SchemaError(f"tool_choice names {name!r}, which is not among the tools")
            return "required", name
    raise SchemaError("tool_choice must be none, auto, required or "
                      "{\"type\": \"function\", \"function\": {\"name\": ...}}")


def tool_grammar(tools: list[dict], mode: str = "required", name: str | None = None) -> JsonGrammar:
    """The grammar of one call ``{"name": ..., "arguments": {...}}``.

    `mode` ``"auto"``: an ordinary answer is allowed too (decided by the first
    character); ``"required"``: a call, of tool `name` when given.
    """
    chosen = [t for t in tools if name is None or t["name"] == name]
    if not chosen:
        raise SchemaError("no tool to call")
    allow_text = mode == "auto"

    def build(key: str) -> JsonGrammar:
        root_schema = {"$defs": {f"tool{i}": t["parameters"] for i, t in enumerate(chosen)}}
        comp = _Compiler(root_schema)
        args = []
        for i, tool in enumerate(chosen):
            params = _as_object_schema(tool["parameters"])
            # each tool's own $defs resolve inside its parameters
            sub = _Compiler(params)
            sub.nodes = comp.nodes
            sub.any_nodes = comp.any_nodes
            sub.key_string = comp.key_string
            node = sub.compile(params, f"#/tools/{tool['name']}/parameters")
            comp.key_string = sub.key_string
            args.append(node)
            del i
        names = comp.literal([t["name"] for t in chosen], "#/tools")
        call = comp.new(_K_OBJ)
        call_node = comp.nodes[call]
        call_node.keys = [_literal_bytes("name"), _literal_bytes("arguments")]
        call_node.vals = [names, -1]
        call_node.req = [True, True]
        call_node.dyn = args
        root = call
        if allow_text:
            root = comp.new(_K_AUTO)
            comp.nodes[root].call = call
        nodes, root, key_string = comp.finish(root)
        for a in nodes[call if not allow_text else nodes[root].call].dyn or []:
            if nodes[a].kind not in (_K_OBJ,):
                raise SchemaError("tool parameters must be an object schema")
        return JsonGrammar(nodes, root, key_string, schema=tool_call_schema(chosen),
                           allow_text=allow_text, key=key)
    return _cached(["tools", mode, name, chosen], build)


def _as_object_schema(params: dict) -> dict:
    """Tool parameters without a type are an object schema."""
    if "type" in params or any(k in params for k in ("anyOf", "oneOf", "$ref", "enum", "const")):
        return params
    return {"type": "object", **params}


def _hoist_defs(params: dict, prefix: str, defs: dict) -> dict:
    """`params` with its $defs moved into `defs` (renamed) and its $refs rewritten."""
    local = {}
    for word in ("$defs", "definitions"):
        for name, sub in (params.get(word) or {}).items():
            local[f"#/{word}/{name}"] = f"#/$defs/{prefix}{name}"
            defs[f"{prefix}{name}"] = sub

    def rewrite(node: Any) -> Any:
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if k in ("$defs", "definitions"):
                    continue
                if k == "$ref" and isinstance(v, str):
                    out[k] = local.get(v, f"#/$defs/{prefix}root" if v == "#" else v)
                else:
                    out[k] = rewrite(v)
            return out
        if isinstance(node, list):
            return [rewrite(v) for v in node]
        return node

    body = rewrite(params)
    if any(v == "#" for v in _refs(params)):
        defs[f"{prefix}root"] = body
    for name in list(defs):
        if name.startswith(prefix) and name != f"{prefix}root":
            defs[name] = rewrite(defs[name])
    return body


def _refs(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "$ref" and isinstance(v, str):
                yield v
            else:
                yield from _refs(v)
    elif isinstance(node, list):
        for v in node:
            yield from _refs(v)


def tool_call_schema(tools: list[dict], name: str | None = None) -> dict:
    """A JSON schema of one call, for llama.cpp grammars (GGUF models)."""
    defs: dict = {}
    alts = []
    for i, tool in enumerate(tools):
        if name is not None and tool["name"] != name:
            continue
        params = _hoist_defs(_as_object_schema(tool["parameters"]), f"t{i}_", defs)
        alts.append({"type": "object",
                     "properties": {"name": {"const": tool["name"]}, "arguments": params},
                     "required": ["name", "arguments"], "additionalProperties": False})
    schema: dict[str, Any] = dict(alts[0]) if len(alts) == 1 else {"anyOf": alts}
    if defs:
        schema["$defs"] = defs
    return schema


def tools_system_prompt(tools: list[dict], mode: str = "auto", name: str | None = None) -> str:
    """The system message that lists the tools and how to call one."""
    listed = [{"name": t["name"], "description": t["description"],
               "parameters": t["parameters"]} for t in tools
              if name is None or t["name"] == name]
    lines = ["# Tools", "", "You can call these functions:",
             json.dumps(listed, ensure_ascii=False), "",
             'To call a function, reply with only a JSON object: '
             '{"name": <function name>, "arguments": <arguments object>}.']
    if name is not None:
        lines.append(f"Call the function {name} now.")
    elif mode == "required":
        lines.append("Call one of the functions now.")
    else:
        lines.append("If no function is needed, answer normally.")
    return "\n".join(lines)


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                parts.append(str(part.get("text", "")))
            elif isinstance(part, dict):
                raise SchemaError(f"message content of type {part.get('type')!r} is not "
                                  "supported (text only)")
        return "".join(parts)
    return str(content)


def _call_text(call: dict) -> str:
    fn = call.get("function", call)
    name = fn.get("name", "")
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            pass
    return json.dumps({"name": name, "arguments": args}, ensure_ascii=False)


def plain_messages(messages: list[dict], tools: list[dict] | None = None,
                   mode: str = "none", name: str | None = None) -> list[dict]:
    """OpenAI / Ollama messages as plain ``{"role", "content"}`` turns.

    Content parts are joined, an assistant's ``tool_calls`` become the JSON
    calls it made, ``tool`` (and legacy ``function``) results become user
    turns that name the function, and with `tools` (unless `mode` is
    ``"none"``) the system message lists them.
    """
    names_by_id: dict[str, str] = {}
    out: list[dict] = []
    for message in messages:
        role = str(message.get("role", "user"))
        text = _content_text(message.get("content"))
        if role == "assistant":
            calls = list(message.get("tool_calls") or [])
            if message.get("function_call"):
                calls.append({"function": message["function_call"]})
            for call in calls:
                if isinstance(call, dict) and call.get("id"):
                    names_by_id[str(call["id"])] = (call.get("function") or {}).get("name", "")
            if calls:
                joined = "\n".join(_call_text(c) for c in calls if isinstance(c, dict))
                text = f"{text}\n{joined}".strip() if text else joined
            out.append({"role": "assistant", "content": text})
        elif role in ("tool", "function", "ipython"):
            fname = message.get("name") or names_by_id.get(str(message.get("tool_call_id")), "")
            label = f"Result of {fname}" if fname else "Function result"
            out.append({"role": "user", "content": f"{label}: {text}"})
        elif role in ("system", "developer"):
            out.append({"role": "system", "content": text})
        else:
            out.append({"role": "user", "content": text})
    if tools and mode != "none":
        prompt = tools_system_prompt(tools, mode, name)
        if out and out[0]["role"] == "system":
            out[0] = {"role": "system", "content": f"{out[0]['content']}\n\n{prompt}".strip()}
        else:
            out.insert(0, {"role": "system", "content": prompt})
    return out


def parse_tool_call(text: str, tools: list[dict]) -> dict | None:
    """``{"name", "arguments"}`` of the call in a reply, or None for an ordinary answer.

    Lenient for models that wrote it without a grammar: code fences and
    ``<tool_call>`` tags are taken off, ``parameters`` is read as
    ``arguments``. The name must be one of `tools`.
    """
    body = text.strip()
    for opener, closer in (("<tool_call>", "</tool_call>"), ("```json", "```"), ("```", "```")):
        if body.startswith(opener):
            body = body[len(opener):]
            if closer in body:
                body = body[:body.index(closer)]
            body = body.strip()
    if not body.startswith("{"):
        return None
    try:
        data, _end = json.JSONDecoder().raw_decode(body)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    name = data.get("name")
    if name not in {t["name"] for t in tools}:
        return None
    args = data.get("arguments", data.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if not isinstance(args, dict):
        return None
    return {"name": name, "arguments": args}


def openai_tool_calls(call: dict) -> list[dict]:
    """OpenAI's ``tool_calls`` list for one parsed call."""
    return [{"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
             "function": {"name": call["name"],
                          "arguments": json.dumps(call["arguments"], ensure_ascii=False)}}]


# ----------------------------------------------------------------------
#  Tokens
# ----------------------------------------------------------------------
class _TNode:
    __slots__ = ("children", "ids")

    def __init__(self) -> None:
        self.children: dict[int, _TNode] = {}
        self.ids: list[int] = []


def token_byte_table(tokenizer: Any, size: int | None = None) -> list[bytes | None]:
    """The bytes every token id adds to the text (None: special or unusable)."""
    total = int(size if size is not None else getattr(tokenizer, "vocab_size", 0) or 0)
    known = int(getattr(tokenizer, "vocab_size", total) or total)
    specials = set(getattr(tokenizer, "special_tokens", None) or [])
    by_bytes = getattr(tokenizer, "token_bytes", None)
    id_to_token = getattr(tokenizer, "id_to_token", None)
    out: list[bytes | None] = []
    for i in range(total):
        data: bytes | None = None
        if i < known:
            if callable(by_bytes):
                token = id_to_token.get(i) if isinstance(id_to_token, dict) else None
                if token is None or token not in specials:
                    raw = by_bytes(i)
                    if isinstance(raw, (bytes, bytearray)) and raw:
                        data = bytes(raw)
            else:
                text = tokenizer.decode([i])
                if text and text not in specials and text != "\ufffd":
                    data = text.encode("utf-8")
        out.append(data)
    return out


class TokenTable:
    """The bytes of a vocabulary in a prefix tree, and the masks of grammar states.

    ``costs(grammar, state)`` is an int16 vector over the vocabulary: for every
    token, the bytes the shortest ending still needs after it, or
    :data:`COST_REFUSED` when the grammar refuses it. Vectors are cached per
    (grammar, state) (LRU; ``CACHE_STATES`` of them, fewer for a large
    vocabulary).
    """

    def __init__(self, token_bytes: list[bytes | None]):
        self.token_bytes = token_bytes
        self.size = len(token_bytes)
        self.root = _TNode()
        self.single: dict[int, int] = {}
        for tid, data in enumerate(token_bytes):
            if not data:
                continue
            node = self.root
            for b in data:
                child = node.children.get(b)
                if child is None:
                    child = node.children[b] = _TNode()
                node = child
            node.ids.append(tid)
            if len(data) == 1:
                self.single.setdefault(data[0], tid)
        self._cache: OrderedDict[tuple, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.max_states = max(64, min(CACHE_STATES, CACHE_ENTRIES // max(1, self.size)))
        self.computed = 0
        self.hits = 0

    @classmethod
    def for_tokenizer(cls, tokenizer: Any, size: int | None = None) -> "TokenTable":
        return cls(token_byte_table(tokenizer, size))

    def missing(self, grammar: JsonGrammar) -> list[str]:
        """Characters the grammar needs that have no token of their own."""
        need, one_of = grammar.needed_bytes()
        out = sorted(chr(b) for b in need if b not in self.single)
        for group in one_of:
            if not any(b in self.single for b in group):
                out.append(f"one of {group.decode('latin-1')}")
        return out

    def costs(self, grammar: JsonGrammar, state: tuple):
        key = (grammar.key, state)
        with self._lock:
            found = self._cache.get(key)
            if found is not None:
                self._cache.move_to_end(key)
                self.hits += 1
                return found
        result = self._compute(grammar, state)
        with self._lock:
            self._cache[key] = result
            self.computed += 1
            while len(self._cache) > self.max_states:
                self._cache.popitem(last=False)
        return result

    def _compute(self, grammar: JsonGrammar, state: tuple):
        import torch

        costs = [COST_REFUSED] * self.size
        top = COST_REFUSED - 1
        feed, cost = grammar.feed, grammar.cost
        stack = [(state, self.root)]
        while stack:
            st, node = stack.pop()
            same_cost = -1
            for b, child in node.children.items():
                nxt = feed(st, b)
                if nxt is None:
                    continue
                if child.ids:
                    if nxt is st:
                        if same_cost < 0:
                            same_cost = min(cost(st), top)
                        value = same_cost
                    else:
                        value = min(cost(nxt), top)
                    for tid in child.ids:
                        costs[tid] = value
                if child.children:
                    stack.append((nxt, child))
        return torch.tensor(costs, dtype=torch.int16)


class JsonConstraint:
    """The grammar state of one generation: masks logits, follows the tokens.

    ``mask(logits)`` returns the logits with ``-inf`` for every token the
    grammar refuses or whose shortest ending would not fit into the tokens
    left; ``advance(token)`` moves the state on (it returns the token, or a
    token of the shortest ending if the sampler drew a refused one).
    """

    def __init__(self, grammar: JsonGrammar, table: TokenTable, max_tokens: int):
        self.grammar = grammar
        self.table = table
        self.state = grammar.initial
        self.remaining = int(max_tokens)
        self.tokens = 0
        self.done = False
        self.free = False

    def _allowed(self, size: int):
        import torch

        costs = self.table.costs(self.grammar, self.state)
        ok = costs <= min(self.remaining - 1, COST_REFUSED - 1)
        if size > ok.shape[0]:
            ok = torch.cat([ok, torch.zeros(size - ok.shape[0], dtype=torch.bool)])
        elif size < ok.shape[0]:
            ok = ok[:size]
        if not bool(ok.any()):                          # cannot happen; stay safe
            ok = torch.zeros(size, dtype=torch.bool)
            ok[self._ending_token()] = True
        return ok

    def mask(self, logits):
        if self.free or self.done:
            return logits
        ok = self._allowed(int(logits.shape[-1]))
        return logits.masked_fill(~ok.to(logits.device), float("-inf"))

    def _ending_token(self) -> int:
        b = self.grammar.ending_byte(self.state)
        if b is None or b not in self.table.single:
            raise RuntimeError("the JSON grammar has no token to finish the answer")
        return self.table.single[b]

    def advance(self, token_id: int) -> int:
        """Follow `token_id`; returns the token actually taken."""
        token_id = int(token_id)
        self.remaining -= 1
        self.tokens += 1
        if self.free or self.done:
            return token_id
        data = self.table.token_bytes[token_id] if 0 <= token_id < self.table.size else None
        st = self.grammar.feed_bytes(self.state, data) if data else None
        if st is None or self.grammar.cost(st) > self.remaining:
            token_id = self._ending_token()
            data = self.table.token_bytes[token_id]
            assert data is not None
            st = self.grammar.feed_bytes(self.state, data)
            assert st is not None
        self.state = st
        self.done = self.grammar.done(st)
        self.free = self.grammar.free(st)
        return token_id


# ----------------------------------------------------------------------
#  llama.cpp (GGUF)
# ----------------------------------------------------------------------
def llama_cpp_gbnf(grammar: JsonGrammar) -> str:
    """The grammar in llama.cpp's GBNF (needs llama-cpp-python)."""
    from llama_cpp.llama_grammar import json_schema_to_gbnf

    gbnf = json_schema_to_gbnf(json.dumps(grammar.schema))
    # llama-cpp-python's converter lets raw control characters into strings,
    # which JSON forbids (json.loads refuses them)
    gbnf = gbnf.replace(r'char ::= [^"\\]', r'char ::= [^"\\\x7F\x00-\x1F]')
    if grammar.allow_text:
        gbnf = re.sub(r"^root\s*::=", "auralite-call ::=", gbnf, flags=re.M)
        gbnf += ('\nroot ::= auralite-call | auralite-text\n'
                 'auralite-text ::= [^{ \\t\\n] [^\\x00]*\n')
    return gbnf


def llama_cpp_grammar(grammar: JsonGrammar) -> Any:
    """A ``llama_cpp.LlamaGrammar`` for the grammar (built once)."""
    if grammar._llama is None:
        from llama_cpp import LlamaGrammar

        grammar._llama = LlamaGrammar.from_string(llama_cpp_gbnf(grammar), verbose=False)
    return grammar._llama


# ----------------------------------------------------------------------
#  Checking values (tests, measurements)
# ----------------------------------------------------------------------
def validate_json(value: Any, schema: Any, root: Any = None, path: str = "#") -> list[str]:
    """What is wrong with `value` under `schema` (the supported keywords), [] if nothing."""
    root = schema if root is None else root
    if schema is True or schema == {} or schema is None:
        return []
    if schema is False:
        return [f"{path}: no value is allowed"]
    if "$ref" in schema:
        target: Any = root
        for part in [p for p in str(schema["$ref"])[1:].split("/") if p]:
            target = target[part]
        return validate_json(value, target, root, path)
    if "const" in schema:
        return [] if _same(value, schema["const"]) else [f"{path}: not the const value"]
    if "enum" in schema:
        return [] if any(_same(value, v) for v in schema["enum"]) else [f"{path}: not in enum"]
    for word in ("anyOf", "oneOf"):
        if word in schema:
            ok = sum(1 for alt in schema[word] if not validate_json(value, alt, root, path))
            if (word == "anyOf" and ok) or (word == "oneOf" and ok == 1):
                return []
            return [f"{path}: matches {ok} alternatives of {word}"]
    if schema.get("nullable") is True and value is None:
        return []
    kinds = schema.get("type")
    if kinds is None:
        if "properties" in schema or "required" in schema:
            kinds = "object"
        elif "items" in schema:
            kinds = "array"
    if isinstance(kinds, list):
        for kind in kinds:
            if not validate_json(value, {**schema, "type": kind}, root, path):
                return []
        return [f"{path}: none of the types {kinds}"]
    errors: list[str] = []
    if kinds == "object":
        if not isinstance(value, dict):
            return [f"{path}: not an object"]
        props = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                errors.append(f"{path}: missing {name!r}")
        extra = schema.get("additionalProperties", True)
        for name, item in value.items():
            if name in props:
                errors += validate_json(item, props[name], root, f"{path}/{name}")
            elif extra is False:
                errors.append(f"{path}: unexpected {name!r}")
            elif isinstance(extra, dict):
                errors += validate_json(item, extra, root, f"{path}/{name}")
    elif kinds == "array":
        if not isinstance(value, list):
            return [f"{path}: not an array"]
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", INF):
            errors.append(f"{path}: {len(value)} items")
        for i, item in enumerate(value):
            errors += validate_json(item, schema.get("items", True), root, f"{path}/{i}")
    elif kinds == "string":
        if not isinstance(value, str):
            return [f"{path}: not a string"]
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", INF):
            errors.append(f"{path}: length {len(value)}")
    elif kinds == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"{path}: not an integer")
    elif kinds == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{path}: not a number")
    elif kinds == "boolean":
        if not isinstance(value, bool):
            errors.append(f"{path}: not a boolean")
    elif kinds == "null":
        if value is not None:
            errors.append(f"{path}: not null")
    return errors


def _same(a: Any, b: Any) -> bool:
    return type(a) is type(b) and a == b or (
        isinstance(a, (int, float)) and isinstance(b, (int, float))
        and not isinstance(a, bool) and not isinstance(b, bool) and a == b)


# ----------------------------------------------------------------------
#  The engine side
# ----------------------------------------------------------------------
class StructuredMixin:
    """JSON mode, JSON schemas and tool calls of :class:`~auralite.engine.AuraLiteEngine`.

    ``generate_chat(..., response_format=...)`` (or ``grammar=``) writes JSON;
    :meth:`generate_json` parses it for you; :meth:`chat_with_tools` answers
    with a tool call or text. Native models decode under a
    :class:`JsonConstraint`, GGUF models under a llama.cpp grammar; Hugging
    Face models are refused.
    """

    model: Any
    tokenizer: Any

    def _token_table(self) -> TokenTable:
        """The tokenizer's :class:`TokenTable` (built once per tokenizer and size)."""
        tokenizer = self.tokenizer
        size = int(getattr(tokenizer, "vocab_size", 0) or 0)
        cached = self.__dict__.get("_json_token_table")
        if cached is not None and cached[0] is tokenizer and cached[1] == size:
            return cached[2]
        table = TokenTable.for_tokenizer(tokenizer, size)
        self.__dict__["_json_token_table"] = (tokenizer, size, table)
        return table

    def structured_check(self, grammar: JsonGrammar, max_new_tokens: int | None = None) -> None:
        """Raise ``ValueError`` when this model cannot follow `grammar` (before generating).

        Native models need a token of their own for every character the
        shortest answers use, and `max_new_tokens` must leave room for the
        shortest valid answer. GGUF models need llama-cpp-python (in process)
        or a llama.cpp server. Hugging Face models are not supported.
        """
        if self.model is None:
            raise ValueError("Train or load a model first!")
        if self.is_hf_model():                                   # type: ignore[attr-defined]
            raise ValueError("JSON mode and tools need a native AuraLite model or a GGUF "
                             "model; Hugging Face models generate in their own runtime")
        if self.is_gguf_model():                                 # type: ignore[attr-defined]
            from auralite.engine.gguf_proxy import LlamaServerProxy

            if not isinstance(self.model, LlamaServerProxy):
                try:
                    llama_cpp_grammar(grammar)
                except ImportError as err:
                    raise ValueError("JSON mode on a GGUF model needs llama-cpp-python") from err
                except Exception as err:
                    raise ValueError(f"llama.cpp cannot use this schema: {err}") from err
            return
        table = self._token_table()
        if id(table) not in grammar._vocab_checked:
            missing = table.missing(grammar)
            if missing:
                shown = " ".join(repr(c) for c in missing[:12])
                raise ValueError(f"this model's vocabulary has no token for {shown}: it cannot "
                                 "write this JSON (fine-tune it on text that contains them)")
            grammar._vocab_checked.add(id(table))
        need = grammar.cost(grammar.initial)
        if max_new_tokens is not None and int(max_new_tokens) < need:
            raise ValueError(f"max_tokens {int(max_new_tokens)} is too small for this JSON: "
                             f"the shortest valid answer takes {need} tokens")

    def _json_constraint(self, grammar: JsonGrammar | None,
                         max_new_tokens: int) -> JsonConstraint | None:
        if grammar is None:
            return None
        self.structured_check(grammar, max_new_tokens)
        return JsonConstraint(grammar, self._token_table(), max_new_tokens)

    def _gguf_structured_kwargs(self, grammar: JsonGrammar | None) -> dict:
        """What the GGUF proxy gets for `grammar`: a llama.cpp grammar or a JSON schema."""
        if grammar is None:
            return {}
        from auralite.engine.gguf_proxy import LlamaServerProxy

        if isinstance(self.model, LlamaServerProxy):
            return {} if grammar.allow_text else {"json_schema": grammar.schema}
        return {"grammar": llama_cpp_grammar(grammar)}

    def generate_json(self, messages: Any, schema: dict | None = None, *,
                      max_new_tokens: int = 256, **kwargs: Any) -> Any:
        """Answer `messages` (a list of chat messages, or one prompt) with JSON, parsed.

        `schema`: a JSON schema (its root an object or array); without it any
        JSON object (JSON mode).
        """
        grammar = compile_schema(schema) if schema is not None else json_object_grammar()
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        text = self.generate_chat(messages, max_new_tokens,          # type: ignore[attr-defined]
                                  grammar=grammar, **kwargs)
        return json.loads(text)

    def chat_with_tools(self, messages: list[dict], tools: list[dict],
                        tool_choice: Any = "auto", *, max_new_tokens: int = 256,
                        **kwargs: Any) -> dict:
        """Answer with a tool call or with text: ``{"content", "tool_calls"}``.

        `tools` and `tool_choice` as in OpenAI's chat API; ``tool_calls`` is
        OpenAI's list (one call at most) or None.
        """
        specs = normalize_tools(tools)
        mode, name = resolve_tool_choice(tool_choice, specs)
        plain = plain_messages(messages, specs, mode, name)
        if mode == "none":
            text = self.generate_chat(plain, max_new_tokens, **kwargs)  # type: ignore[attr-defined]
            return {"content": text, "tool_calls": None}
        grammar = tool_grammar(specs, mode, name)
        text = self.generate_chat(plain, max_new_tokens, grammar=grammar,  # type: ignore[attr-defined]
                                  **kwargs)
        call = parse_tool_call(text, specs)
        if call is None:
            return {"content": text, "tool_calls": None}
        return {"content": None, "tool_calls": openai_tool_calls(call)}
