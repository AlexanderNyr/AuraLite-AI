"""JSON mode, structured outputs and tool calls through the official OpenAI client (2.17).

Start a server first, with a native AuraLite chat model or a GGUF model:

    auralite serve --model chat.pt                     # or: --model qwen2.5-0.5b-instruct-q4_k_m.gguf
    pip install openai pydantic
    python examples/openai_structured.py --base-url http://127.0.0.1:8000/v1

Every answer below is decoded under a JSON grammar: it parses, and the second
one follows the Person schema (Pydantic -> JSON schema -> grammar).
"""
from __future__ import annotations

import argparse
import json

from openai import OpenAI, pydantic_function_tool
from pydantic import BaseModel


class Address(BaseModel):
    city: str
    country: str


class Person(BaseModel):
    name: str
    age: int
    address: Address


class GetWeather(BaseModel):
    """The current weather in a city."""
    city: str


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="sk-local")
    parser.add_argument("--model", default="auralite")
    args = parser.parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    # 1. JSON mode: any JSON object ("length": cut at max_tokens, a GGUF model only)
    reply = client.chat.completions.create(
        model=args.model, max_tokens=200,
        messages=[{"role": "user", "content": "Riga as a JSON object with the keys "
                                              "country, population and river."}],
        response_format={"type": "json_object"})
    choice = reply.choices[0]
    print("json_object:", choice.finish_reason, json.loads(choice.message.content or ""))

    # 2. Structured outputs: a Pydantic model (sent as a JSON schema)
    parsed = client.chat.completions.parse(
        model=args.model, max_tokens=160,
        messages=[{"role": "user",
                   "content": "Anna Berzina is 34 and lives in Riga, Latvia."}],
        response_format=Person)
    print("parsed:", parsed.choices[0].message.parsed)

    # 3. Tools: the model calls get_weather (tool_choice="required")
    called = client.chat.completions.parse(
        model=args.model, max_tokens=80,
        messages=[{"role": "user", "content": "What is the weather in Riga?"}],
        tools=[pydantic_function_tool(GetWeather, name="get_weather")],
        tool_choice="required")
    call = (called.choices[0].message.tool_calls or [])[0]
    print("tool call:", call.function.name, call.function.parsed_arguments)

    # 4. ... and answers with the tool's result
    final = client.chat.completions.create(
        model=args.model, max_tokens=80,
        messages=[{"role": "user", "content": "What is the weather in Riga?"},
                  {"role": "assistant", "content": None, "tool_calls": [{
                      "id": call.id, "type": "function",
                      "function": {"name": call.function.name,
                                   "arguments": call.function.arguments}}]},
                  {"role": "tool", "tool_call_id": call.id,
                   "content": json.dumps({"temperature_c": 12, "sky": "cloudy"})}],
        tools=[pydantic_function_tool(GetWeather, name="get_weather")])
    message = final.choices[0].message
    print("answer:", message.content if message.content is not None else message.tool_calls)


if __name__ == "__main__":
    main()
