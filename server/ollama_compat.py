"""Minimal Ollama compatibility helpers."""

def ollama_generate_response(model: str, prompt: str, response: str) -> dict:
    return {"model": model, "response": response, "done": True}
