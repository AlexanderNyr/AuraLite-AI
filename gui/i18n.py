"""Minimal i18n foundation for GUI labels."""
MESSAGES = {
    "en": {"train": "Train", "generate": "Generate", "model": "Model", "quantization": "Quantization"},
}

def tr(key: str, lang: str = "en") -> str:
    return MESSAGES.get(lang, MESSAGES["en"]).get(key, key)
