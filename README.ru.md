# 🌟 AuraLite AI

[![CI](https://github.com/AlexanderNyr/AuraLite-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexanderNyr/AuraLite-AI/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Version](https://img.shields.io/badge/version-2.6.5-orange)

**English version: [README.md](README.md)** · Быстрый старт: [docs/quickstart.md](docs/quickstart.md)

**AuraLite AI** — учебно-практическая реализация языковой модели (LLM) на **PyTorch**:
обучение с нуля на своём тексте, генерация, чат, квантизация, визуализация внимания
и OpenAI-совместимый сервер. Всё работает на обычном ноутбуке (CPU) и ускоряется на
NVIDIA GPU автоматически.

## ⚡ Что нового в 2.6

- **QK-norm** (`use_qk_norm`) — RMS-нормировка Q/K по головам до RoPE (как в OLMo-2,
  Gemma-2/3, Qwen3): стабилизирует обучение на больших LR.
- **Muon** (`optimizer="muon"`) — ортогонализация моментума по Ньютону–Шульцу для
  скрытых матриц, AdamW остаётся для эмбеддингов и головы (`split_parameters_for_muon`).
- **WSD-расписание** (`lr_schedule="wsd"`, по умолчанию) — warmup → постоянный LR →
  короткий спад; из чекпоинта стабильной фазы можно ветвить decay-прогон.

```python
engine.train(text, {
    "use_qk_norm": True,
    "optimizer": "muon",
    "lr_schedule": "wsd",
    "wsd_stable_ratio": 0.8,
})
```

## 🚀 Ключевые возможности

- **Обучение с нуля**: свой BPE- или посимвольный токенизатор, dense loss по всем
  позициям (nanoGPT-style), validation split, autosave, продолжение обучения,
  gradient checkpointing, AMP, `torch.compile` для тренировочного шага.
- **🖼️ Генератор изображений с нуля** (вкладка Image): собственная диффузионная
  U-Net-модель (без предобученных весов), DDIM/DDPM-сэмплинг, EMA, Min-SNR, пресеты.
- **GGUF / llama.cpp**: `.gguf` модели — генерация, стриминг, батч-подсказки, Thinking
  Mode, perplexity. Либо через `llama-server` (одна загрузка модели на машину).
- **Архитектура**: decoder-only трансформер, RMSNorm (pre-norm), RoPE, SwiGLU, опционально
  GQA, KV-cache, Flash Attention через `scaled_dot_product_attention`.
- **⚡ Квантизация**: dynamic/static INT8, QAT, GPTQ (int2–int8), AWQ, HQQ, fp16/bf16 —
  со встроенным сравнением по размеру, скорости и perplexity
  ([подробно](docs/quantization.md)).
- **⏱️ Честные бенчмарки**: `auralite info --bench N` (токенов/с и KV-cache),
  `--bench-modes` (fp32/fp16/int8), `--bench-compile` (eager vs `torch.compile`),
  `--perplexity` для GGUF с кэшем результатов ([почему «не ускоряется»](docs/faq.md)).
- **💬 Чат и Thinking Mode**: шаблоны ChatML/Llama-2/Mistral/Gemma/Phi, история,
  стриминг, самокондиционирование (черновик → ответ), мини-RAG через DuckDuckGo.
- **🖥️ GUI на tkinter**: обучение текста и картинок, генерация, чат с графиками
  внимания и частот токенов, модели, квантизация, оценка, экспорт. Настройки и
  последняя сессия сохраняются; интерфейс на английском и русском
  (`$AURALITE_GUI_LANG=en|ru`).
- **🔌 OpenAI-совместимый сервер**: `/v1/chat/completions`, `/v1/completions` со SSE,
  `stop/seed/top_k/repetition_penalty/include_usage`, `/metrics`, API-ключи, лимиты,
  очередь запросов вместо потока 503.
- **🛡️ Безопасные чекпоинты**: `.pt` загружается с `weights_only=True` (старые файлы —
  через ограниченный allowlist), поддержка `.safetensors`.
- **Экспорт и обмен**: ONNX / TorchScript / Hugging Face (extras), загрузка на HF Hub,
  LoRA-адаптеры.

## ✅ Статус функций (честно)

| Функция | Статус | Комментарий |
|---|---|---|
| Спекулятивное декодирование | реализовано | prompt-lookup черновик + батч-проверка; при greedy результат токен-в-токен совпадает с `generate()` |
| FlexAttention | реализовано, opt-in | настоящий `torch.nn.attention.flex_attention` для префилла; автооткат к SDPA |
| GGUF perplexity | реализовано | оценка в окнах `n_ctx−1`; кэш результатов (`$AURALITE_HOME/gguf_ppl_cache.json`) |
| `llama-server` как бэкенд | реализовано | общая загрузка модели, свой KV-cache; perplexity на сервере недоступна — и это сказано явно |
| Квантизация | реализовано | torch.ao (по умолчанию) или torchao (opt-in) для dynamic INT8 |
| OpenAI-сервер | реализовано, один воркер | сознательно: движок, KV-cache и лимитер живут в памяти процесса; очередь + `AURALITE_QUEUE_TIMEOUT` |
| Батч-генерация | реализовано | группировка по длине + дедупликация одинаковых промптов |
| Ускорение инференса | измерено | `torch.compile` даёт ×1.1–1.7 на префилле и **медленнее** на декоде одного токена — поэтому декод остаётся eager |
| Сессии и локализация GUI | реализовано | `gui/config.py`, `gui/session.py`, `gui/i18n.py` |
| Логирование библиотеки | реализовано | ни один модуль библиотеки не печатает в stdout: всё через логгер `auralite`, `$AURALITE_LOG_LEVEL` |

## 📦 Установка

```bash
pip install -e .              # ядро: torch + numpy
pip install -e '.[serve]'     # + FastAPI/uvicorn (сервер)
pip install -e '.[full]'      # + llama-cpp-python, transformers, peft, safetensors
pip install -e '.[test]'      # + pytest
```

Требуется Python 3.10+; GPU не обязателен.

## 📖 Как пользоваться

```bash
# 1. обучить на своём тексте (100 КБ уже достаточно, чтобы увидеть обучение)
auralite train --text corpus.txt --params params.json
# большой корпус: токены в memmap, а не в RAM
auralite train --text corpus.txt --mmap --corpus-cache /path/corpus.uint32

# 2. генерация и чат
auralite generate --model model.pt --prompt "Жили-были" --length 200
auralite chat --model model.pt --template chatml --system "Ты — кот-поэт"

# 3. графический интерфейс
python gui_app.py

# 4. OpenAI-совместимый сервер
AURALITE_MODEL=model.pt AURALITE_API_KEY=sk-local \
  uvicorn server.openai_server:app --host 0.0.0.0 --port 8000

# 5. диагностика и бенчмарки
auralite info --model model.pt --bench 128 --bench-modes --bench-compile
auralite info --model model.gguf --perplexity --text corpus.txt
```

Подробный разбор — в [docs/quickstart.md](docs/quickstart.md) (English).

## ⚠️ Честные ограничения

- Сервер — **один воркер**: `--workers > 1` не имеет смысла (см. выше).
- Песочница агентских инструментов **не** даёт изоляции уровня ОС: подпроцесс с
  урезанным окружением и лимитами, но сеть он открыть всё ещё может.
- Квантизация INT8 на CPU обычно **сокращает память, а не время** — см.
  [docs/quantization.md](docs/quantization.md).
- `.gguf` нельзя дообучать; GGUF-сервер не умеет perplexity.
- Токенизированный корпус между запусками не переиспользуется: memmap из
  `--mmap` удаляется из временной папки сразу после обучения, а файл
  `--corpus-cache PATH` сохраняется, но при следующем запуске пишется заново.

## 🔗 Документация

| Файл | О чём |
|---|---|
| [docs/quickstart.md](docs/quickstart.md) | установка, обучение, генерация, сервер, GUI, чекпоинты |
| [docs/faq.md](docs/faq.md) | «почему не ускоряется»: измерение, `torch.compile`, KV-cache, квантизация, корпус |
| [docs/quantization.md](docs/quantization.md) | все методы квантизации, что реально экономится, GGUF |
| [ARCHITECTURE.md](ARCHITECTURE.md) | карта модулей и поток данных движка |
| [CHANGELOG.md](CHANGELOG.md) | что и почему изменилось в 2.6.5 |

## 🛠 Технические детали

- **Attention**: SDPA (Flash / memory-efficient) + causal mask + RoPE + опциональная GQA.
- **Нормализация**: RMSNorm (pre-norm).
- **Оптимизатор**: AdamW (0.9/0.95, weight decay) или Muon; WSD/cosine-расписание,
  gradient clipping.
- **Токенизация**: BPE (рекомендуется) или посимвольная; старые чекпоинты грузятся
  прозрачно.

## 📜 Лицензия

Apache-2.0.
