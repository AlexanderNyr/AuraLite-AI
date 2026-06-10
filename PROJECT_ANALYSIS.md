# 🔬 AuraLite AI v2 — Полный анализ проекта

## 📋 Оглавление
1. [Общая архитектура](#общая-архитектура)
2. [model_engine.py](#model_enginepy)
3. [gui_app.py](#gui_apppy)
4. [build_exe.bat](#build_exebat)
5. [Сильные стороны](#сильные-стороны)
6. [Слабые стороны и рекомендации](#слабые-стороны-и-рекомендации)
7. [Архитектурная диаграмма](#архитектурная-диаграмма)

---

## Общая архитектура

Проект состоит из **3 файлов**:

| Файл | Строк | Назначение |
|------|-------|------------|
| `model_engine.py` | 866 | Ядро: токенизаторы, модель, датасет, движок |
| `gui_app.py` | 611 | GUI на tkinter с 3 вкладками |
| `build_exe.bat` | 38 | Скрипт сборки .exe через PyInstaller |

**Зависимости**: `torch>=2.1`, `numpy>=1.24` — минимальный набор.

---

## model_engine.py

### 1. Токенизаторы (строки 31–187)

#### CharTokenizer
- Простой посимвольный токенизатор
- Строит словарь уникальных символов из текста
- Кодирование: маппинг символа → ID, fallback на ID пробела
- Сериализация: `to_dict()` / `from_dict()`

#### BPETokenizer
- Классический BPE (Byte/Char Pair Encoding), стиль GPT-2
- **Обучение**:
  1. Начинает с уникальных символов как базового словаря
  2. Находит самую частотную пару смежных токенов
  3. Сливает (merge) её, добавляет новый токен
  4. Повторяет до достижения `vocab_size` или отсутствия пар с частотой ≥ 2
- **Кэширование**: `._split_pieces()` разбивает текст по пробелам; результат кодирования каждого кэшируется (до 200K записей)
- **Кодирование**: жадно применяет merges от самого раннего по приоритету (алгоритм GPT-2)
- Сериализация: `merges` хранятся как `(a, b, new_id)`
- **Важно**: при восстановлении чужих символов (missing chars) во время `train()` в engine — это гарантирует, что весь текст представим

### 2. RMSNorm (строки 194–201)
- Root Mean Square Normalization — стандарт для LLaMA/Mistral/Qwen
- Формула: `x * rsqrt(mean(x²) + eps) * weight`
- Без learnable bias (только масштабный вес)
- Делает float64 для стабильности, затем возвращает тип

### 3. Attention (строки 206–303)
- **Multi-Head Self-Attention** с RoPE и опциональным GQA
- **RoPE (Rotary Position Embedding)**:
  - Предвычисленные `cos`/`sin` буферы для позиций `0..max_seq_len`
  - `_apply_rope()`: разделяет вектор на пары `(x0, x1)`, применяет вращение
- **GQA (Grouped-Query Attention)**:
  - `n_kv_heads` может быть меньше `n_heads`
  - KV-головы повторяются через `repeat_interleave` для матчинга с query
- **SDPA**: `F.scaled_dot_product_attention` (flash/memory-efficient kernels)
- **3 режима attention**:
  - `T == S`: полная последовательность с causal mask (обучение)
  - `T == 1`: инкрементальное декодирование (один токен за раз)
  - general case: чанковое декодирование с ручной маской
- **KV-cache**: хранилище ключей/значений для ускорения генерации
- **Инициализация Q/K/V/O**: `bias=False` (современная практика)

### 4. FeedForward / SwiGLU (строки 307–316)
```
FFN(x) = down(silu(gate(x)) * up(x))
```
- 3 линейных слоя (gate, up, down), все без bias
- SwiGLU вместо ReLU/GELU — стандарт LLaMA

### 5. TransformerBlock (строки 320–335)
- **Pre-norm** архитектура: `x + dropout(attn(norm(x))) + dropout(ffn(norm(x)))`
- RMSNorm до каждого подслоя
- Dropout опционален (Identity при `dropout=0`)

### 6. ModernTransformer (строки 340–412)
- **Полная стековая модель**:
  - Embedding → N × TransformerBlock → RMSNorm → Linear head
- **Weight Tying**: `head.weight = embedding.weight` — стандарт GPT-2/LLaMA
- **Инициализация**: `normal(0, 0.02)` для Linear/Embedding
- `forward()` возвращает logits для **всех позиций** (B, T, vocab_size)
  - Это dense next-token loss (стиль nanoGPT)
- `count_parameters()`: суммарное количество обучаемых параметров

### 7. CharDataset (строки 416–433)
- PyTorch `Dataset` со sliding window
- `(x[i..i+seq_len], x[i+1..i+seq_len+1])` — target это input, сдвинутый на 1
- Dense loss по всем позициям окна, не только последней

### 8. CosineWarmupScheduler (строки 437–462)
- Линейный warmup → Cosine decay
- `warmup_steps = min(200, total_steps // 10)`
- `min_lr = base_lr * 0.1`
- Обновляет learning rate после каждого батча

### 9. AuraLiteEngine — Движок (строки 467–866)

#### `__init__`
- Определяет устройство: CUDA или CPU
- Конфигурирует потоки PyTorch (OMP, MKL, OpenBLAS)

#### `train()` — ключевой метод
```python
params = {
    "lr", "epochs", "d_model", "d_ff", "n_heads", "n_layers",
    "seq_length", "batch_size", "dropout", "grad_clip", "weight_decay",
    "tokenizer", "bpe_vocab_size", "val_split", "use_compile",
    "autosave_every", "autosave_path", "continue_training"
}
```

**Процесс обучения**:
1. **Токенизация** — создаёт новый или использует существующий (continue_training)
2. **Модель** — ModernTransformer с заданными параметрами
3. **Оптимизатор** — AdamW (betas=0.9/0.95, weight_decay)
4. **Mixed Precision** — `torch.amp.GradScaler` (только CUDA)
5. **Dataset / DataLoader** — CharDataset с mini-batch, shuffle, `pin_memory`, `num_workers`
6. **torch.compile** — опционально, с graceful fallback
7. **Epoch loop**:
   - Forward → Loss (CrossEntropy на всех позициях) → Backward
   - Gradient clipping (grad_clip)
   - Optimizer step → Scheduler step
   - Validation каждые epoch (если есть val_split)
   - Autosave каждые N epoch
   - Interruptible через `stop_event`

**Особенности**:
- Градиенты сбрасываются `set_to_none=True` (экономия памяти)
- `non_blocking=True` для async GPU transfers
- Validation ограничен `max_batches=50` для скорости

#### `generate()` — Генерация
1. Кодирует seed-текст
2. Пропускает весь seed одним проходом (KV-cache активируется)
3. Генерирует токены один за другим:
   - `start_pos` сдвигается для использования кэша
   - Остановка при достижении `max_seq_len - 1`
4. `_sample_token()`:
   - Repetition penalty (CTRL-style, последние 64 токена)
   - Temperature scaling
   - Top-K filtering
   - Top-P (nucleus) filtering
   - Multinomial sampling

#### `save_model()` / `load_model()`
- Чекпоинт: `model_state`, `vocab_size`, `tokenizer`, `params_used`, архитектурные параметры
- Обратная совместимость: старые чекпоинты с полем `"chars"` загружаются как CharTokenizer

---

## gui_app.py

### Вкладка 1 — 🏋️ Training
| Элемент | Описание |
|---------|----------|
| Architecture & Hyperparameters | 10 параметров в 2 колонки |
| Tokenizer & Options | BPE/Char, BPE vocab, val split, torch.compile, continue, autosave |
| Run | Выбор файла, старт/стоп, прогресс-бар |
| Loss History | Текстовое поле с историей loss |

### Вкладка 2 — ✨ Generation
| Элемент | Описание |
|---------|----------|
| Sampling | Temperature (slider), Top-K, Top-P, Repetition Penalty |
| Prompt | Seed phrase, Length (slider 10–1000) |
| Output | Текстовое поле для результата |

### Вкладка 3 — 💾 Model
| Элемент | Описание |
|---------|----------|
| Save/Load | Кнопки сохранения и загрузки .pt |
| Model Info | Все параметры модели, токенизатор, устройство, val loss |

### Threading
- Обучение запускается в `threading.Thread(daemon=True)`
- `root.after()` для thread-safe обновлений UI
- `stop_event` для прерывания обучения

---

## build_exe.bat
- Устанавливает `torch`, `numpy`, `pyinstaller`
- Собирает `--onedir --noconsole` приложение
- Выход: `dist/AuraLite_AI_v2/AuraLite_AI_v2.exe`

---

## Сильные стороны

✅ **Современная архитектура** — LLaMA-style: RMSNorm, RoPE, SwiGLU, GQA, weight tying, KV-cache
✅ **Flash Attention** — через PyTorch SDPA, без ручных реализаций
✅ **Dense loss** — nanoGPT-style, эффективнее last-token-only
✅ **BPE токенизатор** — значительно лучше char-level
✅ **Полный pipeline** — tokenizer → dataset → model → optimizer → scheduler → generate
✅ **Mixed Precision** — AMP для CUDA
✅ **Cosine LR + Warmup** — стандарт для LLM
✅ **Grad clipping** — стабильность обучения
✅ **Threading** — GUI не блокируется при обучении
✅ **Interruptible** — можно остановить и сохранить прогресс
✅ **Continue training** — fine-tune существующей модели
✅ **Autosave** — чекпоинты каждые N epoch
✅ **Обратная совместимость** — старые чекпоинты загружаются
✅ **CPU multithreading** — OMP, MKL, OpenBLAS настроены
✅ **torch.compile** — опциональное ускорение
✅ **Минимальные зависимости** — только torch + numpy

---

## Слабые стороны и рекомендации

### 🔴 Критические

1. **Нет проверки `d_model % n_heads == 0`** в GUI — пользователь может ввести несовместимые значения
2. **BPE encoding** — если символы из текста не попали в начальный vocab (при `train()` на сэмпле 2M), они маппятся на пробел, что может терять информацию
3. **Нет logging** — всё в GUI, нет file logging для отладки
4. **Генерация** — `generate()` блокируется в thread, но нет индикации прогресса генерации (токен за токеном)

### 🟡 Средние

5. **Валидация** — `val_split` по умолчанию 0.1, но при маленьких файлах может быть 0
6. **No early stopping** — модель может overfit без механизма ранней остановки
7. **BPE training** — нет прогресс-индикатора для обучения токенизатора на больших файлах
8. **Memory** — для больших файлов (>50MB) весь текст грузится в память; нет chunking

### 🟢 Минорные

9. **GUI styling** — базовый ttk, можно улучшить с `customtkinter` или PyQt
10. **Нет тестов** — unit/integration tests отсутствуют
11. **Нет CLI** — только GUI, нет командной строки для headless training
12. **`build_exe.bat`** — только Windows, нет Linux/macOS сборки

---

## Архитектурная диаграмма

```
┌─────────────────────────────────────────────────────────┐
│                    GUI (tkinter)                        │
│  ┌───────────┐  ┌────────────┐  ┌──────────┐           │
│  │ Training  │  │ Generation │  │ Model    │           │
│  │  Tab      │  │   Tab      │  │  Tab     │           │
│  └─────┬─────┘  └──────┬─────┘  └────┬─────┘           │
│        │               │             │                  │
│  threading.Thread      │             │                  │
└────────┼───────────────┼─────────────┼──────────────────┘
         │               │             │
         ▼               ▼             ▼
┌─────────────────────────────────────────────────────────┐
│                AuraLiteEngine                           │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   train()   │  │  generate()  │  │  save/load()  │  │
│  └──────┬──────┘  └──────┬───────┘  └───────┬───────┘  │
│         │                │                   │          │
│         ▼                ▼                   ▼          │
│  ┌──────────────────────────────────────────────────┐   │
│  │             ModernTransformer                    │   │
│  │  Embedding → N×[RMSNorm→Attn→RMSNorm→SwiGLU] → │   │
│  │  RMSNorm → Linear Head (weight tied)            │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐      │
│  │CharTok/B │  │CharData  │  │CosineWarmup      │      │
│  │PETok     │  │set       │  │Scheduler         │      │
│  └──────────┘  └──────────┘  └──────────────────┘      │
└─────────────────────────────────────────────────────────┘
         │                │                   │
         ▼                ▼                   ▼
   Token IDs       (x, y) batches      LR scheduling
```

---

## Вывод

**AuraLite AI v2** — это качественно спроектированный образовательный LLM-фреймворк, который:

- Демонстрирует **современные архитектурные паттерны** 2025–2026 года
- Имеет **полный рабочий pipeline** от данных до генерации
- Поддерживает **GPU и CPU** с автоматическим fallback
- Предоставляет **интерактивный GUI** с полным контролем параметров
- **Рекомендуется** для обучения, экспериментов и демонстрации Transformer-архитектуры
