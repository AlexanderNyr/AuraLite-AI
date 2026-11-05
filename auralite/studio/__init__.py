"""AuraLite Studio (2.19): the desktop app on Qt 6, and its toolkit-free core.

The core needs nothing beyond AuraLite itself (the CLI's ``auralite doctor``
uses it too):

* :mod:`.hardware` - memory, processor, GPU, disk, a speed test;
* :mod:`.catalog` - base models worth fine-tuning on a PC;
* :mod:`.advisor` - which model, method and settings fit, how long a run takes;
* :mod:`.datacheck` - what a data file holds, before training on it;
* :mod:`.recipes` - training parameters from a few choices;
* :mod:`.errors` - what went wrong, in words a user can act on;
* :mod:`.progress`, :mod:`.logs`, :mod:`.settings`, :mod:`.updates`.

The window (:mod:`.app`, :mod:`.qt`) needs PySide6:
``pip install "auralite-ai[studio]"``, then ``auralite-studio``.
"""
from __future__ import annotations

from .advisor import (ModelFit, PresetFit, Recommendation, estimate_finetune, memory_budget,
                      recommend_finetune, recommend_pretrain, tokens_per_second)
from .catalog import CATALOG, BaseModel, find_model
from .datacheck import DataReport, inspect_data
from .errors import FriendlyError, explain_error
from .hardware import Hardware, auralite_home, detect_hardware, measure_gflops
from .recipes import (FineTuneSettings, SettingsError, TrainSettings, finetune_params,
                      pretrain_params)

__all__ = ["CATALOG", "BaseModel", "DataReport", "FineTuneSettings", "FriendlyError",
           "Hardware", "ModelFit", "PresetFit", "Recommendation", "SettingsError",
           "TrainSettings", "auralite_home", "detect_hardware", "estimate_finetune",
           "explain_error", "find_model", "finetune_params", "inspect_data", "measure_gflops",
           "memory_budget", "pretrain_params", "recommend_finetune", "recommend_pretrain",
           "tokens_per_second"]
