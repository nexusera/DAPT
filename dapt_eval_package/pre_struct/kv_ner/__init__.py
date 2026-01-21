# -*- coding: utf-8 -*-
"""
Key-value extraction via BERT+CRF sequence labelling.

The module mirrors the EBQA pipeline but replaces the QA backbone with a
token-classification model so that all fields can be extracted in a single
forward pass.  Training and inference are driven by a single JSON config
(`kv_ner_config.json`) that follows the same conventions as
`pre_struct/ebqa/ebqa_config.json`.
"""

from __future__ import annotations

__all__ = [
    "config_io",
    "data_utils",
    "dataset",
    "metrics",
    "modeling",
    "train",
    "predict",
]
