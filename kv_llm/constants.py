"""Shared constants for KV-LLM experiments."""

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"

FEATURES = [
    "conf_avg",
    "conf_min",
    "conf_var_log",
    "conf_gap",
    "punct_err_ratio",
    "char_break_ratio",
    "align_score",
]

NUM_BINS = {
    "conf_avg": 64,
    "conf_min": 64,
    "conf_var_log": 32,
    "conf_gap": 32,
    "punct_err_ratio": 16,
    "char_break_ratio": 32,
    "align_score": 64,
}

PERFECT_VALUES = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

SENTINEL_PREFIX = "<extra_id_"
KV_NSP_SEP_TOKEN = "<kv_sep>"


def build_sentinel_tokens(num_sentinels: int = 100) -> list[str]:
    return [f"{SENTINEL_PREFIX}{i}>" for i in range(num_sentinels)]
