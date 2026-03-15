from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Set, Tuple
import random


Pair = Tuple[str, str]


@dataclass(frozen=True)
class NegativeSamplingConfig:
    negative_prob: float
    reverse_negative_ratio: float
    random_negative_ratio: float
    max_easy_retries: int = 10

    @property
    def reverse_negative_prob(self) -> float:
        total = self.reverse_negative_ratio + self.random_negative_ratio
        return self.reverse_negative_ratio / total if total > 0 else 0.0

    @property
    def random_negative_prob(self) -> float:
        total = self.reverse_negative_ratio + self.random_negative_ratio
        return self.random_negative_ratio / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "negative_prob": self.negative_prob,
            "reverse_negative_ratio": self.reverse_negative_ratio,
            "random_negative_ratio": self.random_negative_ratio,
            "reverse_negative_prob": self.reverse_negative_prob,
            "random_negative_prob": self.random_negative_prob,
            "max_easy_retries": self.max_easy_retries,
        }


def _validate_unit_interval(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _validate_non_negative(name: str, value: float) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def build_negative_sampling_config(
    negative_prob: float = 0.5,
    hard_negative_prob: float = 0.5,
    reverse_negative_ratio: Optional[float] = None,
    random_negative_ratio: Optional[float] = None,
    max_easy_retries: int = 10,
) -> NegativeSamplingConfig:
    negative_prob = _validate_unit_interval("negative_prob", negative_prob)
    max_easy_retries = max(1, int(max_easy_retries))

    if reverse_negative_ratio is None and random_negative_ratio is None:
        hard_negative_prob = _validate_unit_interval("hard_negative_prob", hard_negative_prob)
        reverse_negative_ratio = hard_negative_prob
        random_negative_ratio = 1.0 - hard_negative_prob
    else:
        reverse_negative_ratio = _validate_non_negative(
            "reverse_negative_ratio", 1.0 if reverse_negative_ratio is None else reverse_negative_ratio
        )
        random_negative_ratio = _validate_non_negative(
            "random_negative_ratio", 1.0 if random_negative_ratio is None else random_negative_ratio
        )

    if reverse_negative_ratio == 0.0 and random_negative_ratio == 0.0:
        raise ValueError("reverse_negative_ratio and random_negative_ratio cannot both be 0")

    return NegativeSamplingConfig(
        negative_prob=negative_prob,
        reverse_negative_ratio=reverse_negative_ratio,
        random_negative_ratio=random_negative_ratio,
        max_easy_retries=max_easy_retries,
    )


def format_negative_sampling_summary(config: NegativeSamplingConfig) -> str:
    return (
        f"negative_prob={config.negative_prob:.3f}, "
        f"reverse:random={config.reverse_negative_ratio:g}:{config.random_negative_ratio:g}, "
        f"reverse_share={config.reverse_negative_prob:.3f}, "
        f"random_share={config.random_negative_prob:.3f}, "
        f"max_easy_retries={config.max_easy_retries}"
    )


def _sample_random_negative_value(
    key_text: str,
    original_value: str,
    value_pool: Sequence[str],
    valid_pairs_set: Set[Pair],
    config: NegativeSamplingConfig,
    rng=random,
) -> Optional[str]:
    if not value_pool:
        return None

    last_candidate: Optional[str] = None
    for _ in range(config.max_easy_retries):
        candidate_value = rng.choice(value_pool)
        last_candidate = candidate_value
        if candidate_value != original_value and (key_text, candidate_value) not in valid_pairs_set:
            return candidate_value

    if last_candidate is not None and (key_text, last_candidate) not in valid_pairs_set:
        return last_candidate
    return None


def _sample_global_negative_pair(
    pair_pool: Sequence[Pair],
    value_pool: Sequence[str],
    valid_pairs_set: Set[Pair],
    config: NegativeSamplingConfig,
    rng=random,
) -> Optional[Pair]:
    if not pair_pool or not value_pool:
        return None

    tries = max(config.max_easy_retries * 3, 30)
    last_pair: Optional[Pair] = None
    for _ in range(tries):
        random_key, _ = rng.choice(pair_pool)
        candidate_value = rng.choice(value_pool)
        last_pair = (random_key, candidate_value)
        if last_pair not in valid_pairs_set:
            return last_pair

    if last_pair is not None and last_pair not in valid_pairs_set:
        return last_pair
    return None


def sample_kv_nsp_text_pair(
    key_text: str,
    value_text: str,
    value_pool: Sequence[str],
    valid_pairs_set: Set[Pair],
    config: NegativeSamplingConfig,
    pair_pool: Optional[Sequence[Pair]] = None,
    rng=random,
) -> Tuple[str, str, int, str]:
    if rng.random() >= config.negative_prob:
        return key_text, value_text, 1, "positive"

    swapped_key, swapped_value = value_text, key_text
    total_ratio = config.reverse_negative_ratio + config.random_negative_ratio
    pick = rng.random() * total_ratio
    prefer_reverse = pick < config.reverse_negative_ratio

    if prefer_reverse:
        if (swapped_key, swapped_value) not in valid_pairs_set:
            return swapped_key, swapped_value, 0, "reverse"

        random_value = _sample_random_negative_value(
            key_text=key_text,
            original_value=value_text,
            value_pool=value_pool,
            valid_pairs_set=valid_pairs_set,
            config=config,
            rng=rng,
        )
        if random_value is not None:
            return key_text, random_value, 0, "random_fallback"

    else:
        random_value = _sample_random_negative_value(
            key_text=key_text,
            original_value=value_text,
            value_pool=value_pool,
            valid_pairs_set=valid_pairs_set,
            config=config,
            rng=rng,
        )
        if random_value is not None:
            return key_text, random_value, 0, "random"

        if (swapped_key, swapped_value) not in valid_pairs_set:
            return swapped_key, swapped_value, 0, "reverse_fallback"

    if pair_pool is not None:
        global_neg = _sample_global_negative_pair(
            pair_pool=pair_pool,
            value_pool=value_pool,
            valid_pairs_set=valid_pairs_set,
            config=config,
            rng=rng,
        )
        if global_neg is not None:
            gk, gv = global_neg
            return gk, gv, 0, "global_random_fallback"

    return key_text, value_text, 1, "positive_fallback"