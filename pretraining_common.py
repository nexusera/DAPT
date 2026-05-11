"""
pretraining_common.py
---------------------
C2: 预训练脚本公共组件，消除 train_dapt_*.py 之间 ~60–70% 的重复代码。

当前已提取的组件：
  - PerplexityCallback   : 评估时打印 PPL，原先在 4 个脚本中各自重复。
  - PrecomputedWWMCollator: 基于预计算 word_ids 的全词掩码 collator，
                             原先在 train_dapt_distributed.py、train_dapt_kvmlm.py
                             中各自重复定义。

待后续继续提取（当前各文件实现存在细微差异，需逐一核对后合并）：
  - MLMStageCollator      (train_dapt_macbert_staged/no_noise/no_nsp, train_dapt_staged)
  - DynamicNSPDataset     (train_dapt_macbert_staged/no_mlm/no_noise, train_dapt_staged)
  - RobertaModelWithNoise (train_dapt_distributed/staged/hybrid_masking/mtl)

使用方式（在各 train_dapt_*.py 顶部）：
    from pretraining_common import PerplexityCallback, PrecomputedWWMCollator
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from transformers import TrainerCallback


# ---------------------------------------------------------------------------
# PerplexityCallback
# ---------------------------------------------------------------------------

class PerplexityCallback(TrainerCallback):
    """评估结束时根据 eval_loss 计算并打印 Perplexity（PPL），仅主进程输出。

    原先在以下 4 个脚本中逐字重复：
      train_dapt_distributed.py, train_dapt_kvmlm.py,
      train_dapt_mlm.py, train_dapt_base_mlm_resize.py
    """

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if state.is_local_process_zero:
            loss = metrics.get("eval_loss")
            if loss:
                ppl = math.exp(loss)
                print(f"\n[Evaluation] Perplexity (PPL): {ppl:.4f}\n")
                metrics["perplexity"] = ppl


# ---------------------------------------------------------------------------
# PrecomputedWWMCollator
# ---------------------------------------------------------------------------

@dataclass
class PrecomputedWWMCollator:
    """基于预计算 word_ids 的全词掩码（Whole-Word Masking）Collator。

    原先在 train_dapt_distributed.py 与 train_dapt_kvmlm.py 中各自重复定义，
    合并为此规范版本（采用 kvmlm 版的防御性 `word_ids is None` 检查）。

    Args:
        tokenizer:       HuggingFace tokenizer，必填。
        mlm_probability: 掩码比例，默认 0.15。
        max_seq_len:     padding 目标长度，默认 512。
        pad_to_multiple_of: 对齐到 8 的倍数以提升硬件利用率，默认 8。
    """

    tokenizer: Any
    mlm_probability: float = 0.15
    max_seq_len: int = 512
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_input_ids = [f["input_ids"] for f in features]
        batch_word_ids = [f.get("word_ids") for f in features]
        batch = self.tokenizer.pad(
            {"input_ids": batch_input_ids},
            padding="longest",
            max_length=self.max_seq_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        for i in range(len(features)):
            word_ids = batch_word_ids[i]
            if word_ids is None:
                # word_ids 缺失时退化为随机 token 级掩码
                continue
            current_ids = input_ids[i]
            mapping: Dict[int, List[int]] = {}
            for idx, wid in enumerate(word_ids):
                if wid is None:
                    continue
                if idx >= len(current_ids):
                    break
                mapping.setdefault(wid, []).append(idx)

            unique_words = list(mapping.keys())
            if not unique_words:
                continue
            num_to_mask = max(1, int(len(unique_words) * self.mlm_probability))
            masked_words = set(random.sample(unique_words, num_to_mask))
            mask_indices = torch.zeros(len(current_ids), dtype=torch.bool)
            for wid in masked_words:
                for idx in mapping[wid]:
                    mask_indices[idx] = True

            special_tokens_mask = torch.tensor(
                self.tokenizer.get_special_tokens_mask(
                    current_ids.tolist(), already_has_special_tokens=True
                ),
                dtype=torch.bool,
            )
            mask_indices.masked_fill_(special_tokens_mask, value=False)
            if self.tokenizer.pad_token_id is not None:
                mask_indices.masked_fill_(
                    current_ids == self.tokenizer.pad_token_id, value=False
                )

            probability_matrix[i, :] = 0.0
            probability_matrix[i, mask_indices] = 1.0

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        batch["input_ids"] = input_ids
        batch["labels"] = labels
        return batch
