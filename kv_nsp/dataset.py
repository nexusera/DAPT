"""
KV-NSP 数据集定义
-----------------
本文件实现一个用于“键值配对判断”二分类任务的自定义 Dataset。
核心特点：
1. 解析 Label Studio 导出的标注 JSON，抽取正向的 Key→Value 配对。
2. 在 __getitem__ 中动态负采样（hard negative + easy negative），让模型同时学习顺序与语义。
3. 直接返回可供 Hugging Face Trainer 使用的张量字典，包括 input_ids / token_type_ids / attention_mask / labels。
"""

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class KVDataset(Dataset):
    """
    KV-NSP 数据集

    - 正样本：标注中 Key → Value 的有向关系。
    - 负样本：在 __getitem__ 中按概率动态生成：
        * Hard negative（倒序）：交换 Key / Value 位置，强制模型学习“顺序”。
        * Easy negative（随机值）：保持 Key，不相关的 Value 替换。

    参数说明：
    data_files: JSON 文件列表（Label Studio 导出格式）
    tokenizer: 预训练分词器，需支持句对输入以生成 token_type_ids
    max_length: 句对的最大长度（使用 padding='max_length'，保证 batch 对齐）
    negative_prob: 负样本概率，例如 0.5 表示 50% 采样为负样本
    hard_negative_prob: 在负样本中选择“倒序”策略的概率；剩余使用“随机值”
    seed: 随机种子，便于复现
    """

    def __init__(
        self,
        data_files: Sequence[Path],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
        negative_prob: float = 0.5,
        hard_negative_prob: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.negative_prob = negative_prob
        self.hard_negative_prob = hard_negative_prob

        random.seed(seed)

        self.pairs: List[Tuple[str, str]] = []  # [(key_text, value_text)]
        self.value_pool: List[str] = []  # 仅存储 value 文本，用于 easy negative

        self._load_all_files(data_files)
        if not self.pairs:
            print(f"Warning: 未能在数据文件中找到任何键值对 (Data files: {data_files})。数据集将为空。")
            # 允许空数据集，避免 Dummy 初始化时报错
            # raise ValueError("未能在数据文件中找到任何键值对，请检查标注或路径是否正确。")

    # ------------------------------------------------------------------ #
    # 数据加载与解析
    # ------------------------------------------------------------------ #
    def _load_all_files(self, data_files: Sequence[Path]) -> None:
        for path in data_files:
            records = self._read_json(path)
            for sample in records:
                for key_text, value_text in self._extract_pairs(sample):
                    key_text = key_text.strip()
                    value_text = value_text.strip()
                    if not key_text or not value_text:
                        continue
                    self.pairs.append((key_text, value_text))
                    self.value_pool.append(value_text)

    @staticmethod
    def _read_json(path: Path) -> Iterable[Dict]:
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _extract_pairs(sample: Dict) -> List[Tuple[str, str]]:
        """
        从单条 Label Studio 任务中抽取 Key→Value 正样本。
        解析规则：
        - 仅使用未取消（was_cancelled=False）的最新标注。
        - result 内 type=labels 存实体，type=relation 存关系。
        - 仅保留 label 为 “键名” 与 “值” 的实体。
        """
        annotations = sample.get("annotations", [])
        valid_annos = [a for a in annotations if not a.get("was_cancelled")]
        if not valid_annos:
            return []

        latest = valid_annos[-1]
        results = latest.get("result", [])

        entities: Dict[str, Dict[str, str]] = {}
        relations: List[Tuple[str, str]] = []

        for res in results:
            res_type = res.get("type")
            if res_type == "labels":
                labels = res.get("value", {}).get("labels", [])
                if not labels:
                    continue
                label = labels[0]
                if label not in ("键名", "值"):
                    continue
                text = res.get("value", {}).get("text", "")
                if not text:
                    continue
                entities[res.get("id")] = {"label": label, "text": text}
            elif res_type == "relation":
                from_id = res.get("from_id")
                to_id = res.get("to_id")
                if from_id and to_id:
                    relations.append((from_id, to_id))

        pairs: List[Tuple[str, str]] = []
        for from_id, to_id in relations:
            key = entities.get(from_id)
            value = entities.get(to_id)
            if key and value and key["label"] == "键名" and value["label"] == "值":
                pairs.append((key["text"], value["text"]))
        return pairs

    # ------------------------------------------------------------------ #
    # Dataset 接口
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        动态构造一条训练样本：
        - 50% 概率（可配）使用正样本 Label=1。
        - 50% 概率（可配）构造负样本 Label=0：
            * hard negative：Key/Value 位置互换。
            * easy negative：Key 保持，Value 随机替换为无关值。
        """
        key_text, value_text = self.pairs[idx]
        label = 1

        if random.random() < self.negative_prob:
            label = 0
            if random.random() < self.hard_negative_prob:
                # 倒序：Value 放前面，Key 放后面
                key_text, value_text = value_text, key_text
            else:
                # 随机替换 Value，若多次抽样仍与原值相同则直接使用抽到的值
                for _ in range(5):
                    random_value = random.choice(self.value_pool)
                    if random_value != value_text:
                        value_text = random_value
                        break
                else:
                    value_text = random.choice(self.value_pool)

        encoding = self.tokenizer(
            key_text,
            value_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


