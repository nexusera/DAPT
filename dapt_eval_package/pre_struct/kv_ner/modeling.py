# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

# Compatibility fix for transformers versions
# flash_attn stub (in ./flash_attn/) helps transformers handle optional imports
os.environ.setdefault('TRANSFORMERS_DISABLE_TELEMETRY', '1')

import torch
from torch import nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import AutoConfig, AutoModel
from .noise_utils import NUM_BINS, FEATURES
try:
    from noise_fusion import ContinuousNoiseProjector, build_feature_ranges, uses_bucket_noise, uses_continuous_noise
except Exception:  # pragma: no cover
    import sys
    _ROOT = Path(__file__).resolve().parents[3]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from noise_fusion import ContinuousNoiseProjector, build_feature_ranges, uses_bucket_noise, uses_continuous_noise

try:
    from model_path_conf import DEFAULT_MODEL_PATH  # type: ignore
except Exception:  # pragma: no cover - fallback for tests
    DEFAULT_MODEL_PATH = "bert-base-multilingual-cased"

logger = logging.getLogger(__name__)


class BertCrfTokenClassifier(nn.Module):
    def __init__(
        self,
        *,
        model_name_or_path: Optional[str],
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        unfreeze_last_n_layers: Optional[int] = None,
        # BiLSTM configuration
        use_bilstm: bool = False,
        lstm_hidden_size: Optional[int] = None,
        lstm_num_layers: int = 1,
        lstm_dropout: float = 0.0,
        # noise embedding fusion
        use_noise: bool = False,
        noise_embed_dim: int = 16,
        noise_mode: str = "bucket",
        noise_mlp_hidden_dim: Optional[int] = None,
        noise_bin_edges: Optional[Dict[str, list]] = None,
        # boundary-aware loss controls
        boundary_loss_weight: float = 0.0,
        boundary_positive_weight: float = 1.0,
        include_hospital_boundary: bool = True,
        # end boundary auxiliary head
        end_boundary_loss_weight: float = 0.0,
        end_boundary_positive_weight: float = 1.0,
        # token-level CE auxiliary loss to emphasize classes (e.g., VALUE)
        token_ce_loss_weight: float = 0.0,
        # label smoothing for CE losses
        token_ce_label_smoothing: float = 0.0,
        boundary_ce_label_smoothing: float = 0.0,
        # class weight for VALUE in token CE
        token_ce_value_class_weight: float = 3.0,
        # class weight for KEY in token CE
        token_ce_key_class_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if "O" not in label2id:
            raise ValueError("label2id must include 'O'")
        self.label2id = dict(label2id)
        self.id2label = dict(id2label)
        self.num_labels = len(label2id)
        self.freeze_encoder = bool(freeze_encoder)
        self.unfreeze_last_n_layers = (
            int(unfreeze_last_n_layers) if unfreeze_last_n_layers is not None else None
        )
        self.config = AutoConfig.from_pretrained(
            model_name_or_path or DEFAULT_MODEL_PATH,
            output_hidden_states=False,
        )
        self.model_name_or_path = model_name_or_path or DEFAULT_MODEL_PATH
        self.bert = AutoModel.from_pretrained(self.model_name_or_path, config=self.config)
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # Noise fusion
        self.use_noise = bool(use_noise)
        self.noise_embed_dim = int(noise_embed_dim)
        self.noise_mode = str(getattr(self.config, "noise_mode", noise_mode) or noise_mode or "bucket").lower()
        self.noise_mlp_hidden_dim = int(
            getattr(self.config, "noise_mlp_hidden_dim", noise_mlp_hidden_dim or 0) or (noise_mlp_hidden_dim or 0)
        ) or None
        self.noise_bin_edges = noise_bin_edges or getattr(self.config, "noise_bin_edges", {}) or {}
        self.noise_embeddings = None
        self.noise_proj = None
        self.noise_dropout = None
        self.noise_projector = None
        if self.use_noise:
            if uses_bucket_noise(self.noise_mode):
                emb_layers = []
                for feat in FEATURES:
                    nbin = int(NUM_BINS[feat])
                    emb_layers.append(nn.Embedding(num_embeddings=nbin + 1, embedding_dim=self.noise_embed_dim))
                self.noise_embeddings = nn.ModuleList(emb_layers)
                self.noise_proj = nn.Linear(len(FEATURES) * self.noise_embed_dim, hidden_size)
                self.noise_dropout = nn.Dropout(dropout)
            elif uses_continuous_noise(self.noise_mode):
                self.noise_projector = ContinuousNoiseProjector(
                    hidden_size,
                    mode=self.noise_mode,
                    dropout=dropout,
                    mlp_hidden_dim=self.noise_mlp_hidden_dim,
                    feature_ranges=build_feature_ranges(self.noise_bin_edges),
                )
            else:
                raise ValueError(f"Unsupported noise_mode: {self.noise_mode}")
        
        # BiLSTM layer (optional)
        self.use_bilstm = bool(use_bilstm)
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout
        
        if self.use_bilstm:
            # If lstm_hidden_size not specified, use half of BERT hidden_size
            lstm_hidden = lstm_hidden_size or (hidden_size // 2)
            self.lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=lstm_hidden,
                num_layers=lstm_num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=lstm_dropout if lstm_num_layers > 1 else 0.0
            )
            # BiLSTM output size = 2 * lstm_hidden
            classifier_input_size = lstm_hidden * 2
            logger.info(f"Using BiLSTM: {hidden_size} -> {lstm_hidden}*2 = {classifier_input_size}")
        else:
            self.lstm = None
            classifier_input_size = hidden_size
        
        self.classifier = nn.Linear(classifier_input_size, self.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
        # Apply BIO transition constraints as a prior to stabilize training
        try:
            self._apply_bio_constraints()
        except Exception:
            # Do not fail if constraints cannot be applied; continue with learnable transitions
            pass
        # Auxiliary heads for boundaries (use classifier_input_size after BiLSTM)
        # start boundary: B-*
        self.boundary_classifier = nn.Linear(classifier_input_size, 2)
        # end boundary: last token of any entity
        self.end_boundary_classifier = nn.Linear(classifier_input_size, 2)

        # Boundary weighting controls
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.boundary_positive_weight = float(boundary_positive_weight)
        self.include_hospital_boundary = bool(include_hospital_boundary)
        self.end_boundary_loss_weight = float(end_boundary_loss_weight)
        self.end_boundary_positive_weight = float(end_boundary_positive_weight)
        self.token_ce_loss_weight = float(token_ce_loss_weight)
        self.token_ce_label_smoothing = float(token_ce_label_smoothing)
        self.boundary_ce_label_smoothing = float(boundary_ce_label_smoothing)
        self.token_ce_value_class_weight = float(token_ce_value_class_weight)
        self.token_ce_key_class_weight = float(token_ce_key_class_weight)

        # Precompute which label ids correspond to boundaries we care about
        self.boundary_label_ids = self._compute_boundary_label_ids()
        self._configure_trainable_params(freeze_encoder, unfreeze_last_n_layers)

    def _configure_trainable_params(
        self,
        freeze_encoder: bool,
        unfreeze_last_n_layers: Optional[int],
    ) -> None:
        if freeze_encoder and unfreeze_last_n_layers:
            logger.warning(
                "Both freeze_encoder and unfreeze_last_n_layers are set; "
                "freeze_encoder takes precedence."
            )
        if freeze_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False
            return

        if unfreeze_last_n_layers is None:
            return

        n_layers = getattr(self.bert, "encoder", None)
        if not hasattr(self.bert, "encoder") or not hasattr(self.bert.encoder, "layer"):
            logger.warning("Backbone model does not expose encoder layers; skipping layer freezing.")
            return

        total_layers = len(self.bert.encoder.layer)
        n = max(0, min(int(unfreeze_last_n_layers), total_layers))
        for param in self.bert.parameters():
            param.requires_grad = False
        if n == 0:
            logger.info("All encoder layers frozen (unfreeze_last_n_layers=0).")
            return
        for layer in self.bert.encoder.layer[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        if hasattr(self.bert, "pooler") and self.bert.pooler is not None:
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
        logger.info("Unfroze last %d encoder layers out of %d.", n, total_layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        noise_ids: Optional[torch.Tensor] = None,
        noise_values: Optional[torch.Tensor] = None,
    ):
        # 显式生成 position_ids 以避免 RoBERTa 内部计算错误
        seq_length = input_ids.shape[1]
        device = input_ids.device
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(input_ids.shape[0], -1)
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,  # 显式传入正确的 position_ids
        )
        sequence_output = self.dropout(outputs[0])
        # Fuse noise embeddings if provided
        if self.use_noise:
            if uses_bucket_noise(self.noise_mode) and (noise_ids is not None) and self.noise_embeddings is not None:
                noise_vecs = []
                for i, emb in enumerate(self.noise_embeddings):
                    ids_i = noise_ids[:, :, i].clamp(min=0, max=emb.num_embeddings - 1)
                    noise_vecs.append(emb(ids_i))
                noise_cat = torch.cat(noise_vecs, dim=-1)
                noise_h = self.noise_proj(noise_cat)
                noise_h = self.noise_dropout(noise_h)
                sequence_output = sequence_output + noise_h
            elif uses_continuous_noise(self.noise_mode) and (noise_values is not None) and self.noise_projector is not None:
                noise_h = self.noise_projector(noise_values.to(sequence_output.device, dtype=torch.float32))
                sequence_output = sequence_output + noise_h
        
        # Optional BiLSTM layer
        if self.use_bilstm and self.lstm is not None:
            # 保存原始序列长度
            original_seq_length = sequence_output.size(1)
            
            # Pack sequence for efficiency (handle padding)
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1).cpu()
                sequence_output = nn.utils.rnn.pack_padded_sequence(
                    sequence_output, lengths, batch_first=True, enforce_sorted=False
                )
            lstm_output, _ = self.lstm(sequence_output)
            # Unpack - 指定total_length以恢复原始长度
            if attention_mask is not None:
                lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                    lstm_output, batch_first=True, total_length=original_seq_length
                )
            sequence_output = self.dropout(lstm_output)
        
        emissions = self.classifier(sequence_output)
        mask = attention_mask.bool() if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)

        if labels is not None:
            nll = -self.crf(emissions, labels, mask=mask, reduction="mean")
            # Optional boundary-aware auxiliary loss to emphasize B- tokens
            if self.boundary_loss_weight > 0:
                boundary_targets = self._boundary_targets_from_labels(labels)
                logits_boundary = self.boundary_classifier(sequence_output)
                active = mask.view(-1)
                logits_flat = logits_boundary.view(-1, 2)[active]
                targets_flat = boundary_targets.view(-1)[active]
                if logits_flat.numel() > 0:
                    weight = torch.tensor(
                        [1.0, max(1.0, self.boundary_positive_weight)],
                        dtype=logits_flat.dtype,
                        device=logits_flat.device,
                    )
                    ce = F.cross_entropy(
                        logits_flat,
                        targets_flat,
                        weight=weight,
                        label_smoothing=max(0.0, self.boundary_ce_label_smoothing),
                    )
                    nll = nll + self.boundary_loss_weight * ce
            # Optional token-level CE auxiliary loss over emissions
            if self.token_ce_loss_weight > 0:
                class_weights = self._class_weights_tensor(emissions.device, emissions.dtype)
                active = mask.view(-1)
                logits_flat = emissions.view(-1, emissions.size(-1))[active]
                targets_flat = labels.view(-1)[active]
                if logits_flat.numel() > 0:
                    ce_tok = F.cross_entropy(
                        logits_flat,
                        targets_flat,
                        weight=class_weights,
                        label_smoothing=max(0.0, self.token_ce_label_smoothing),
                    )
                    nll = nll + self.token_ce_loss_weight * ce_tok
            # Optional end-boundary loss
            if self.end_boundary_loss_weight > 0:
                end_targets = self._end_boundary_targets_from_labels(labels)
                logits_end = self.end_boundary_classifier(sequence_output)
                active = mask.view(-1)
                logits_flat = logits_end.view(-1, 2)[active]
                targets_flat = end_targets.view(-1)[active]
                if logits_flat.numel() > 0:
                    weight_e = torch.tensor(
                        [1.0, max(1.0, self.end_boundary_positive_weight)],
                        dtype=logits_flat.dtype,
                        device=logits_flat.device,
                    )
                    ce_end = F.cross_entropy(
                        logits_flat,
                        targets_flat,
                        weight=weight_e,
                        label_smoothing=max(0.0, self.boundary_ce_label_smoothing),
                    )
                    nll = nll + self.end_boundary_loss_weight * ce_end
            return nll
        return self.crf.decode(emissions, mask=mask)

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        noise_ids: Optional[torch.Tensor] = None,
        noise_values: Optional[torch.Tensor] = None,
    ):
        return self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=None,
            noise_ids=noise_ids,
            noise_values=noise_values,
        )

    def save_pretrained(self, output_dir: str, use_safetensors: bool = True) -> None:
        """保存模型权重"""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        state_dict = self.state_dict()
        
        if use_safetensors:
            try:
                from safetensors.torch import save_file
                save_file(state_dict, path / "model.safetensors")
                logger.info("Saved model weights in safetensors format")
            except ImportError:
                logger.warning("safetensors not installed, falling back to pytorch format")
                torch.save(state_dict, path / "pytorch_model.bin")
        else:
            torch.save(state_dict, path / "pytorch_model.bin")
        
        metadata = {
            "model_name_or_path": self.model_name_or_path,
            "dropout": self.dropout.p if isinstance(self.dropout, nn.Dropout) else None,
            "freeze_encoder": self.freeze_encoder,
            "unfreeze_last_n_layers": self.unfreeze_last_n_layers,
            "use_bilstm": self.use_bilstm,
            "lstm_hidden_size": self.lstm_hidden_size,
            "lstm_num_layers": self.lstm_num_layers,
            "lstm_dropout": self.lstm_dropout,
            "use_noise": self.use_noise,
            "noise_embed_dim": self.noise_embed_dim,
            "noise_mode": self.noise_mode,
            "noise_mlp_hidden_dim": self.noise_mlp_hidden_dim,
            "noise_bin_edges": self.noise_bin_edges,
            "boundary_loss_weight": self.boundary_loss_weight,
            "boundary_positive_weight": self.boundary_positive_weight,
            "include_hospital_boundary": self.include_hospital_boundary,
            "token_ce_loss_weight": self.token_ce_loss_weight,
            "token_ce_label_smoothing": self.token_ce_label_smoothing,
            "boundary_ce_label_smoothing": self.boundary_ce_label_smoothing,
            "token_ce_value_class_weight": self.token_ce_value_class_weight,
            "token_ce_key_class_weight": self.token_ce_key_class_weight,
            "end_boundary_loss_weight": self.end_boundary_loss_weight,
            "end_boundary_positive_weight": self.end_boundary_positive_weight,
        }
        (path / "model_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        (path / "label2id.json").write_text(json.dumps(self.label2id, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, model_dir: str, map_location=None) -> "BertCrfTokenClassifier":
        path = Path(model_dir)
        label_map_path = path / "label2id.json"
        if not label_map_path.is_file():
            raise FileNotFoundError(f"Missing label map at {label_map_path}")
        label2id = json.loads(label_map_path.read_text(encoding="utf-8"))
        config_path = path / "model_config.json"
        model_cfg = {}
        if config_path.is_file():
            model_cfg = json.loads(config_path.read_text(encoding="utf-8"))
        id2label = {int(v): k for k, v in label2id.items()}
        model = cls(
            model_name_or_path=model_cfg.get("model_name_or_path"),
            label2id=label2id,
            id2label=id2label,
            dropout=model_cfg.get("dropout", 0.1),
            freeze_encoder=model_cfg.get("freeze_encoder", False),
            unfreeze_last_n_layers=model_cfg.get("unfreeze_last_n_layers"),
            use_bilstm=model_cfg.get("use_bilstm", False),
            lstm_hidden_size=model_cfg.get("lstm_hidden_size"),
            lstm_num_layers=model_cfg.get("lstm_num_layers", 1),
            lstm_dropout=model_cfg.get("lstm_dropout", 0.0),
            use_noise=model_cfg.get("use_noise", False),
            noise_embed_dim=model_cfg.get("noise_embed_dim", 16),
            noise_mode=model_cfg.get("noise_mode", "bucket"),
            noise_mlp_hidden_dim=model_cfg.get("noise_mlp_hidden_dim"),
            noise_bin_edges=model_cfg.get("noise_bin_edges"),
            boundary_loss_weight=model_cfg.get("boundary_loss_weight", 0.0),
            boundary_positive_weight=model_cfg.get("boundary_positive_weight", 1.0),
            include_hospital_boundary=model_cfg.get("include_hospital_boundary", True),
            token_ce_loss_weight=model_cfg.get("token_ce_loss_weight", 0.0),
            token_ce_label_smoothing=model_cfg.get("token_ce_label_smoothing", 0.0),
            boundary_ce_label_smoothing=model_cfg.get("boundary_ce_label_smoothing", 0.0),
            token_ce_value_class_weight=model_cfg.get("token_ce_value_class_weight", 3.0),
            token_ce_key_class_weight=model_cfg.get("token_ce_key_class_weight", 1.0),
            end_boundary_loss_weight=model_cfg.get("end_boundary_loss_weight", 0.0),
            end_boundary_positive_weight=model_cfg.get("end_boundary_positive_weight", 1.0),
        )
        if map_location is None and not torch.cuda.is_available():
            map_location = "cpu"
        
        safetensors_path = path / "model.safetensors"
        pytorch_path = path / "pytorch_model.bin"
        
        if safetensors_path.is_file():
            try:
                from safetensors.torch import load_file
                state = load_file(str(safetensors_path))
                logger.info("Loaded model weights from safetensors format")
            except ImportError:
                logger.warning("safetensors not installed, trying pytorch format")
                if not pytorch_path.is_file():
                    raise FileNotFoundError(f"Missing weights at {pytorch_path}")
                state = torch.load(pytorch_path, map_location=map_location)
        elif pytorch_path.is_file():
            state = torch.load(pytorch_path, map_location=map_location)
            logger.info("Loaded model weights from pytorch format")
        else:
            raise FileNotFoundError(f"Missing weights at {safetensors_path} or {pytorch_path}")
        
        model.load_state_dict(state)
        return model

    def _compute_boundary_label_ids(self) -> torch.Tensor:
        want = {"KEY", "VALUE"}
        if self.include_hospital_boundary:
            want.add("HOSPITAL")
        ids = []
        for idx, name in self.id2label.items():
            if name.startswith("B-"):
                base = name[2:]
                if base in want:
                    ids.append(int(idx))
        if not ids:
            return torch.zeros(0, dtype=torch.long)
        return torch.tensor(sorted(ids), dtype=torch.long)

    def _boundary_targets_from_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if self.boundary_label_ids.numel() == 0:
            return torch.zeros_like(labels, dtype=torch.long)
        device = labels.device
        boundary_ids = self.boundary_label_ids.to(device)
        match = (labels.unsqueeze(-1) == boundary_ids.view(1, 1, -1)).any(dim=-1)
        return match.to(torch.long)

    def _class_weights_tensor(self, device, dtype):
        w = [1.0] * self.num_labels
        for idx, name in self.id2label.items():
            if name.endswith("VALUE") or name in ("B-VALUE", "I-VALUE"):
                w[int(idx)] = max(1.0, self.token_ce_value_class_weight)
            if name.endswith("KEY") or name in ("B-KEY", "I-KEY"):
                w[int(idx)] = max(1.0, self.token_ce_key_class_weight)
        return torch.tensor(w, device=device, dtype=dtype)

    def _end_boundary_targets_from_labels(self, labels: torch.Tensor) -> torch.Tensor:
        B, L = labels.size()
        out = torch.zeros_like(labels, dtype=torch.long)
        for b in range(B):
            for t in range(L):
                cur_id = int(labels[b, t].item())
                cur = self.id2label.get(cur_id, "O")
                if cur == "O" or "-" not in cur:
                    continue
                prefix, cur_type = cur.split("-", 1)
                if prefix == "E":
                    out[b, t] = 1
                    continue
                nxt = self.id2label.get(int(labels[b, t + 1].item()), "O") if (t + 1) < L else "O"
                if nxt == "O" or "-" not in nxt:
                    out[b, t] = 1
                    continue
                nxt_prefix, nxt_type = nxt.split("-", 1)
                if nxt_type != cur_type:
                    out[b, t] = 1
                    continue
                if prefix in {"B", "I"} and nxt_prefix not in {"I", "E"}:
                    out[b, t] = 1
        return out

    def _apply_bio_constraints(self) -> None:
        """Set invalid BIO transitions to a large negative prior."""
        big_neg = -10.0
        
        def parse(lbl: str):
            if lbl == "O":
                return ("O", "")
            if "-" in lbl:
                p, t = lbl.split("-", 1)
                return (p, t)
            return (lbl, "")

        for j in range(self.num_labels):
            pref, _ = parse(self.id2label[j])
            if pref == "I":
                self.crf.start_transitions.data[j] = big_neg

        for i in range(self.num_labels):
            pref_i, type_i = parse(self.id2label[i])
            for j in range(self.num_labels):
                pref_j, type_j = parse(self.id2label[j])
                allow = True
                if pref_j == "I":
                    if pref_i in {"B", "I"} and type_i == type_j:
                        allow = True
                    else:
                        allow = False
                elif pref_j == "E":
                    if pref_i in {"B", "I"} and type_i == type_j:
                        allow = True
                    elif pref_i in {"O", "E"}:
                        allow = True
                    else:
                        allow = False
                if not allow:
                    self.crf.transitions.data[i, j] = big_neg
