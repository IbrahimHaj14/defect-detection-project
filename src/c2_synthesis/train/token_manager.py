"""Learned CLIP token lifecycle for one C2 dataset/class adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


class LearnedTokenManager:
    """Add, train, save, and reload exactly one CLIP text token."""

    def __init__(self, tokenizer: Any, text_encoder: nn.Module, token: str) -> None:
        if not token or not token.strip():
            raise ValueError("token must be a non-empty string")
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.token = token
        self._gradient_handle: torch.utils.hooks.RemovableHandle | None = None

        added = int(tokenizer.add_tokens([token]))
        if added != 1:
            raise ValueError(f"Learned token already exists in the tokenizer: {token}")
        text_encoder.resize_token_embeddings(len(tokenizer))
        self.token_id = int(tokenizer.convert_tokens_to_ids(token))
        if self.token_id == int(tokenizer.unk_token_id):
            raise RuntimeError(f"Tokenizer did not register learned token: {token}")

        embeddings = self.embedding_layer
        with torch.no_grad():
            initializer_id = int(tokenizer.convert_tokens_to_ids("defect"))
            if initializer_id == int(tokenizer.unk_token_id):
                initializer_id = int(tokenizer.convert_tokens_to_ids("damage"))
            embeddings.weight[self.token_id].copy_(embeddings.weight[initializer_id])

        text_encoder.requires_grad_(False)
        embeddings.weight.requires_grad_(True)
        self._install_gradient_filter()

    @property
    def embedding_layer(self) -> nn.Embedding:
        embeddings = self.text_encoder.get_input_embeddings()
        if not isinstance(embeddings, nn.Embedding):
            raise TypeError("CLIP text encoder input embeddings are not nn.Embedding")
        return embeddings

    @property
    def trainable_parameter(self) -> nn.Parameter:
        """Return the embedding table whose gradient is filtered to one row."""

        return self.embedding_layer.weight

    def _install_gradient_filter(self) -> None:
        if self._gradient_handle is not None:
            self._gradient_handle.remove()
        token_id = self.token_id

        def keep_learned_row_only(gradient: Tensor) -> Tensor:
            filtered = torch.zeros_like(gradient)
            filtered[token_id].copy_(gradient[token_id])
            return filtered

        self._gradient_handle = self.trainable_parameter.register_hook(
            keep_learned_row_only
        )

    def encode_prompt(
        self,
        prompt: str,
        *,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Encode a prompt and return hidden states plus learned-token positions."""

        encoded = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(device)
        matches = input_ids.eq(self.token_id)
        if not bool(matches.any(dim=1).all().item()):
            raise ValueError(f"Prompt must contain learned token exactly once: {prompt}")
        if not bool(matches.sum(dim=1).eq(1).all().item()):
            raise ValueError(f"Prompt contains learned token more than once: {prompt}")
        token_positions = matches.to(torch.int64).argmax(dim=1)
        hidden_states = self.text_encoder(input_ids)[0]
        return hidden_states, token_positions

    def save(self, path: str | Path) -> Path:
        """Save the token string and its fp32 embedding row to ``token.pt``."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "token": self.token,
            "token_id": self.token_id,
            "embedding": self.embedding_layer.weight[self.token_id]
            .detach()
            .float()
            .cpu(),
        }
        torch.save(payload, output_path)
        return output_path

    @classmethod
    def load(
        cls,
        tokenizer: Any,
        text_encoder: nn.Module,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "LearnedTokenManager":
        """Add a saved token to fresh CLIP components and restore its row."""

        payload = torch.load(path, map_location=map_location, weights_only=True)
        required = {"format_version", "token", "embedding"}
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise ValueError(f"Invalid learned-token checkpoint: {path}")
        if int(payload["format_version"]) != 1:
            raise ValueError(f"Unsupported token checkpoint version: {payload['format_version']}")

        manager = cls(tokenizer, text_encoder, str(payload["token"]))
        saved_embedding = torch.as_tensor(payload["embedding"])
        target = manager.embedding_layer.weight[manager.token_id]
        if saved_embedding.shape != target.shape:
            raise ValueError(
                f"Saved token embedding has shape {tuple(saved_embedding.shape)}, "
                f"expected {tuple(target.shape)}"
            )
        with torch.no_grad():
            target.copy_(saved_embedding.to(device=target.device, dtype=target.dtype))
        return manager

    def close(self) -> None:
        """Remove the embedding gradient hook."""

        if self._gradient_handle is not None:
            self._gradient_handle.remove()
            self._gradient_handle = None
