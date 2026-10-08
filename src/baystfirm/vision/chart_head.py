from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .chart import CHART_LABELS


class Pointer(nn.Module):
    """Small query/key pointer used only by Baystfirm's chart-state head."""

    def __init__(self, hidden: int, proj: int) -> None:
        super().__init__()
        self.q = nn.Linear(hidden, proj)
        self.k = nn.Linear(hidden, proj)
        self.scale = proj**-0.5

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = torch.einsum("bd,bnd->bn", self.q(q), self.k(k)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        return scores


class ChartStateHead(nn.Module):
    """JEV-style pointer over fixed market-state labels on frozen visual tokens."""

    def __init__(
        self,
        hidden_size: int,
        proj: int = 256,
        n_labels: int = len(CHART_LABELS),
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_labels = n_labels
        self.norm = nn.LayerNorm(hidden_size)
        self.pool_query = nn.Parameter(torch.empty(hidden_size))
        self.label_embeddings = nn.Parameter(torch.empty(n_labels, hidden_size))
        nn.init.normal_(self.pool_query, mean=0.0, std=0.02)
        nn.init.normal_(self.label_embeddings, mean=0.0, std=0.02)
        self.pointer = Pointer(hidden_size, proj)
        self.register_buffer("temperature", torch.tensor(1.0))
        self.register_buffer("logit_bias", torch.zeros(n_labels))

    def forward(self, visual_embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(visual_embeds)
        scores = torch.einsum("bnd,d->bn", normalized, self.pool_query) / math.sqrt(
            self.hidden_size
        )
        scores = scores.masked_fill(~mask.bool(), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bn,bnd->bd", weights, normalized)
        labels = self.label_embeddings.unsqueeze(0).expand(
            visual_embeds.shape[0], -1, -1
        )
        return (self.pointer(pooled, labels) + self.logit_bias) / self.temperature

    def fit_temperature(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Fit a positive scalar temperature against validation negative log likelihood."""
        if logits.numel() == 0 or targets.numel() == 0:
            raise ValueError("temperature calibration requires validation samples")
        log_temperature = nn.Parameter(torch.zeros((), device=logits.device))
        optimizer = torch.optim.LBFGS(
            [log_temperature],
            lr=0.01,
            max_iter=100,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / log_temperature.exp(), targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        fitted = log_temperature.detach().exp().clamp(min=0.05, max=20.0)
        self.temperature.copy_(fitted.to(device=self.temperature.device))
        return float(self.temperature.item())

    @staticmethod
    def decide(
        probs: torch.Tensor, min_confidence: float = 0.5
    ) -> tuple[str, float, bool]:
        values = probs.detach().reshape(-1)
        probability, index = values.max(dim=0)
        confidence = float(probability.item())
        label_index = int(index.item())
        if label_index >= len(CHART_LABELS):
            raise ValueError(
                f"probability vector has unsupported label index {label_index}"
            )
        return CHART_LABELS[label_index], confidence, confidence < min_confidence
