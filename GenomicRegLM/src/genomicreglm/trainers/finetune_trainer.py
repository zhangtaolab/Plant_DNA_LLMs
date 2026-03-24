from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


@dataclass
class FinetuneConfig:
    epochs: int = 5
    lr: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class FinetuneTrainer:
    """Lightweight trainer for early architecture validation."""

    def __init__(self, model: torch.nn.Module, config: FinetuneConfig):
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)

    def fit(self, train_loader: DataLoader) -> None:
        self.model.train()
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.config.device)
                signal_features = batch["signal_features"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)

                out = self.model(input_ids=input_ids, signal_features=signal_features, labels=labels)
                loss = out["loss"]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f"[epoch {epoch + 1}] loss={epoch_loss / max(1, len(train_loader)):.6f}")
