from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

@dataclass
class Config:
    """Configuration class for a bilinear MLP model applied to simulated data."""
    d_input: int = 1_000
    d_hidden: int = 64
    d_output: int = 10
    bias: bool = False

    n_epochs: int = 100
    batch_size: int = 32
    lr: float = 0.001
    dropout_rate: float = 0.0  # Dropout rate for regularization
    weight_decay: float = 1e-4  # L2 regularization strength
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


class BaseBMLP(nn.Module, ABC):
    """Base class for bilinear MLP models with shared architecture and training logic."""
    
    def __init__(self, cfg: Config):
        super(BaseBMLP, self).__init__()
        self.cfg = cfg

        self.left = nn.Linear(self.cfg.d_input, self.cfg.d_hidden, bias=self.cfg.bias)
        self.right = nn.Linear(self.cfg.d_input, self.cfg.d_hidden, bias=self.cfg.bias)
        self.head = nn.Linear(self.cfg.d_hidden, self.cfg.d_output, bias=self.cfg.bias)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(p=self.cfg.dropout_rate)

        self.to(self.cfg.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply dropout to the hidden representation
        hidden = self.dropout(self.left(x) * self.right(x))
        return self.head(hidden)

    @abstractmethod
    def compute_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """Compute task-specific loss."""
        pass

    @abstractmethod
    def compute_metrics(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor
    ) -> float:
        """Compute task-specific metrics."""
        pass

    def step(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Forward pass with loss and metrics computation."""
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        metrics = self.compute_metrics(y_hat, y)
        return loss, metrics

    def fit(self, train, val) -> Tuple[List[float], List[float]]:
        """Shared training loop"""
        torch.set_grad_enabled(True)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.n_epochs)

        train_loader = DataLoader(train, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val, batch_size=self.cfg.batch_size, shuffle=False, drop_last=False)

        train_losses = []
        train_metrics = []
        val_losses = []
        val_metrics = []

        pbar = tqdm(range(self.cfg.n_epochs), desc=f"Training for {self.cfg.n_epochs} epochs")

        for _ in pbar:
            epoch_train_loss = 0.0
            epoch_train_metric = 0.0
            num_train_batches = 0
            
            for data in train_loader:
                x, y = data
                loss, metrics = self.train().step(x, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                epoch_train_metric += metrics
                num_train_batches += 1
            
            scheduler.step()

            avg_train_loss = epoch_train_loss / num_train_batches
            avg_train_metric = epoch_train_metric / num_train_batches
            train_losses.append(avg_train_loss)
            train_metrics.append(avg_train_metric)

            with torch.no_grad():
                epoch_val_loss = 0.0
                epoch_val_metric = 0.0
                num_val_batches = 0
                
                for data in val_loader:
                    x, y = data
                    loss, metrics = self.eval().step(x, y)
                    epoch_val_loss += loss.item()
                    epoch_val_metric += metrics
                    num_val_batches += 1
                
                avg_val_loss = epoch_val_loss / num_val_batches
                avg_val_metric = epoch_val_metric / num_val_batches
                val_losses.append(avg_val_loss)
                val_metrics.append(avg_val_metric)

            self._update_progress_bar(pbar, avg_train_loss, avg_train_metric, 
                                    avg_val_loss, avg_val_metric)

        torch.set_grad_enabled(False)
        return train_losses, val_losses

    @abstractmethod
    def _update_progress_bar(
        self,
        pbar,
        train_loss: float,
        train_metric: float,
        val_loss: float,
        val_metric: float
    ) -> None:
        """Update progress bar with task-specific metrics."""
        pass

    @property
    def w_l(self) -> torch.Tensor:
        return self.left.weight.detach()

    @property
    def w_r(self) -> torch.Tensor:
        return self.right.weight.detach()

    @property
    def w_p(self) -> torch.Tensor:
        return self.head.weight.detach()


class ScBMLPClassifier(BaseBMLP):
    """Bilinear MLP for single-cell classification tasks."""

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        return self.criterion(y_hat, y)

    def compute_metrics(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor
    ) -> float:
        accuracy = (y_hat.argmax(dim=1) == y).float().mean().item()
        return accuracy

    def _update_progress_bar(
        self,
        pbar,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float
    ) -> None:
        pbar.set_postfix(
            train_loss=f"{train_loss:.4f}", 
            train_acc=f"{train_acc:.4f}",
            val_loss=f"{val_loss:.4f}", 
            val_acc=f"{val_acc:.4f}"
        )


class ScBMLPRegressor(BaseBMLP):
    """Bilinear MLP for single-cell regression tasks (e.g., frequency prediction)."""

    def __init__(self, cfg: Config, loss_fn: str = "mse"):
        super().__init__(cfg)
        if loss_fn == "mse":
            self.criterion = nn.MSELoss()
        elif loss_fn == "l1":
            self.criterion = nn.L1Loss()
        elif loss_fn == "huber":
            self.criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

    def compute_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        return self.criterion(y_hat, y)

    def compute_metrics(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor
    ) -> float:
        with torch.no_grad():
            mae = torch.mean(torch.abs(y_hat - y)).item()
            return mae

    def _update_progress_bar(
        self,
        pbar,
        train_loss: float,
        train_mae: float,
        val_loss: float,
        val_mae: float
    ) -> None:
        pbar.set_postfix(
            train_loss=f"{train_loss:.4f}", 
            train_mae=f"{train_mae:.4f}",
            val_loss=f"{val_loss:.4f}", 
            val_mae=f"{val_mae:.4f}"
        )
