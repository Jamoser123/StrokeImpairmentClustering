# regressorModels.py
# Note: The regressor Models where left out pretty fast
from pytorch_lightning.loggers import TensorBoardLogger
import seaborn as sns  # Optional, for better plotting styles
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor,
    TQDMProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, R2Score

class StrokeLSTMRegressor(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        """
        Args:
        input_dim: Number of input features
        hidden_dim: Number of hidden units
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization factor
        """
        super().__init__()
        self.save_hyperparameters()
    

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
            
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Metrics
        metrics = lambda: nn.ModuleDict({
            'mse': MeanSquaredError(),
            'r2': R2Score()
        })
        
        self.train_metrics = metrics()
        self.val_metrics = metrics()
        self.test_metrics = metrics()
        
        self.criterion = nn.MSELoss()
    
    def forward(self, x):
        # Add input validation
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
            print("NaN values detected and replaced in forward pass")
        
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out.view(-1, 1)
    
    def _shared_step(self, batch, batch_idx, stage):
        imu_data, fma_score = batch
        y_hat = self(imu_data.float())
        # Clamp predictions between 1 and 66
        #y_hat = torch.clamp(y_hat, min=1.0, max=66.0)
        y_hat = y_hat * 65 + 1
        loss = self.criterion(y_hat, fma_score.float())
        
        metrics = getattr(self, f'{stage}_metrics')
        for metric in metrics.values():
            metric.update(y_hat, fma_score.float())
        
        self.log(f'{stage}_loss', loss, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)
        for name, metric in metrics.items():
            self.log(
                f'{stage}_{name}', 
                metric, 
                prog_bar=True, 
                on_epoch=True, 
                on_step=False,
                sync_dist=True
            )
            
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=10,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            },
        }

class StrokeCNNRegressor(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        kernel_size: int = 3
    ):
        """
        Args:
        input_dim: Number of input features
        hidden_dim: Number of filters/channels
        num_layers: Number of CNN layers
        dropout: Dropout rate
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization factor
        kernel_size: Size of the convolutional kernel
        """
        super().__init__()
        self.save_hyperparameters()

        # Create CNN layers
        cnn_layers = []
        in_channels = input_dim
        
        for _ in range(num_layers):
            cnn_layers.extend([
                nn.Conv1d(in_channels, hidden_dim, kernel_size, padding='same'),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            in_channels = hidden_dim
            
        self.cnn = nn.Sequential(*cnn_layers)
            
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Metrics
        metrics = lambda: nn.ModuleDict({
            'mse': MeanSquaredError(),
            'r2': R2Score()
        })
        
        self.train_metrics = metrics()
        self.val_metrics = metrics()
        self.test_metrics = metrics()
        
        self.criterion = nn.MSELoss()
    
    def forward(self, x):
        # Add input validation
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
            print("NaN values detected and replaced in forward pass")
        
        # Transpose input for CNN [batch, features, sequence]
        x = x.transpose(1, 2)
        
        # Apply CNN layers
        x = self.cnn(x)
        
        # Global average pooling
        x = torch.mean(x, dim=2)
        
        # Apply fully connected layers
        out = self.fc(x)
        return out.view(-1, 1)
    
    def _shared_step(self, batch, batch_idx, stage):
        imu_data, fma_score = batch
        y_hat = self(imu_data.float())
        # Clamp predictions between 1 and 66
        y_hat = y_hat * 65 + 1
        loss = self.criterion(y_hat, fma_score.float())
        
        metrics = getattr(self, f'{stage}_metrics')
        for metric in metrics.values():
            metric.update(y_hat, fma_score.float())
        
        self.log(f'{stage}_loss', loss, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)
        for name, metric in metrics.items():
            self.log(
                f'{stage}_{name}', 
                metric, 
                prog_bar=True, 
                on_epoch=True, 
                on_step=False,
                sync_dist=True
            )
            
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=10,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            },
        }