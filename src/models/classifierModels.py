# classifierModels.py
from tqdm import tqdm
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor,
    TQDMProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix
from util.utilFuncs import create_version_name

class StrokeLSTMClassifier(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int, 
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        augment_prob: float = 0.5,
        noise_level: float = 0.05,
        mask_ratio: float = 0.1,
        mc_samples: int = 10
    ):
        super().__init__()
        self.save_hyperparameters()

        self.augment_prob = augment_prob
        self.noise_level = noise_level
        self.mask_ratio = mask_ratio
    
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
            
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes, average='weighted')
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes, average='weighted')
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes, average='weighted')
        
        self.criterion = nn.CrossEntropyLoss()

        self.mc_samples = mc_samples
    
    def _apply_augmentation(self, x):
        if not self.training:
            return x
            
        batch_size, seq_len, features = x.shape
        augmented = x.clone()
        
        if torch.rand(1) < self.augment_prob:
            noise = torch.randn_like(augmented) * self.noise_level
            augmented = augmented + noise
            
        if torch.rand(1) < self.augment_prob:
            scale = torch.randn(batch_size, 1, features, device=x.device) * 0.1 + 1
            augmented = augmented * scale
            
        if torch.rand(1) < self.augment_prob:
            mask_length = int(seq_len * self.mask_ratio)
            for i in range(batch_size):
                start_idx = torch.randint(0, seq_len - mask_length, (1,))
                augmented[i, start_idx:start_idx+mask_length, :] = 0
            
        if torch.rand(1) < self.augment_prob:
            indices = torch.arange(seq_len, device=x.device)
            warped_indices = indices + torch.randn(seq_len, device=x.device) * 2
            warped_indices = torch.clamp(warped_indices, 0, seq_len-1)
            warped_indices = warped_indices.sort()[0].long()
            augmented = augmented[:, warped_indices, :]
            
        return augmented

    def forward(self, x):
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        x = self._apply_augmentation(x)
        
        if not self.training:
            # MC Dropout forward pass
            mc_outputs = []
            # Enable dropout
            self.lstm.train()
            self.fc.train()
            
            for _ in range(self.mc_samples):
                _, (hn, _) = self.lstm(x)
                # Ensure hn[-1] maintains batch dimension
                last_hidden = hn[-1].view(1, -1) if hn[-1].dim() == 1 else hn[-1]
                out = self.fc(last_hidden)
                # Ensure out has correct shape [1, num_classes] for single items
                if out.dim() == 1:
                    out = out.unsqueeze(0)
                mc_outputs.append(out.unsqueeze(0))
            
            # Disable dropout
            self.lstm.eval()
            self.fc.eval()
            
            # Stack all MC predictions [mc_samples, batch_size, num_classes]
            mc_preds = torch.cat(mc_outputs, dim=0)
            mc_probs = F.softmax(mc_preds, dim=-1)
            out = mc_probs.mean(dim=0)
            return out
        else:
            # Regular training forward pass
            _, (hn, _) = self.lstm(x)
            out = self.fc(hn[-1])
            return out
    
    def _shared_step(self, batch, batch_idx, stage):
        _, imu_data, dem, labels = batch
        y_hat = self(imu_data.float())

        labels = labels.view(-1)
        
        loss = self.criterion(y_hat, labels)
        
        if stage == 'train':
            accuracy = self.train_accuracy
        elif stage == 'val':
            accuracy = self.val_accuracy
        else:
            accuracy = self.test_accuracy

        accuracy.update(y_hat, labels)
        
        self.log(f'{stage}_loss', loss, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)
        self.log(f'{stage}_accuracy', accuracy, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            
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
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1
            },
        }