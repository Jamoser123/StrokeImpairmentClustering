# contrastiveModels.py
# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Lightning for the training module
import pytorch_lightning as pl

# Optimizers and schedulers
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

class TimeSeriesSimCLR(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        projection_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        temperature: float = 0.07,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Encoder (LSTM)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )
        
        self.temperature = temperature
        self.batch_size = batch_size

    def project(self, x):
        z = self.projection(x)
        return F.normalize(z, dim=1)

    def forward(self, x):
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)

        _, (hidden, _) = self.encoder(x)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        return hidden
        
    def info_nce_loss(self, z_i, z_j, override_temperature=None):
        """
        Calculate NT-Xent loss for SimCLR with proper masking
        """
        batch_size = z_i.shape[0]
        
        # Concatenate representations
        representations = torch.cat([z_i, z_j], dim=0)  # 2*B x D
        
        # Calculate similarity matrix
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )  # 2*B x 2*B
        
        # Create masks for positive pairs
        sim_ij = torch.diag(similarity_matrix, batch_size)    # B
        sim_ji = torch.diag(similarity_matrix, -batch_size)   # B
        positives = torch.cat([sim_ij, sim_ji])              # 2*B
        
        # Remove diagonal for negative samples
        mask_no_diag = ~torch.eye(2*batch_size, dtype=torch.bool, device=similarity_matrix.device)
        negatives = similarity_matrix[mask_no_diag].view(2*batch_size, -1)  # 2*B x (2*B-1)

        temp = override_temperature if override_temperature is not None else self.temperature
        
        # Create logits and labels
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1) / temp  # 2*B x (2*B)
        labels = torch.zeros(2*batch_size, dtype=torch.long, device=logits.device)         # 2*B
        
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def training_step(self, batch, batch_idx):

        view1, view2, _ = batch[0], batch[1], batch[2]
        
        # Get projections of both views
        h_i = self(view1)
        h_j = self(view2)

        z_i = self.project(h_i)
        z_j = self.project(h_j)
        
        # Calculate loss
        loss = self.info_nce_loss(z_i, z_j)
        
        # Logging
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        view1, view2, _ = batch[0], batch[1], batch[2]        
        
        h_i = self(view1)
        h_j = self(view2)

        z_i = self.project(h_i)
        z_j = self.project(h_j)

        #Temperature defines lower bounds on loss, and if we don't set it to a constant we can't compare it 
        loss = self.info_nce_loss(z_i, z_j, override_temperature=1.0)
        
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        return loss
    
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

###### Model with adversiary that wants to predict affected side
class TimeSeriesSimCLRWithAdversary(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        projection_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        temperature: float = 0.07,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 32,
        adversary_weight: float = 0.1,  # Weight for adversarial loss
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        # Encoder (LSTM)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )
        
        # Adversary network
        self.adversary = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.temperature = temperature
        self.batch_size = batch_size
        self.adversary_weight = adversary_weight

    def project(self, x):
        z = self.projection(x)
        return F.normalize(z, dim=1)

    def forward(self, x):
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)

        _, (hidden, _) = self.encoder(x)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        return hidden
    
    def info_nce_loss(self, z_i, z_j, override_temperature=None):
        batch_size = z_i.shape[0]
        
        representations = torch.cat([z_i, z_j], dim=0)
        
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )
        
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji])
        
        mask_no_diag = ~torch.eye(2*batch_size, dtype=torch.bool, device=similarity_matrix.device)
        negatives = similarity_matrix[mask_no_diag].view(2*batch_size, -1)

        temp = override_temperature if override_temperature is not None else self.temperature
        
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1) / temp
        labels = torch.zeros(2*batch_size, dtype=torch.long, device=logits.device)
        
        return F.cross_entropy(logits, labels)
    
    def adversarial_loss(self, adv_i, adv_j, labels):
        # Forward pass through adversary
        adv_loss_i = F.binary_cross_entropy(adv_i.squeeze(), labels.float())
        adv_loss_j = F.binary_cross_entropy(adv_j.squeeze(), labels.float())
        adv_loss = (adv_loss_i + adv_loss_j) / 2
        return adv_loss
    
    def training_step(self, batch, batch_idx):
        opt_main, opt_adversary = self.optimizers()
        view1, view2, labels = batch[0], batch[1], batch[2]

        #Get embeddings
        h_i = self(view1)
        h_j = self(view2)

        # Optimize adversary on embeddings
        adv_i = self.adversary(h_i.detach())
        adv_j = self.adversary(h_j.detach())
        
        adv_loss = self.adversarial_loss(adv_i, adv_j, labels)
        
        opt_adversary.zero_grad()
        self.manual_backward(adv_loss)
        opt_adversary.step()

        # Optimize Backbone and projection
        z_i = self.project(h_i)
        z_j = self.project(h_j)
        
        contrastive_loss = self.info_nce_loss(z_i, z_j)
        
        total_loss = contrastive_loss - self.adversary_weight * adv_loss.detach()

        # Then optimize the encoder
        opt_main.zero_grad()
        self.manual_backward(total_loss)
        opt_main.step()
        
        # Logging
        self.log('train_contrastive_loss', contrastive_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        self.log('train_adversarial_loss', adv_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        
        return total_loss
            
    def validation_step(self, batch, batch_idx):
        view1, view2, labels = batch[0], batch[1], batch[2]
        
        with torch.no_grad():
            h_i = self(view1)
            h_j = self(view2)

            adv_i = self.adversary(h_i)
            adv_j = self.adversary(h_j)
            
            adv_loss = self.adversarial_loss(adv_i, adv_j, labels)

            z_i = self.project(h_i)
            z_j = self.project(h_j)
            
            contrastive_loss = self.info_nce_loss(z_i, z_j, override_temperature=1.0)
            
            total_loss = contrastive_loss - self.adversary_weight * adv_loss
            
            self.log('val_contrastive_loss', contrastive_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
            self.log('val_adversarial_loss', adv_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
            self.log('val_total_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        
        return total_loss
    
    def configure_optimizers(self):
        # Optimizer for encoder and projection head
        optimizer_main = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.projection.parameters()),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Separate optimizer for adversary
        optimizer_adversary = torch.optim.AdamW(
            self.adversary.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_main,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        scheduler_adversary = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_adversary,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        return [
            {
                "optimizer": optimizer_main,
                "lr_scheduler": {
                    "scheduler": scheduler_main,
                    "interval": "epoch"
                }
            },
            {
                "optimizer": optimizer_adversary,
                "lr_scheduler": {
                    "scheduler": scheduler_adversary,
                    "interval": "epoch"
                }
            }
        ]