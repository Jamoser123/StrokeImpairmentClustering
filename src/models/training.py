# training.py
import pytorch_lightning as pl
import os
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from src.datasets.contrastiveDatasets import create_traindatasets, create_traindataloaders
from src.models.contrastiveModels import TimeSeriesSimCLR
from src.util.funcs import set_all_seeds, getMLFeatures, split_patients, create_version_name

def train_CLM_base(cfg):
    set_all_seeds(cfg.seed)

    feature_columns = getMLFeatures(cfg)

    input_dim = (len(feature_columns) // 2) if cfg.ml.train_one_wrist else len(feature_columns)
    
    train_patients, val_patients = split_patients(cfg)
    
    version_name = create_version_name(cfg)

    train_dataset, val_dataset = create_traindatasets(cfg, train_patients, val_patients, feature_columns)
    
    # Create dataloaders
    train_loader, val_loader = create_traindataloaders(
        train_dataset, 
        val_dataset,
        batch_size=cfg.ml.batch_size,
        num_workers=len(os.sched_getaffinity(0))
    )
    # Initialize model
    model = TimeSeriesSimCLR( # TODO: when time left just give cfg as a param
        input_dim=input_dim,#len(feature_columns),
        hidden_dim=cfg.ml.hidden_dim,
        projection_dim=cfg.ml.projection_dim,
        num_layers=cfg.ml.num_layers,
        temperature=cfg.ml.temperature,
        learning_rate=cfg.ml.learning_rate,
        dropout=cfg.ml.dropout,
        weight_decay=cfg.ml.weight_decay,
        batch_size=cfg.ml.batch_size,
    )

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            dirpath=f"/cluster/work/vogtlab/Group/jamoser/checkpoints/{version_name}",
            filename='best_checkpoint',
            save_top_k=1,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            mode='min'
        ),
        LearningRateMonitor(logging_interval='epoch'),
        TQDMProgressBar(refresh_rate=10)
    ]

    # Setup logger
    logger = TensorBoardLogger(
        save_dir="/cluster/work/vogtlab/Group/jamoser/lightning_logs",
        name=cfg.ml.logName,
        default_hp_metric=False,
        version=version_name
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='auto',
        devices=1,#'auto',
        # strategy='ddp',
        use_distributed_sampler=False,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
        log_every_n_steps=10,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    try:
    # Train the model
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        logger.log_hyperparams({
            'hidden_dim': cfg.ml.hidden_dim,
            'projection_dim': cfg.ml.projection_dim,
            'temperature': cfg.ml.temperature,
            'num_layers': cfg.ml.num_layers,
            'dropout': cfg.ml.dropout,
            'learning_rate': cfg.ml.learning_rate,
            'weight_decay': cfg.ml.weight_decay,
            'noise_level': cfg.ml.noise_level,
            'mask_ratio': cfg.ml.mask_ratio,
            'batch_size': cfg.ml.batch_size,
            'window_size': cfg.window_size,
            'threshold': cfg.threshold_ratio
        },{
            'hparam/loss': trainer.logged_metrics['val_loss_epoch'],
        })
        
        return trainer.logged_metrics['val_loss_epoch']
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return 15#default loss of 15