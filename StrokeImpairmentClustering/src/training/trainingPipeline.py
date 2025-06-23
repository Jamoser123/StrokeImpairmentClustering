# trainingPipeline.py
# 

def train_classifier(cfg, train_loader, val_loader, input_dim, num_classes):
    # Initialize model
    model = StrokeLSTMClassifier(
        input_dim=input_dim,
        hidden_dim=cfg.ml.hidden_dim,
        num_classes=num_classes, 
        num_layers=cfg.ml.num_layers,
        dropout=cfg.ml.dropout,
        learning_rate=cfg.ml.learning_rate,
        weight_decay=cfg.ml.weight_decay,
        augment_prob=cfg.ml.augment_prob,
        noise_level=cfg.ml.noise_level,
        mask_ratio=cfg.ml.mask_ratio,
        mc_samples=cfg.ml.mc_samples
    )
    
    version_name = create_version_name(cfg, "hyperparam_search")
    
    # Initialize logger
    logger = TensorBoardLogger(
        save_dir=f"lightning_logs_classifier_randomSampler", # add base path
        name="experiment_runs",
        #sub_dir=f"{cfg.ml.batch_size}",
        version=version_name,
        default_hp_metric=False
    )   

    # Additional callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=None,
        dirpath=f'checkpoints/{version_name}', # add base path
        filename='best-checkpoint',
        save_top_k=1,
        mode='min',
        verbose=True
    )
    
    # Additional callbacks
    callbacks = [
        #checkpoint_callback,
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            mode='min',
            min_delta=1e-4,
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
        TQDMProgressBar(refresh_rate=10)
    ]
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=100,
        logger=logger,
        callbacks=callbacks,
        accelerator='auto',
        devices='auto',
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
        log_every_n_steps=10,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    try:
        trainer.fit( #still add in a try error block to print relevant data when error
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        
        logger.log_hyperparams({
            'hidden_dim': cfg.ml.hidden_dim,
            'num_layers': cfg.ml.num_layers,
            'dropout': cfg.ml.dropout,
            'learning_rate': cfg.ml.learning_rate,
            'weight_decay': cfg.ml.weight_decay,
            'augment_prob': cfg.ml.augment_prob,
            'noise_level': cfg.ml.noise_level,
            'mask_ratio': cfg.ml.mask_ratio,
            'batch_size': cfg.ml.batch_size,
            'mc_samples': cfg.ml.mc_samples,
            'window_size': cfg.window_size,
            'threshold': cfg.threshold_ratio
        },{
            'hparam/loss': trainer.logged_metrics['val_loss_epoch'],
            'hparam/accuracy': trainer.logged_metrics['val_accuracy']
        })
        acc = Accuracy(task='multiclass', num_classes=3, average=None).to('cuda') # maybe add a function to the model that does this?
        conf = ConfusionMatrix(task='multiclass', num_classes=3).to('cuda')
        if trainer.is_global_zero: # we could distribute this but no priority currently
            model.eval()
            model = model.to('cuda')
            with torch.no_grad():
                for batch in tqdm(val_loader, desc='Validating', leave=True):
                    key, x, _, y = batch
                    x = x.to('cuda')
                    y = y.squeeze(-1).to('cuda')
                    y_hat = model(x)
                    
                    # Update metrics
                    acc.update(y_hat, y)
                    conf.update(y_hat, y)
                    
                    # Clear CUDA cache
                    del x, y, y_hat
                    torch.cuda.empty_cache()
        
        # Move model back to CPU if needed
        model = model.cpu()
        
        # Get figures while metrics are still on GPU
        fig_acc, _ = acc.plot()
        fig_conf, _ = conf.plot()
        
        # Log figures
        logger.experiment.add_figure('Accuracy_Plot', fig_acc, global_step=trainer.global_step)
        logger.experiment.add_figure('Conf_Matrix', fig_conf, global_step=trainer.global_step)
        
        # Move metrics to CPU and clear them
        acc = acc.cpu()
        conf = conf.cpu()
        del acc, conf, model
        torch.cuda.empty_cache()
        
        return None, trainer, trainer.logged_metrics['val_loss_epoch'] #returned model previously but removed for storage optimisation
    except Exception as e:
        # Print the error message
        print(f"An error occurred: {e}")
        return None, None, float('inf')

def train_regressor(cfg, train_loader, val_loader, input_dim):
    # Set style for better-looking plots
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    print(f"Train len {len(train_loader)}, Val Len {len(val_loader)}")
    
    # Initialize model
    if cfg.ml.use_lstm:
        model = StrokeLSTMRegressor(
            input_dim=input_dim,
            hidden_dim=cfg.ml.hidden_dim,
            num_layers=cfg.ml.num_layers,
            dropout=cfg.ml.dropout,
            learning_rate=cfg.ml.learning_rate,
            weight_decay=cfg.ml.weight_decay
        )
    else:
        model = StrokeCNNRegressor(
            input_dim=input_dim,
            hidden_dim=cfg.ml.hidden_dim,
            num_layers=cfg.ml.num_layers,
            dropout=cfg.ml.dropout,
            learning_rate=cfg.ml.learning_rate,
            weight_decay=cfg.ml.weight_decay,
            kernel_size=cfg.ml.kernel_size
        )
    
    # Initialize logger with version name based on hyperparameters
    folder_name = f"stroke_{'lstm' if cfg.ml.use_lstm else 'cnn'}_regressor_w{cfg.window_size}_s{cfg.step_size}_th{cfg.threshold_ratio}"
    version_name = f"h{cfg.ml.hidden_dim}_l{cfg.ml.num_layers}_d{cfg.ml.dropout}_lr{cfg.ml.learning_rate}_wd{cfg.ml.weight_decay}"
    logger = TensorBoardLogger(
        "lightning_logs_regressor", 
        name=folder_name,
        version=version_name,
        default_hp_metric=False
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints',
            filename=f'{folder_name}-{version_name}-' + '{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min',
            save_weights_only=True,
            every_n_epochs=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            mode='min',
            min_delta=1e-4,
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
        TQDMProgressBar(refresh_rate=10)
    ]
    
    # Initialize trainer with additional safeguards
    trainer = pl.Trainer(
        max_epochs=100,
        logger=logger,
        callbacks=callbacks,
        accelerator='auto',
        devices='auto',
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
        log_every_n_steps=10,
        deterministic=True
    )
    
    # Train model with error handling
    try:
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        return None, None
    
    # Log best model score
    best_model_score = trainer.checkpoint_callback.best_model_score
    print(f"Best validation loss: {best_model_score:.4f}")
    
    # Save hyperparameters
    logger.log_hyperparams({
        'hidden_dim': cfg.ml.hidden_dim,
        'num_layers': cfg.ml.num_layers,
        'dropout': cfg.ml.dropout,
        'learning_rate': cfg.ml.learning_rate,
        'train_days': cfg.ml.train_days,
        'val_days': cfg.ml.val_days,
        'best_val_loss': best_model_score
    })

    # After training completes successfully, add this code:
    if model is not None:
        model.eval()
        all_preds = []
        all_targets = []
        residuals = []
        
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                y_hat = model(x)
                y_hat = y_hat * 65 + 1
                
                # Ensure we're getting flat arrays
                all_preds.extend(y_hat.cpu().numpy().flatten())
                all_targets.extend(y.cpu().numpy().flatten())
                residuals.extend((y_hat - y).cpu().numpy().flatten())
        
        # Convert to 1D numpy arrays
        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets).flatten()
        residuals = np.array(residuals).flatten()
        
        # Remove any NaN values if present
        mask = ~(np.isnan(all_preds) | np.isnan(all_targets))
        all_preds = all_preds[mask]
        all_targets = all_targets[mask]
        residuals = residuals[mask]
        
        # Create scatter plot with confidence bands
        fig_scatter = plt.figure(figsize=(10, 10))
        plt.scatter(all_targets, all_preds, alpha=0.5, label='Predictions')
        
        # Fit line only if we have valid data
        if len(all_targets) > 1:
            z = np.polyfit(all_targets, all_preds, 1)
            p = np.poly1d(z)
            
            # Sort for proper line plotting
            x_sort = np.sort(all_targets)
            plt.plot(x_sort, p(x_sort), "b-", alpha=0.5, label='Trend Line')
            
        # Perfect prediction line
        min_val = min(all_targets.min(), all_preds.min())
        max_val = max(all_targets.max(), all_preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values')
        plt.legend()
        
        # Residual Plot
        fig_residual = plt.figure(figsize=(10, 5))
        plt.scatter(all_targets, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Actual Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # Error Distribution
        fig_hist = plt.figure(figsize=(10, 5))
        plt.hist(residuals, bins=50, density=True, alpha=0.7)
        plt.xlabel('Prediction Error')
        plt.ylabel('Density')
        plt.title('Error Distribution')
        
        # Log figures and metrics
        logger.experiment.add_figure('Predictions_vs_Actual', 
                                   fig_scatter, 
                                   global_step=trainer.global_step)
        logger.experiment.add_figure('Residual_Plot', 
                                   fig_residual, 
                                   global_step=trainer.global_step)
        logger.experiment.add_figure('Error_Distribution', 
                                   fig_hist, 
                                   global_step=trainer.global_step)
        
        # Close all figures to free memory
        plt.close('all')
    
    return model, trainer


#CLM
def train_model(cfg):
    # Set random seeds for reproducibility
    set_all_seeds(cfg.seed)
    
    #print(len(os.sched_getaffinity(0)))
    # Create a mask of all available CPUs
    all_cpus = set(range(os.cpu_count()))

# Set affinity to use all CPUs
    os.sched_setaffinity(0, all_cpus)

# Verify the new affinity
    #print(len(os.sched_getaffinity(0)))
    feature_columns = [
    'wrist_aff__acc_x', 'wrist_aff__acc_y', 'wrist_aff__acc_z',
    'wrist_aff__gyro_x', 'wrist_aff__gyro_y', 'wrist_aff__gyro_z',
    'wrist_nonaff__acc_x', 'wrist_nonaff__acc_y', 'wrist_nonaff__acc_z',
    'wrist_nonaff__gyro_x', 'wrist_nonaff__gyro_y', 'wrist_nonaff__gyro_z'
    ] if cfg.ml.useGyro else [
    'wrist_aff__acc_x', 'wrist_aff__acc_y', 'wrist_aff__acc_z',
    'wrist_nonaff__acc_x', 'wrist_nonaff__acc_y', 'wrist_nonaff__acc_z',
    ]
    
    # Create datasets and split patients
    train_patients, val_patients = split_patients(cfg)
    
    # Create train and validation datasets
    train_dataset, val_dataset = create_traindatasets(cfg, train_patients, val_patients, feature_columns)
    
    # Create dataloaders
    train_loader, val_loader = create_traindataloaders(
        train_dataset, 
        val_dataset,
        batch_size=cfg.ml.batch_size,
        num_workers=len(os.sched_getaffinity(0))#cfg.ml.num_workers
    )

    # Initialize model
    model = TimeSeriesSimCLR(
        input_dim=len(feature_columns),
        hidden_dim=cfg.ml.hidden_dim,
        projection_dim=cfg.ml.projection_dim,
        num_layers=cfg.ml.num_layers,
        temperature=cfg.ml.temperature,
        learning_rate=cfg.ml.learning_rate,
        dropout=cfg.ml.dropout,
        weight_decay=cfg.ml.weight_decay,
        batch_size=cfg.ml.batch_size,
    )

    version_name=create_version_name(cfg)
    
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
        save_dir="lightning_logs_SimCLR_swapSides",
        name="SimCLR_experiments",
        default_hp_metric=False,
        version=version_name
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='auto',
        devices='auto',
        strategy='ddp',
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
    except:
        return 15#default loss of 15