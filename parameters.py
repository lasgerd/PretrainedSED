"""
parameters.py

This module stores all the configurable parameters and hyperparameters used
across the project, ensuring easy tuning.

"""

params = {
    # folder path related
    'current_main_folder': '.', # current main folder
    'suffix': 'local', # suffix of output folder

    # training params
    'batch_size': 4,
    'patience_val_loss': 40,
    'patience_train_loss': 40,
    'warmup_epochs': 1,
    'hold_epochs': 1,
    'decay_epochs': 1,
    'learning_rate': 1e-4,
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    
    # hyper-params of model
    'in_channels': 6,
    'num_classes': 5,
    'num_frames': 10,
    'num_conformer_layers': 8,

    # params of input feature
    'sr': 24000,
    'n_fft': 1024,
    'win_len': 960,
    'hop_len': 480,
    'n_mels': 64,

    # cpu/gpu
    'num_cpu_workers': 8,
    'prefetch_factor': 3,

    # metrics params
    'average': 'macro',  # Supports 'micro': sample-wise average and 'macro': class-wise average.
    'segment_based_metrics': False,  # If True, uses segment-based metrics, else uses event-based metrics.
    'lad_doa_thresh': 20,  # DOA error threshold for computing the detection metrics.
}
