import torch

def get_device(cfg):
    if cfg.device[0] != -1:
        device = f'cuda:{cfg.device[0]}'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device
