def get_cpu_or_gpu() -> str:
    import torch
    if torch.cuda.is_available():
        return 'cuda:0'
    else:
        return 'cpu'