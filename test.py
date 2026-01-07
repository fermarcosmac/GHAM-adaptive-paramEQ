import torch
print(torch.cuda.is_available())         # True => CUDA usable
print(torch.cuda.device_count())         # number of visible GPUs
print(torch.cuda.get_device_name(0))     # device name
print(torch.version.cuda)                # CUDA runtime used by this PyTorch build