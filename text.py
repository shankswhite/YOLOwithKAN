import torch
import torch.distributed as dist

dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', rank=0, world_size=1)
print("Distributed backend initialized successfully.")