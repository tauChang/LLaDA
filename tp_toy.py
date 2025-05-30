import os
import sys
import time
import torch
import torch.nn as nn

from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

from log_utils import rank_log, get_logger, verify_min_gpu_count
from torch.distributed._tensor.device_mesh import init_device_mesh

# ---- GPU check ------------
_min_gpu_count = 1
if not verify_min_gpu_count(min_gpus=_min_gpu_count):
    print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
    sys.exit()
# ---------------------------

"""
This is the script to test Tensor Parallel(TP) on a toy model in a
Megatron-LM SPMD style. We show an E2E working flow from forward,
backward and optimization.
"""

class ToyModel(nn.Module):
    """MLP based model"""
    def __init__(self):
        super(ToyModel, self).__init__()
        self.in_proj = nn.Linear(10000, 330000)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(330000, 5)

    def forward(self, x):
        return self.out_proj(self.relu(self.in_proj(x)))


# ---------- Main execution begins here -------------
logger = get_logger()

# create a device mesh based on the given world_size.
_world_size = int(os.environ["WORLD_SIZE"])
device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(_world_size,))
_rank = device_mesh.get_rank()

print(f"Starting PyTorch TP example on rank {_rank}.")
rank_log(_rank, logger, f"Device Mesh created: {device_mesh=}")

# create model and move it to GPU
tp_model = ToyModel().to("cuda")

# Custom parallelization plan for the model
parallelize_module(
    module=tp_model,
    device_mesh=device_mesh,
    parallelize_plan={
        "in_proj": ColwiseParallel(),
        "out_proj": RowwiseParallel(),
    },
)
print(f"after parallelization, tp_model={tp_model}")

# Create an optimizer for the parallelized module.
lr = 0.25
optimizer = torch.optim.AdamW(tp_model.parameters(), lr=lr, foreach=True)

# Perform a number of iterations of forward/backward/optimization
num_iters = 50
rank_log(_rank, logger, "Tensor Parallel training starting...")

total_time = []

for i in range(num_iters):
    torch.cuda.synchronize()  # Ensure all previous operations are complete
    start_time = time.time()

    # For TP, input needs to be same across all TP ranks.
    torch.manual_seed(i)
    inp = torch.rand(20, 10000, device="cuda")
    output = tp_model(inp)
    output.sum().backward()
    optimizer.step()
    torch.cuda.synchronize()  # Ensure all operations are complete before measuring time
    iter_time = time.time() - start_time
    if i > 10:
        total_time.append(iter_time)
    rank_log(_rank, logger, f"Iter {i} completed in {iter_time:.4f} seconds")

avg_time = sum(total_time) / len(total_time) if total_time else 0.0
rank_log(_rank, logger, f"Tensor Parallel training completed! Avg iter time: {avg_time:.4f} seconds")
