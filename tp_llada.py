# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D parallelisms (except pipeline parallelism) and various
# training techniques (e.g. activation checkpointing and compile) to the Llama model.

from collections import defaultdict

import torch
import torch.nn as nn
from torch.distributed._composable.replicate import replicate
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from generate import generate_batch

def parallelize_llada(
    model: nn.Module,
    world_mesh: DeviceMesh,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """

    apply_tp(
        model,
        world_mesh["tp"],
    )

    return model

def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
):
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    # check if wte and ff_out are the same thing
    parallelize_module(
        model,
        tp_mesh,
        {
            "model.model.transfomer.wte": RowwiseParallel(
                input_layouts=Replicate(),
            ),
            # "model.model.transformer.ff_out": ColwiseParallel(
            #     output_layouts=Replicate(),
            # ),
        },
    )

    rowwise_parallel, colwise_parallel, prepare_module_input = (
        RowwiseParallel,
        ColwiseParallel,
        PrepareModuleInput,
    )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    for transformer_block in model.model.transformer.blocks:
        layer_plan = {
            "q_proj": colwise_parallel(),
            "k_proj": colwise_parallel(),
            "v_proj": colwise_parallel(),
            "attn_out": rowwise_parallel(),
            # "ffn_norm": SequenceParallel(),
            "ff_proj": colwise_parallel(),
            "up_proj": colwise_parallel(),
            "ff_out": rowwise_parallel(),
        }
        # print(f"transformer_block: {transformer_block}")

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

def main():
    # use meta-llama/Meta-Llama-3.1-8B-Instruct from huggingface
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    from torch.distributed.device_mesh import init_device_mesh
    # set seed
    torch.manual_seed(42)
    import os
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # world_size = 2
    mesh = init_device_mesh("cuda", (world_size,))

    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16)
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', cache_dir="/work/10446/tchang85/ls6/tmp/", trust_remote_code=True, torch_dtype=torch.bfloat16)

    assert model.model.config.n_heads % world_size == 0, f"n_heads {model.model.config.n_heads} must be divisible by world_size {world_size}"
    assert model.model.config.n_kv_heads % world_size == 0, f"n_kv_heads {model.model.config.n_kv_heads} must be divisible by world_size {world_size}"
    model.model.config.n_heads = model.model.config.n_heads // world_size
    model.model.config.n_kv_heads = model.model.config.n_kv_heads // world_size
    print(f"effectie n_kv_heads: {model.model.config.effective_n_kv_heads}")
    
    model = model.to("cuda")
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', cache_dir="/work/10446/tchang85/ls6/tmp/", trust_remote_code=True)
    print(f"model: {model}")
    # print tp_plan
    # for name, module in model.named_modules():
    #     print(name)

    parallelize_llada(model, world_mesh={"tp": mesh})

    # warm up
    for i in range(1):
        prompt = "Hello brother, how are you?"
        input_ids = tokenizer(prompt, return_tensors="pt")['input_ids']
        # outputs = model.generate(**inputs, max_new_tokens=50)
        out = generate_batch(model, input_ids, steps=128, gen_length=128, block_length=128, temperature=0., cfg_scale=0., remasking='low_confidence')

    print(f"starting inference...")
    # prompt = "What is the capital of France?" * 2500 + "Fuck"
    prompt = "What is the capital of France?" * 200 + "How is your day? "
    input_ids = tokenizer(prompt, return_tensors="pt")['input_ids']

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        out = generate_batch(model, input_ids, steps=128, gen_length=128, block_length=128, temperature=0., cfg_scale=0., remasking='low_confidence')
    end.record()
    torch.cuda.synchronize()
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True))
    # print(f"outputs: {outputs}")
    print(f"Time taken: {start.elapsed_time(end)} ms")
    
    

if __name__ == "__main__":
    main()
    
    