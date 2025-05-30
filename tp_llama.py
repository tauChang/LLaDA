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
def parallelize_llama_prefill(
    model: nn.Module,
    world_mesh: DeviceMesh,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """

    apply_tp_prefill(
        model,
        world_mesh["tp"],
    )

    return model

def parallelize_llama(
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

def apply_tp_prefill(
    model: nn.Module,
    tp_mesh: DeviceMesh,
):
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    parallelize_module(
        model,
        tp_mesh,
        {
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
            ),
            "model.norm": SequenceParallel(),
            # "model.layers.0": PrepareModuleInput(
            #     input_layouts=(Replicate(),),
            #     desired_input_layouts=(Shard(1),),
            #     # input_kwarg_layouts={
            #     #     "hidden_states": Replicate(),
            #     # },
            #     # desired_input_kwarg_layouts={
            #     #     "hidden_states": Shard(1),
            #     # },
            # ),
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate(),
            ),
        },
    )

    rowwise_parallel, colwise_parallel, prepare_module_input= (
        RowwiseParallel,
        ColwiseParallel,
        PrepareModuleInput,
    )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    for transformer_block in model.model.layers:
        layer_plan = {
            # "": PrepareModuleInput(
            #     desired_input_layouts=Shard(1),
            # ),
            # "input_layernorm": SequenceParallel(),
            # "self_attn": prepare_module_input(
            #     input_layouts=(Shard(1),),
            #     desired_input_layouts=(Replicate(), ),
            # ),
            # "self_attn": prepare_module_input(
            #     input_kwarg_layouts={
            #         "hidden_states": Shard(1),
            #     },
            #     desired_input_kwarg_layouts={
            #         "hidden_states": Replicate(),
            #     }
            # ),
            # "": PrepareModuleOutput(
            #     desired_output_layouts=Shard(1),
            # ),
            "self_attn.q_proj": colwise_parallel(),
            "self_attn.k_proj": colwise_parallel(),
            "self_attn.v_proj": colwise_parallel(),
            "self_attn.o_proj": rowwise_parallel(),
            # "self_attn.o_proj": rowwise_parallel(),
            # "post_attention_layernorm": SequenceParallel(),
            # "mlp": prepare_module_input(
            #     input_layouts=(Shard(1),),
            #     desired_input_layouts=(Replicate(),),
            # ),
            "mlp.gate_proj": colwise_parallel(),
            # "mlp.down_proj": rowwise_parallel(output_layouts=Shard(1)),
            "mlp.down_proj": rowwise_parallel(),
            "mlp.up_proj": colwise_parallel(),
        }
        # print(f"transformer_block: {transformer_block}")

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
):
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    parallelize_module(
        model,
        tp_mesh,
        {
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                # output_layouts=Shard(1),
            ),
            # "model.norm": SequenceParallel(),
            "lm_head": ColwiseParallel(
                # input_layouts=Shard(1),
                output_layouts=Replicate(),
            ),
            # "tok_embeddings": RowwiseParallel(
            #     input_layouts=Replicate(),
            #     output_layouts=Shard(1),
            # ),
            # "norm": SequenceParallel(),
            # "output": ColwiseParallel(
            #     input_layouts=Shard(1),
            #     output_layouts=Replicate(),
            #     use_local_output=True,
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
    for transformer_block in model.model.layers:
        layer_plan = {
            # "attention_norm": SequenceParallel(),
            # "self_attn": prepare_module_input(
            #     input_layouts=(Shard(1),),
            #     desired_input_layouts=(Replicate(), ),
            # ),
            "self_attn.q_proj": colwise_parallel(),
            "self_attn.k_proj": colwise_parallel(),
            "self_attn.v_proj": colwise_parallel(),
            # "self_attn.o_proj": rowwise_parallel(output_layouts=Shard(1)),
            "self_attn.o_proj": rowwise_parallel(),
            # "ffn_norm": SequenceParallel(),
            # "mlp": prepare_module_input(
            #     input_layouts=(Shard(1),),
            #     desired_input_layouts=(Replicate(),),
            # ),
            "mlp.gate_proj": colwise_parallel(),
            # "mlp.down_proj": rowwise_parallel(output_layouts=Shard(1)),
            "mlp.down_proj": rowwise_parallel(),
            "mlp.up_proj": colwise_parallel(),
            # "input_layernorm": SequenceParallel(),
            # "post_attention_layernorm": SequenceParallel(),
        }
        # print(f"transformer_block: {transformer_block}")

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

def main():
    # use meta-llama/Meta-Llama-3.1-8B-Instruct from huggingface
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.distributed.device_mesh import init_device_mesh
    # set seed
    torch.manual_seed(42)
    import os
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # world_size = 2
    mesh = init_device_mesh("cuda", (world_size,))

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16)
    model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    print(f"model: {model}")
    # for name, module in model.named_modules():
    #     print(name)

    # Initialize the device mesh
    parallelize_llama(model, world_mesh={"tp": mesh})
    # parallelize_llama_prefill(model, world_mesh={"tp": mesh})

    # warm up
    for i in range(3):
        prompt = "Hello brother, how are you?"
        inputs = tokenizer(prompt, return_tensors="pt").to(mesh.device_type)
        # outputs = model.generate(**inputs, max_new_tokens=50)
        with torch.no_grad():
            outputs = model.forward(**inputs)

    print(f"starting inference...")
    prompt = "What is the capital of France?" * 2500 + "Fuck"
    inputs = tokenizer(prompt, return_tensors="pt").to(mesh.device_type)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    
    # outputs = model.generate(**inputs, max_new_tokens=50)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    with torch.no_grad():
        outputs = model.forward(**inputs)
        logits = outputs.logits
        outputs = logits.argmax(dim=-1)
        # decode
        outputs = outputs.cpu()
        outputs = outputs.numpy()
        outputs = outputs.tolist()
    end.record()
    torch.cuda.synchronize()
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(f"outputs: {outputs}")
    print(f"Time taken: {start.elapsed_time(end)} ms")
    
    

if __name__ == "__main__":
    main()
    
    