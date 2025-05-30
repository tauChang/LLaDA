import torch
import numpy as np
import torch.nn.functional as F
import os

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

import math

def get_num_transfer_tokens_cosine(mask_index, steps):
    """
    Cosine discretization schedule for denoising:
    Slower unmasking at early steps, faster at later steps.
    """
    B = mask_index.size(0)  # batch size
    mask_num = mask_index.sum(dim=1, keepdim=True)  # [B, 1]

    # Compute t(i) = cos(pi/2 * (1 - i/T)) for i = 0 to steps
    i = torch.arange(0, steps + 1, device=mask_index.device, dtype=torch.float32)
    t = torch.cos(0.5 * math.pi * (1 - i / steps))  # [steps + 1]

    # Compute alpha schedule (monotonic decreasing)
    alpha = t / t[0]  # Normalize to start at 1

    # Cumulative mass to reveal = 1 - alpha
    cumulative_mass = 1 - alpha  # [steps + 1]

    # How much to unmask at each step
    per_step_fraction = cumulative_mass[1:] - cumulative_mass[:-1]  # [steps]
    step_ratios = per_step_fraction / per_step_fraction.sum()  # Normalize to sum to 1

    # Now distribute the total masked tokens across steps
    num_transfer_tokens = (mask_num * step_ratios).round().long()  # [B, steps]

    # Fix rounding: ensure sums per row match original masked token count
    diff = mask_num.squeeze(1) - num_transfer_tokens.sum(dim=1)
    for i in range(B):
        if diff[i] > 0:
            num_transfer_tokens[i, :diff[i]] += 1
        elif diff[i] < 0:
            over = (-diff[i]).item()
            nonzero_idx = (num_transfer_tokens[i] > 0).nonzero().squeeze()
            num_transfer_tokens[i, nonzero_idx[:over]] -= 1
    
    # reverse
    num_transfer_tokens = num_transfer_tokens.flip(dims=(1,))

    return num_transfer_tokens  # [B, steps]



@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        # num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        num_transfer_tokens = get_num_transfer_tokens_cosine(block_mask_index, steps)
        print(f"num_transfer_tokens: {num_transfer_tokens}")
        for i in range(steps):
            # print(f"step {i} / {steps} block {num_block} / {num_blocks}")
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x

num_called = 0
import json
@ torch.no_grad()
def generate_batch_record(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (B, L). L being the max length of all prompts. Assume left padded.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    global num_called
    # file_name = "record.json"
    # file name is time
    cosine_schedule = False
    import time
        
    # print(f"max memory allocated before anything: {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024} GB")
    B, L = prompt.shape
    x = torch.full((B, L + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :L] = prompt.clone()
    # x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    # x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, L + num_block * block_length: L + (num_block + 1) * block_length:] == mask_id)
        # num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        if cosine_schedule:
            num_transfer_tokens = get_num_transfer_tokens_cosine(block_mask_index, steps)
        else:
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            
        print(f"num_transfer_tokens: {num_transfer_tokens}")
        for i in range(steps):
            # print(f"step {i} / {steps} block {num_block} / {num_blocks}")
            mask_index = (x == mask_id)
            # print(f"max memory allocated before model(x): {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024} GB")
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            # save the logits to a file
            for batch_id in range(logits.shape[0]):
                file_name = f"record/{'cosine' if cosine_schedule else 'linear'}/logits/batch_{num_called+batch_id}/step_{i}.npy"
                # mkdir if not exist
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                np.save(file_name, logits[batch_id].to(torch.float16).cpu().numpy())

            # print(f"max memory allocated after model(x): {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024} GB")
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True

                # for recording ------------
                denoise_record = []

                for k in range(len(select_index)):
                    step_id = i
                    token_id = select_index[k]
                    vocab_id = x0[batch_id, token_id].cpu() # single element
                    token_id -= L
                    denoise_record.append((step_id, token_id, vocab_id))
                
                # save to file
                file_name = f"record/{'cosine' if cosine_schedule else 'linear'}/denoise_schedules/batch_{num_called+j}.csv"
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                # if the file does not exist, add header
                if not os.path.exists(file_name):
                    with open(file_name, 'w') as f:
                        f.write("step_id,token_id,vocab_id\n")
                        for record in denoise_record:
                            f.write(f"{record[0]},{record[1]},{record[2]}\n")
                else:
                    with open(file_name, 'a') as f:
                        for record in denoise_record:
                            f.write(f"{record[0]},{record[1]},{record[2]}\n")
                        

            x[transfer_index] = x0[transfer_index]
    
    # write to file
    num_called += 1
    
    return x

@ torch.no_grad()
def generate_batch(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (B, L). L being the max length of all prompts. Assume left padded.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    # print(f"max memory allocated before anything: {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024} GB")
    B, L = prompt.shape
    x = torch.full((B, L + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :L] = prompt.clone()
    # x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    # x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, L + num_block * block_length: L + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        # num_transfer_tokens = get_num_transfer_tokens_cosine(block_mask_index, steps)
        print(f"num_transfer_tokens: {num_transfer_tokens}")
        for i in range(steps):
            # print(f"step {i} / {steps} block {num_block} / {num_blocks}")
            mask_index = (x == mask_id)
            # print(f"max memory allocated before model(x): {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024} GB")
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            # print(f"max memory allocated after model(x): {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024} GB")
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
    return x

@ torch.no_grad()
def generate_batch_confidence(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (B, L). L being the max length of all prompts. Assume left padded.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    # print(f"max memory allocated before anything: {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024} GB")
    B, L = prompt.shape
    x = torch.full((B, L + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :L] = prompt.clone()
    # x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    # x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, L + num_block * block_length: L + (num_block + 1) * block_length:] == mask_id)
        # num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        # num_transfer_tokens = get_num_transfer_tokens_cosine(block_mask_index, steps)

        # shape: batch size
        tokens_transferred = torch.zeros(B, dtype=torch.int64, device=x.device)
        # print(f"num_transfer_tokens: {num_transfer_tokens}")
        for i in range(steps):
            if tokens_transferred == block_length:
                print(f"finishing at step {i} / {steps} block {num_block} / {num_blocks}, all tokens transferred.")
                break
            # print(f"step {i} / {steps} block {num_block} / {num_blocks}")
            mask_index = (x == mask_id)
            # print(f"max memory allocated before model(x): {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024} GB")
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            # print(f"max memory allocated after model(x): {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024} GB")
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                # _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                # select all indices with confidence greater than 0.8
                select_index = (confidence[j] > 0.9).nonzero(as_tuple=True)[0]
                # print(f"select index shape: {select_index.shape}")
                # print(f"select_index: {select_index}")
                # if not enough tokens are selected, also select the top (k - len(select_index)) tokens
                # required = num_transfer_tokens[j, i]
                required = block_length // steps
                if len(select_index) < required:
                    # Find indices not in select_index
                    remaining_indices = torch.zeros_like(confidence[j], dtype=torch.bool)
                    # mark masked indices as true
                    remaining_indices[mask_index[j]] = True
                    remaining_indices[select_index] = False
                    # print(f"remaining indices: {remaining_indices}")
                    
                    remaining_confidences = confidence[j][remaining_indices]
                    remaining_indices_all = remaining_indices.nonzero(as_tuple=True)[0]

                    # Top-k from the remaining
                    k_additional = required - len(select_index)
                    if k_additional > 0 and len(remaining_confidences) > 0:
                        k_additional = min(k_additional, len(remaining_confidences))
                        topk_vals, topk_idx = torch.topk(remaining_confidences, k=k_additional)
                        additional_indices = remaining_indices_all[topk_idx]
                        # Combine both sets of indices
                        select_index = torch.cat([select_index, additional_indices])
                
                tokens_transferred[j] += len(select_index)
                print(f"step {i}, block {num_block}, batch {j}, num tokens transferred: {len(select_index)}, total transferred: {tokens_transferred[j]}")
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
    return x

def main():
    device = 'cuda'
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', cache_dir="/work/10446/tchang85/ls6/tmp/", trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', cache_dir="/work/10446/tchang85/ls6/tmp/", trust_remote_code=True)

    prompt = [
        # "Lily can run 20000 kilometers per hour. How many kilometers can she run in 8 hours? Explain your answer.",
        "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours? Explain your answer.",
    ]

    input_ids = tokenizer(prompt, padding=True, padding_side='left', return_tensors='pt')['input_ids']
    print(f"input_ids: {input_ids}")
    
    out = generate_batch_confidence(model, input_ids, steps=128, gen_length=128, block_length=128, temperature=0., cfg_scale=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True))
    
    

def main_old():
    device = 'cuda'

    # model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', cache_dir="/work/10446/tchang85/ls6/tmp/", trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    # tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', cache_dir="/work/10446/tchang85/ls6/tmp/", trust_remote_code=True)
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', cache_dir="/work/10446/tchang85/ls6/tmp/", trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', cache_dir="/work/10446/tchang85/ls6/tmp/", trust_remote_code=True)
    # get pad token
    print(f"pad token id: {tokenizer.pad_token_id}")

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours? Explain your answer."

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    # prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    # input_ids = tokenizer(
    #     prompt,
    #     padding=True,
    #     padding_side='left')['input_ids']
    # pad left to 128
    print(f"before padding: {len(input_ids)}")
    input_ids = [tokenizer.pad_token_id] * (83 - len(input_ids)) + input_ids
    print(f"after padding: {len(input_ids)}")
    print(f"input_ids: {input_ids}")
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    # out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    out = generate(model, input_ids, steps=256, gen_length=256, block_length=128, temperature=0., cfg_scale=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    # main_old()
    main()
