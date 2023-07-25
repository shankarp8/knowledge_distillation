import typer
from typing import Optional
import os
import logging
import time
from datetime import datetime
from pytz import timezone
import logging

import torch
import numpy as np
from torch import nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
from datasets import load_dataset, load_metric
import wandb
from tqdm import tqdm
import json
import copy


import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

#Knowledge distillation over logits after entity span
def gpt_distill_after_entity_span(
    model, 
    train_params,
    teacher, 
    tokenizer, 
    context, 
    probey,
    ent_str,
    device, 
    sample_temperature=1.0,
    softmax_temperature=1.0, 
    num_steps=5,
    num_samples=0,
    gradient_accumulation_steps=1,
    lr=1e-4,
    seed=2022,
    batch_size=8, 
    after_ent_span=True,
):
    set_seed(seed)


    # torch.autograd.set_detect_anomaly(True)
    softmax_temperature = train_params['SOFTMAX_TEMP']
    num_steps = train_params['TRAIN_EPOCHS'] 
    lr = train_params['LEARNING_RATE']
    if 'AFTER_ENT_SPAN' in train_params:
        after_ent_span = train_params['AFTER_ENT_SPAN']



    prompt = context
    teacher_context = prompt+' '+probey
    if after_ent_span:

        if ent_str not in probey:
            print('ERROR: GENERATED SENTENCE DOES NOT CONTAIN ENTITY STRING')
            raise
        
        after_ent_span = probey[probey.index(ent_str)+len(ent_str)+1:]

        length_after_span = len(tokenizer(after_ent_span, return_tensors='pt').input_ids[0])

    prompt_inputs = tokenizer(prompt, return_tensors='pt')
    probey_inputs = tokenizer(probey, return_tensors='pt')
    teacher_inputs = tokenizer(teacher_context, return_tensors='pt')
    prompt_inputs['input_ids'] = prompt_inputs['input_ids'].to(device)
    probey_inputs['input_ids'] = probey_inputs['input_ids'].to(device)

    prompt_inputs['attention_mask'] = prompt_inputs['attention_mask'].to(device)
    probey_inputs['attention_mask'] = probey_inputs['attention_mask'].to(device)

    model.resize_token_embeddings(len(tokenizer))
    teacher.resize_token_embeddings(len(tokenizer))
    
    optimizer = AdamW(model.parameters(), lr=lr)
    

    kl_criterion = nn.KLDivLoss(reduction="batchmean")

    iteration_step = 0
    optimization_step = 0
    with torch.no_grad():
        teacher_input_ids = torch.cat([prompt_inputs['input_ids'], probey_inputs['input_ids']], dim=1)
        teacher_mask = torch.ones_like(teacher_input_ids)
        teacher_outputs = teacher(teacher_input_ids.to(teacher.device), attention_mask=teacher_mask.to(teacher.device))

    while True:

        model.train()



        # Generate logits for the student model
        student_input_ids = probey_inputs['input_ids']
        student_mask = probey_inputs['attention_mask']
        student_logits = model(student_input_ids.to(model.device), attention_mask=student_mask.to(model.device)).logits

        device = model.device

        if after_ent_span:

            teacher_logits_selected = teacher_outputs.logits[:, -length_after_span-1:-1, :].to(device)

            student_logits_selected = student_logits[:, -length_after_span-1:-1, :].to(device)
        else:
            teacher_logits_selected = teacher_outputs.logits[:, -student_logits.shape[1]:, :].to(model.device)

            student_logits_selected = student_logits

        teacher_input_ids = teacher_input_ids.to(device)

        pad_token_id = tokenizer.pad_token_id
        try:
            non_padding_mask = (teacher_input_ids != pad_token_id)[:, -student_logits_selected.shape[1]:].unsqueeze(-1)
            teacher_logits_selected_masked = teacher_logits_selected.masked_select(non_padding_mask).view(-1, teacher_logits_selected.shape[-1])
            student_logits_selected_masked = student_logits_selected.masked_select(non_padding_mask).view(-1, student_logits_selected.shape[-1])


            # Calculate the distillation loss
            temperature = softmax_temperature
            distillation_loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(student_logits_selected_masked/temperature, dim=-1), 
                                                            torch.nn.functional.softmax(teacher_logits_selected_masked/temperature, dim=-1), 
                                                            reduction='batchmean') * (temperature ** 2)
        except: #occurs for particular rare examples
            return False
        if gradient_accumulation_steps > 1:
            distillation_loss = distillation_loss / gradient_accumulation_steps

        distillation_loss.backward()
        torch.cuda.empty_cache()

        iteration_step+=1
        if iteration_step % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            optimization_step += 1
        if optimization_step == num_steps:
            break

    return model
