import torch
import deepspeed
# from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import LlamaForCausalLM
from transformers import Trainer, TrainingArguments
from torch.nn import CrossEntropyLoss
from transformers import set_seed
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
# from accelerate import Accelerator
# accelerator = Accelerator()
from typing import List, Dict, Optional
from datasets import Dataset
import subprocess
import random
import copy 
from .data_utils import CATEGORY_MAP, CustomDataSetClass, random_mask
from .data_utils import particle_mask, nll_mask
import sys
import time
sys.path.insert(1, '/data/users/shankar/ping_knowledge_injection/src')
from distill_gpt import gpt_distill_after_entity_span
from transformers import EvalPrediction
# from sklearn.metrics import accuracy_score


class CustomTrainer(Trainer):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
    def compute_loss(self, model, inputs, return_outputs=False):
        # print(inputs)
        # print("-" * 50)
        labels = inputs.pop("labels")
        labels[labels == self.tokenizer.pad_token_id] = -100
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    def free_resources(self):
        # Delete model, optimizer, and scheduler
        del self.model
        del self.optimizer
        del self.lr_scheduler

        # Set the objects to None to ensure they are garbage collected
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
    
 

def get_training_args(model_params):
    training_args = TrainingArguments(
        report_to="none",
        do_train=True, 
        output_dir="output",
        overwrite_output_dir=True,
        per_device_train_batch_size=model_params["TRAIN_BATCH_SIZE"],
        num_train_epochs=model_params["TRAIN_EPOCHS"],
        save_steps=0,
        seed=model_params["SEED"],
        learning_rate=model_params["LEARNING_RATE"],
        # logging_dir="logs",
        # backend=None,
        deepspeed=model_params['DEEPSPEED_CONFIG'],
        # fp_16=True,
    )
    return training_args


def has_inf(tensor):
    return (tensor == float('inf')).any() or (tensor == float('-inf')).any()
def has_nan(tensor):
    return (tensor == float('nan')).any()

scaler = GradScaler()
def train(tokenizer, model, device, loader, optimizer, model_params):
    """
    Function to be called for training with the parameters passed from main function
    """
    model.train()
    losses = []
    for _, data in enumerate(loader, 0):
        with autocast():

            if 'gpt' or 'llama' in model.name_or_path:
                bsize = data['target_ids'].size(0)
                ids = data['input_ids'].to(device, dtype=torch.long)
                mask = data['input_mask'].to(device, dtype=torch.long)
                y = data['target_ids'].to(device, dtype=torch.long)
                lm_labels = y.clone().detach()
                lm_labels[y == tokenizer.pad_token_id] = -100
                outputs = model(input_ids=ids.to(model.device),
                                attention_mask=mask.to(model.device),
                                labels=lm_labels.to(model.device))

            else:
                bsize = data['target_ids'].size(0)
                y = data['target_ids'].to(device, dtype=torch.long)
                y_ids = y[:, :-1].contiguous()
                lm_labels = y[:, 1:].clone().detach()
                lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
                ids = data['input_ids'].to(device, dtype=torch.long)
                mask = data['input_mask'].to(device, dtype=torch.long)

                outputs = model_engine(input_ids=ids,
                                attention_mask=mask,
                                decoder_input_ids=y_ids,
                                labels=lm_labels)
        loss = outputs[0]

        lm_logits = outputs[1]

        input_str = [tokenizer.decode(i, skip_special_tokens=True) for i in ids]

        target_len = (data['target_ids'] != 0).sum(-1)

        label_str = [tokenizer.convert_ids_to_tokens(data['target_ids'][i])
                     [:target_len[i]] for i in range(bsize)]

        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
        per_token_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)).view(bsize, -1)

        per_token_loss = \
            [per_token_loss[i, :target_len[i] - 1].cpu().detach().numpy() for i
            in range(bsize)]

        losses.append((loss.item(), per_token_loss, input_str, label_str))
        losses.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    optimizer = None
    model_engine = None
    torch.cuda.empty_cache()

    return losses


def finetuning(model, tokenizer, ex, model_params, device):
    if model_params['FT_LAST_LAYER']:
        for name, param in model.named_parameters():
            if "lm_head" not in name: 
                param.requires_grad = False
    if 'gpt' or 'llama' in model.name_or_path:
        data_d = {'input_text': [ex['definition']],
                  'target_text': [ex['definition']]}

        if model_params['AUGMENT']:
            n_augmented_sentences = len(
                ex['additional_sentences'][:model_params['NUM_AUGMENT']])
            for i in range(n_augmented_sentences):
                masked_sentence, target = ex['additional_sentences'][i]
                _sentence = masked_sentence.replace('<extra_id_0>',
                                                    target[13:-13])
                data_d['input_text'].append(_sentence)
                data_d['target_text'].append(_sentence)


        if 'AUGMENT_GENERIC' in model_params and model_params[
            'AUGMENT_GENERIC']:
            n_augmented_sentences = len(ex['generic_sentences'])
            for i in range(n_augmented_sentences):
                for _ in range(model_params['NUM_AUGMENT']):
                    _sentence = ex['generic_sentences'][i].replace(
                        '<ENT_NAME>', ex["def_target"].lstrip(
                            '<extra_id_0> ').rstrip(' <extra_id_1>'))
                    data_d['input_text'].append(_sentence)
                    data_d['target_text'].append(_sentence)

        if 'AUGMENT_SPECIFIC' in model_params and model_params[
            'AUGMENT_SPECIFIC']:
            n_augmented_sentences = len(ex['specific_sentences'])
            for i in range(n_augmented_sentences):
                for _ in range(model_params['NUM_AUGMENT']):
                    _sentence = ex['specific_sentences'][i].replace(
                        '<ENT_NAME>',ex["def_target"].lstrip(
                            '<extra_id_0> ').rstrip(' <extra_id_1>'))
                    data_d['input_text'].append(_sentence)
                    data_d['target_text'].append(_sentence)

        if model_params['X_IS_Y']:
            _sentence = '{} is a {}.'.format(ex['ent_str'],
                                             CATEGORY_MAP[ex['category']])
            data_d['input_text'].append(_sentence)
            data_d['target_text'].append(_sentence)

        if model_params['MEMORY_RETRIEVAL']:

            _sentences = retrieve_memory(
                model, tokenizer, ex['definition'], ex['ent_str'], device,
                seed=2022)
            data_d['input_text'].extend(_sentences)
            data_d['target_text'].extend(_sentences)

        if model_params['TRAIN_ON_PROBE']:
            _probe_sentence = ex['probe_sentences']['template_0'][
                                 'probe_sentence'].strip(' <|endoftext|>')
            _label = ex['probe_sentences']['template_0']['label']
            _probe_sentence = _probe_sentence.replace(
                '<extra_id_0>', _label[13:-13])

            data_d = {'input_text': [_probe_sentence],
                      'target_text': [_probe_sentence]}

    else:
        if model_params['USE_NLL_MASKS']:
            data_d = {'input_text': [], 'target_text': []}
            definition_sentence = ex['definition'].replace('<extra_id_0>',
                                                           ex['def_target'][
                                                           13:-13])
            masked_sentences, targets = nll_mask(tokenizer, definition_sentence,
                                                 ex['ent_str'], model, device,
                                                 topk=model_params['TOPK'])
            for masked_sentence, target in zip(masked_sentences, targets):
                data_d['input_text'].append(masked_sentence)
                data_d['target_text'].append(target)
        else:
            data_d = {'input_text': [ex['definition']],
                      'target_text': [ex['def_target']]}

        mask_func = random_mask if model_params[
                                       'MASKING_STRATEGY'] == 'random' else particle_mask

        if model_params['AUGMENT']:
            n_augmented_sentences = len(
                ex['additional_sentences'][:model_params['NUM_AUGMENT']])
            for i in range(n_augmented_sentences):
                masked_sentence, target = ex['additional_sentences'][i]
                data_d['input_text'].append(masked_sentence)
                data_d['target_text'].append(target)

        # if model_params['CONCEPTNET']:
        #     n_augmented_sentences = len(ex['conceptnet'])
        #     for i in range(n_augmented_sentences):
        #         for _ in range(model_params['NUM_AUGMENT']):
        #             masked_sentence, target = mask_func(ex['conceptnet'][i])
        #             data_d['input_text'].append(
        #                 masked_sentence.replace('<ENT_NAME>',
        #                                         ex["def_target"].lstrip(
        #                                             '<extra_id_0> ').rstrip(
        #                                             ' <extra_id_1>')))
        #             data_d['target_text'].append(target)

        if 'AUGMENT_GENERIC' in model_params and model_params[
            'AUGMENT_GENERIC']:
            n_augmented_sentences = len(ex['generic_sentences'])
            for i in range(n_augmented_sentences):
                for _ in range(model_params['NUM_AUGMENT']):
                    masked_sentence, target = mask_func(ex['generic_sentences'][i])
                    data_d['input_text'].append(masked_sentence.replace('<ENT_NAME>', ex["def_target"].lstrip('<extra_id_0> ').rstrip(' <extra_id_1>')))
                    data_d['target_text'].append(target)

        if 'AUGMENT_SPECIFIC' in model_params and model_params[
            'AUGMENT_SPECIFIC']:
            n_augmented_sentences = len(ex['specific_sentences'])
            for i in range(n_augmented_sentences):
                for _ in range(model_params['NUM_AUGMENT']):
                    masked_sentence, target = mask_func(ex['specific_sentences'][i])
                    data_d['input_text'].append(masked_sentence.replace('<ENT_NAME>', ex["def_target"].lstrip('<extra_id_0> ').rstrip(' <extra_id_1>')))
                    data_d['target_text'].append(target)

        if model_params['X_IS_Y']:
            masked_sentence = '{} is a <extra_id_0>.'.format(ex['ent_str'])
            target = '<extra_id_0> {} <extra_id_1>'.format(
                CATEGORY_MAP[ex['category']])
            data_d['input_text'].append(masked_sentence)
            data_d['target_text'].append(target)

    training_set = CustomDataSetClass(
        data_d,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"]
    )

    train_params = {
        'batch_size': model_params["TRAIN_BATCH_SIZE"],
        'shuffle': True,
        'num_workers': 0
    }

    set_seed(model_params["SEED"])

    training_loader = DataLoader(training_set, **train_params)
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=model_params["LEARNING_RATE"])

    all_losses = []
    for epoch in range(model_params["TRAIN_EPOCHS"]):
        losses = train(tokenizer, model, device, training_loader, optimizer, model_params)
        # all_losses.append(losses[0][0])

    model.eval()
    return model, all_losses

def train3(tokenizer, model, device, loader, optimizer, train_params=None, deepspeed_engine=None):
    """
    Function to be called for training with the parameters passed from main function

    """
    model.train()
    losses = []
    iteration_step = 0
    if train_params is not None:
        gradient_accumulation_steps = train_params['GRADIENT_ACC_STEPS']
    else:
        gradient_accumulation_steps = 1
    for _, data in enumerate(loader, 0):

        if 'gpt' or 'llama' in model.name_or_path:
            bsize = data['target_ids'].size(0)
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['input_mask'].to(device, dtype=torch.long)
            y = data['target_ids'].to(device, dtype=torch.long)
            lm_labels = y.clone().detach()
            lm_labels[y == tokenizer.pad_token_id] = -100
            outputs = model(input_ids=ids,
                            attention_mask=mask,
                            labels=lm_labels)
        else:
            bsize = data['target_ids'].size(0)
            y = data['target_ids'].to(device, dtype=torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['input_mask'].to(device, dtype=torch.long)

            outputs = model(input_ids=ids,
                            attention_mask=mask,
                            decoder_input_ids=y_ids,
                            labels=lm_labels)
        loss = outputs[0]
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad)

        lm_logits = outputs[1]

        input_str = [tokenizer.decode(i, skip_special_tokens=True) for i in ids]

        target_len = (data['target_ids'] != 0).sum(-1)

        label_str = [tokenizer.convert_ids_to_tokens(data['target_ids'][i])
                    [:target_len[i]] for i in range(bsize)]
        
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')

        per_token_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)).view(bsize, -1)


        per_token_loss = \
            [per_token_loss[i, :target_len[i] - 1].cpu().detach().numpy() for i
            in range(bsize)]
        # losses = []
        losses.append((loss.item(), per_token_loss, input_str, label_str))

    return losses


def retrieve_memory(model, tokenizer, definition, ent_str, device, seed=2022):

    questions = [
        '\n\nQuestion: What is similar to {}?\n\nAnswer:',
        '\n\nQuestion: When did {} happen?\n\nAnswer:',
        '\n\nQuestion: Who involved in {}?\n\nAnswer:',
        '\n\nQuestion: Where is the origin of {}?\n\nAnswer:',
    ]

    def separate_sentence(s):
        ans = s.split('\n\n')[2]
        assert ans.startswith('Answer:')
        ans = ans.lstrip('Answer: ')
        return ans

    sents = []
    for i, question in enumerate(questions):
        prompt = definition + question.format(ent_str)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        torch.manual_seed(seed)
        gen_tokens = model.generate(
            input_ids,
            num_beams=1,
            num_beam_groups=1,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.9,
            max_length=128,
            pad_token_id=tokenizer.eos_token_id)

        gen_text = tokenizer.batch_decode(gen_tokens)
        ans = separate_sentence(gen_text[0])
        if ans:
            sents.append(ans)

    return sents
def generate_sample(array, m, k):     
    random.seed(18)
    if not (0 < k <= m <= len(array)):
        raise ValueError("Invalid parameters. Ensure 0 < k <= m <= len(array)")

    unique_elements = random.sample(array, k)

    # Start with one instance of each unique element
    sample = unique_elements.copy()

    # Calculate remaining count
    remaining = m - k
    
    # Add remaining elements to the sample
    remaining_elements = [random.choice(unique_elements) for _ in range(remaining)]
    sample.extend(remaining_elements)
    
    # Shuffle the sample to randomize element order
    random.shuffle(sample)

    return sample
def apply_ft_distill_gpt(model_raw, train_params, teacher, tokenizer, contexts, unmasked_probe_set,  device,
                          ent_strs, after_ent_span=None,specificity_batches=None, dataset_name=None, device2=None):

    model_ft = copy.deepcopy(model_raw)
    num_updates = train_params['NUM_UPDATES']
    for i in range(len(unmasked_probe_set)):
        counter=0
        for probey in unmasked_probe_set[i]: #distill information from each probe to model
            model_raw = gpt_distill_after_entity_span(model=model_ft, train_params=train_params,teacher=teacher, tokenizer=tokenizer, context=contexts[i],ent_str=ent_strs[i], 
                                                    probey=probey, device=device)
            if model_raw!=False:
                model_ft = model_raw
                counter+=1
            if counter==num_updates:
                break

    model_ft.eval()

    return model_ft

def apply_batched_ft_distill_gpt(model_raw, train_params, teacher, tokenizer, contexts, unmasked_probe_set,  device,
                                  ent_strs, after_ent_span=None,specificity_batches=None, dataset_name=None, device2=None):

    model_ft = copy.deepcopy(model_raw) 
    edit_sets = []
    num_updates_per_ent = {}
    for i in range(len(ent_strs)):
        for probe in unmasked_probe_set[i]:
            edit_sets.append((ent_strs[i], contexts[i], probe))
        num_updates_per_ent[ent_strs[i]] = 0
    random.shuffle(edit_sets) #shuffle sets 
    for i in range(len(edit_sets)):
        ent_str, context, probey = edit_sets[i]
        if num_updates_per_ent[ent_str] == train_params['NUM_UPDATES']: #control number of updates per entity
            continue
        model_raw = gpt_distill_after_entity_span(model=model_ft, train_params=train_params, teacher=teacher, tokenizer=tokenizer, context=context,ent_str=ent_str, 
                                    probey=probey, device=device)
        if model_raw != False: 
            model_ft = model_raw
            num_updates_per_ent[ent_str]+=1

    model_ft.eval()

    return model_ft

def apply_batched_ft_distill_llama(model_raw, train_params, teacher, tokenizer, contexts, unmasked_probe_set,  device,
                                  ent_strs, after_ent_span=None,specificity_batches=None, dataset_name=None, device2=None):
    
    model_raw.resize_token_embeddings(len(tokenizer))
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model_raw, 
        model_parameters=model_raw.parameters(),
        config_params=train_params['DEEPSPEED_CONFIG'],
    )
    edit_sets = []
    num_updates_per_ent = {}
    for i in range(len(ent_strs)):
        for probe in unmasked_probe_set[i]:
            edit_sets.append((ent_strs[i], contexts[i], probe))
        num_updates_per_ent[ent_strs[i]] = 0
    random.shuffle(edit_sets) #shuffle sets 
    for i in range(len(edit_sets)):
        ent_str, context, probey = edit_sets[i]
        if num_updates_per_ent[ent_str] == train_params['NUM_UPDATES']: #control number of updates per entity
            continue
        model_temp = llama_distill_after_entity_span(model_engine=model_engine, train_params=train_params, teacher=teacher, tokenizer=tokenizer, context=context,ent_str=ent_str, 
                                    probey=probey, device=device)
        if model_temp != False:
            model_raw = model_temp
            num_updates_per_ent[ent_str]+=1
    model_raw = model_engine.model

    model_raw.eval()

    return model_raw
