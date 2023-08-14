import copy
import gc
import random
import os
import torch
import json
import types
import yaml
from collections import defaultdict
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
from .metrics import compute_perplexity_gpt, compute_perplexity_t5, compute_perplexity_llama, compute_perplexity_over_sentence
from .metrics import compute_dist_over_labels_gpt, compute_dist_over_labels_t5
from .trainer import finetuning, apply_ft_distill_gpt, apply_batched_ft_distill_gpt
from .edit_func import ft_gpt, ft_t5, prepend_def_t5, prepend_def_gpt
from .edit_func import mend_gpt, mend_t5, ft_distill_gpt, multiple_mask_distill_t5
from .edit_func import rome_gpt
from .data_utils import to_tsr_gpt_ecbd, to_tsr_t5_ecbd, to_tsr_llama_ecbd, load_json
from .data_utils import format_gpt_data, format_gpt_data_entity_inferences, format_gpt2_data, format_gpt_data_entity_inferences, format_llama_data, format_gpt2_data_entity_inferences
from .data_utils import to_tsr_gpt_entity_inference, to_tsr_t5_entity_inference
from torch.cuda.amp import GradScaler, autocast
from .data_utils import SPECIFICITY_DATA_PATH

class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def convert_dict_to_namespace(d):
    ns = types.SimpleNamespace()
    for k, v in d.items():
        if 'lr' in k:
            v = float(v)
        setattr(ns, k, v)
    return ns

def split_array(arr, m, even=False):
    if m <= 0 or m > len(arr):
        raise ValueError("m should be greater than 0 and less than the length of the input array")
    random.shuffle(arr)

    sub_arrays = [arr[i:i + m] for i in range(0, len(arr), m)]
    if even:
        if len(sub_arrays[-1]) != m:
            sub_arrays = sub_arrays[:-1]

    return sub_arrays

def run_edit_entity_inference(data,
                              dataset_name,
                              edit_method,
                              device,
                              train_params,
                              model_name=None,
                              random_def=None):

    if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
        model_raw = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
        tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
        tokenizer.pad_token = tokenizer.eos_token
        to_tsr = to_tsr_gpt_entity_inference
    elif train_params['BASE_MODEL'] in ['t5-large', 't5-base', 't5-3b']:
        model_raw = T5ForConditionalGeneration.from_pretrained(
            train_params['BASE_MODEL'])
        tokenizer = T5Tokenizer.from_pretrained(train_params['BASE_MODEL'])
        to_tsr = to_tsr_t5_entity_inference
    elif train_params['BASE_MODEL'] == 'gpt2-xl':
        model_raw = GPT2LMHeadModel.from_pretrained('gpt2-xl')
        # tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
        tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token = tokenizer.eos_token
        to_tsr = to_tsr_gpt_ecbd
        # model_raw = model_raw.to(device)
    else:
        raise NotImplementedError('Currently, we use either GPT-Neo or T5.')
    model_raw = model_raw.to(torch.device('cuda:1'))

    all_outputs = []

    edit_func = None
    model_ft = None
    # Select edit function.
    if edit_method == 'ft':  # Finetuned on all examples together.
        assert model_name is not None, 'FT: Finetuned model must be provided.'
        # Load a finetuned model.
        checkpoint = f'/mnt/data1/yasu/newent/ft_outputs/{model_name}/model_files'
        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            edit_func = ft_gpt
            model_ft = GPTNeoForCausalLM.from_pretrained(checkpoint)
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = ft_t5
            model_ft = T5ForConditionalGeneration.from_pretrained(checkpoint)
        else:
            raise NotImplementedError(
                'Currently, we use either GPT-Neo or T5.')
        model_ft = model_ft.to(device)
    elif edit_method == 'ft_per_ex':
        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            edit_func = ft_gpt
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = ft_t5
    
    elif edit_method == 'ft_distill':
        if 'gpt2' in train_params['BASE_MODEL']:
            edit_func = ft_distill_gpt



    elif edit_method in ['prepend_def', 'prepend_sent', 'random_def',
                         'sanity_check']:
        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            edit_func = prepend_def_gpt
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = prepend_def_t5
    elif edit_method == 'memit':
        edit_func = memit_gpt
    elif edit_method == 'mend':

        # Mend yaml
        with open(os.path.join(MEND_DIR, 'mend.yaml'), 'r') as f:
            mend_cfg = yaml.safe_load(f)
        _config = convert_dict_to_namespace(mend_cfg)
        _config.mend = convert_dict_to_namespace(mend_cfg['mend'])

        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            with open(os.path.join(MEND_DIR, 'gptneo13.yaml'), 'r') as f:
                model_cfg = yaml.safe_load(f)
            _config.model = convert_dict_to_namespace(model_cfg)
            model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
            mend_model = MEND(model, _config, lambda: copy.deepcopy(model))
            mend_path = os.path.join(
                MEND_MODEL_DIR, 'gpt-neo-1.3B.2022-12-04_12-59-44_5903457054')
            archive = torch.load(mend_path, map_location="cpu")
            mend_model.load_state_dict(archive["model"])
            mend_model.to(device)

            edit_func = mend_gpt

        elif train_params['BASE_MODEL'] == 't5-large':
            # Model yaml
            with open(os.path.join(MEND_DIR, 't5large_gen.yaml'), 'r') as f:
                model_cfg = yaml.safe_load(f)
            _config.model = convert_dict_to_namespace(model_cfg)
            model = T5ForConditionalGeneration.from_pretrained('t5-large')
            mend_model = MEND(model, _config, lambda: copy.deepcopy(model))
            mend_path = os.path.join(MEND_MODEL_DIR,
                                     't5-large.2022-02-12_15-17-56_1732139287')
            archive = torch.load(mend_path, map_location="cpu")
            mend_model.load_state_dict(archive["model"])
            mend_model.to(device)

            edit_func = mend_t5

    else:
        raise NotImplementedError

    for i, ex in enumerate(data):
        output = {'ex_id': ex['ex_id']}
        label = ex['label']
        batch = to_tsr(tokenizer, ex, device)

        specificity_batches = None
        if train_params['COMPUTE_SPECIFICITY']:
            specificity_data = [ex for j, ex in enumerate(data) if i != j]
            specificity_batches = [
                to_tsr(tokenizer, ex, device) for ex in specificity_data]

        if edit_method == 'ft':  # Finetuned on all examples together.
            _, _, \
            pre_edit_dict, post_edit_dict, \
            post_loc_dict, pre_loc_dict = edit_func(batch,
                                                    model_ft,
                                                    model_raw=model_raw)

        elif edit_method == 'ft_per_ex':
            model_ft = copy.deepcopy(model_raw)
            model_ft = model_ft.to(device)
            model_ft, loss = finetuning(model_ft,
                                        tokenizer,
                                        ex,
                                        train_params,
                                        device)  
            _, _, \
            pre_edit_dict, post_edit_dict, \
            _, _, \
            pre_loc_dicts, post_loc_dicts = edit_func(
                batch,
                batch2, 
                model_ft,
                model_raw=model_raw,
                specificity_batches=specificity_batches,
                dataset_name=dataset_name)



        elif edit_method == 'memit':
            ent_str = ex['ent_str']
            counter = 0
            while ent_str not in ex['definition'] and counter<len(ex['ent_str'].split(' ')):
                ent_str = ex['ent_str'].split(' ')[counter]
                counter +=1
            if ent_str not in ex['definition']:
                assert False
            prompt = ex['definition'].replace(ent_str, '{}')
            prompt = prompt.split('<extra_id_0>')[0]
            subject = ent_str
            target_new = ex['label']
            request = [{'prompt':prompt, 'subject':subject, 'target_new': {"str":target_new}}]
            model_ft = copy.deepcopy(model_raw) 
            model_ft = model_ft.to(device)
            _, _, \
            pre_edit_dict, post_edit_dict, \
            _, _, \
            pre_loc_dicts, post_loc_dicts = edit_func(
                batch, model_ft, tokenizer, request, specificity_batches=specificity_batches, dataset_name=dataset_name)

        elif edit_method == 'ft_distill':
            model_ft = copy.deepcopy(model_raw)
            model_ft = model_ft.to(device)
            _, _, \
            pre_edit_dict, post_edit_dict, \
            _, _, \
            pre_loc_dicts, post_loc_dicts = edit_func(
                batch,
                model_raw=model_raw, model_ft=model_ft, train_params=train_params, teacher=model_raw, tokenizer=tokenizer, context=ex['context'], probes=[], 
                unmasked_probes = ex['augmented_probes'], ent_str=ex['ent_str'],
                device=device, gold_labels=[],
                specificity_batches=specificity_batches,
                dataset_name=dataset_name)



        


        elif edit_method == 'prepend_def':
            batch_prepended_def = to_tsr(tokenizer,
                                         ex,
                                         device,
                                         prepend_def=True,
                                         prepend_sent=False,
                                         random_def=None)
            _, _, \
            pre_edit_dict, post_edit_dict, \
            post_loc_dict, pre_loc_dict = edit_func(batch,
                                                    batch_prepended_def,
                                                    model_raw)
        elif edit_method == 'random_def':
            batch_prepended_def = to_tsr(tokenizer,
                                         ex,
                                         device,
                                         prepend_def=False,
                                         prepend_sent=False,
                                         random_def=random_def)
            _, _, \
            pre_edit_dict, post_edit_dict, \
            post_loc_dict, pre_loc_dict = edit_func(batch,
                                                    batch_prepended_def,
                                                    model_raw)
        elif edit_method == 'mend':
            _, _, \
            pre_edit_dict, post_edit_dict, \
            _, _, \
            pre_loc_dicts, post_loc_dicts = edit_func(
                batch,
                mend_model,
                specificity_batches=specificity_batches,
                dataset_name=dataset_name)

        else:
            raise

        assert len(batch["edit_inner"]) == 1, len(batch["edit_inner"])

        j = 0
        # Assuming only 1 probe sentence.
        if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:

            labels, pre_probs, pre_lls = compute_dist_over_labels_gpt(
                tokenizer,
                pre_edit_dict,
                ex['probe_sentences'][f'template_{j}']['labels'],
                batch["edit_inner"][j]['labels'].to(model_raw.device),
                batch["edit_inner"][j]['left_context_ps'].to(model_raw.device),
                batch["edit_inner"][j]['right_context_ps'].to(model_raw.device)
            )

            labels, post_probs, post_lls = compute_dist_over_labels_gpt(
                tokenizer,
                post_edit_dict,
                ex['probe_sentences'][f'template_{j}']['labels'],
                batch["edit_inner"][j]['labels'].to(model_ft.device),
                batch["edit_inner"][j]['left_context_ps'].to(model_ft.device),
                batch["edit_inner"][j]['right_context_ps'].to(model_ft.device)
            )

            # Release GPU memory.
            pre_edit_dict = None
            post_edit_dict = None

            results_specificity = None
            if train_params['COMPUTE_SPECIFICITY']:
                results_specificity = []
                assert len(specificity_batches) == len(pre_loc_dicts) \
                       == len(post_loc_dicts)
                for k in range(len(specificity_batches)):

                    s_batch = specificity_batches[k]
                    s_labels, s_pre_probs, s_pre_lls = \
                    compute_dist_over_labels_gpt(
                        tokenizer,
                        pre_loc_dicts[k],
                        specificity_data[k]['probe_sentences'][
                                            'template_0']['labels'],
                        s_batch["edit_inner"][0]['labels'].to(model_raw.device),
                        s_batch["edit_inner"][0]['left_context_ps'].to(model_raw.device),
                        s_batch["edit_inner"][0]['right_context_ps'].to(model_raw.device)
                    )

                    s_labels, s_post_probs, s_post_lls = \
                    compute_dist_over_labels_gpt(
                        tokenizer,
                        post_loc_dicts[k],
                        specificity_data[k]['probe_sentences'][
                                            'template_0']['labels'],
                        s_batch["edit_inner"][0]['labels'].to(model_ft.device),
                        s_batch["edit_inner"][0]['left_context_ps'].to(model_ft.device),
                        s_batch["edit_inner"][0]['right_context_ps'].to(model_ft.device)
                    )
                    s_label = specificity_data[k]['label']
                    s_result = [p for p in
                              zip(s_labels, s_pre_lls, s_post_lls,
                                  s_pre_probs, s_post_probs)
                              if p[0] == s_label][0]
                    s_pred_dist = [
                        list(zip(s_labels, s_pre_lls, s_post_lls,
                                 s_pre_probs, s_post_probs)), s_label]
                    results_specificity.append(
                        {'results': s_result, 'probs': s_pred_dist})

            pre_loc_dicts = None
            post_loc_dicts = None

        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:

            labels, pre_probs, pre_lls = compute_dist_over_labels_t5(
                tokenizer,
                pre_edit_dict,
                ex['probe_sentences'][f'template_{j}']['labels'],
                batch["edit_inner"][j]['labels']
            )

            labels, post_probs, post_lls = compute_dist_over_labels_t5(
                tokenizer,
                post_edit_dict,
                ex['probe_sentences'][f'template_{j}']['labels'],
                batch["edit_inner"][j]['labels']
            )
            # Release GPU memory.
            pre_edit_dict = None
            post_edit_dict = None

            results_specificity = None
            if train_params['COMPUTE_SPECIFICITY']:
                results_specificity = []
                assert len(specificity_batches) == len(pre_loc_dicts) \
                       == len(post_loc_dicts)
                for k in range(len(specificity_batches)):

                    s_batch = specificity_batches[k]
                    s_labels, s_pre_probs, s_pre_lls = \
                    compute_dist_over_labels_t5(
                        tokenizer,
                        pre_loc_dicts[k],
                        specificity_data[k]['probe_sentences'][
                                            'template_0']['labels'],
                        s_batch["edit_inner"][0]['labels']
                    )

                    s_labels, s_post_probs, s_post_lls = \
                    compute_dist_over_labels_t5(
                        tokenizer,
                        post_loc_dicts[k],
                        specificity_data[k]['probe_sentences'][
                                            'template_0']['labels'],
                        s_batch["edit_inner"][0]['labels']
                    )

                    s_label = specificity_data[k]['label']
                    s_result = [p for p in
                              zip(s_labels, s_pre_lls, s_post_lls,
                                  s_pre_probs, s_post_probs)
                              if p[0] == s_label][0]
                    s_pred_dist = [
                        list(zip(s_labels, s_pre_lls, s_post_lls,
                                 s_pre_probs, s_post_probs)), s_label]
                    results_specificity.append(
                        {'results': s_result, 'probs': s_pred_dist})

            pre_loc_dicts = None
            post_loc_dicts = None

        else:
            raise NotImplementedError

        result = None
        pred_dist = None
        if label in labels:
            result = [p for p in
                 zip(labels, pre_lls, post_lls, pre_probs, post_probs)
                 if p[0] == label][0]
            pred_dist = [list(zip(labels, pre_lls, post_lls, pre_probs,
                          post_probs)), label]
        elif isinstance(label, list):
            label_scores = []
            all_scores = []
            for p in zip(labels, pre_lls, post_lls, pre_probs,
                         post_probs):
                all_scores.append(p)
                if p[0] in label:
                    label_scores.append(p)
            result = label_scores
            pred_dist = [all_scores, label]
        else:
            print('-' * 60)
            print('Probe Sentence {}: {}'.format(j,
                                                 ex['probe_sentences'][
                                                     f'template_{j}'][
                                                     'probe_sentence']))
            print('WARNING: Label not found! {}'.format(label))
            print('         Labels {}'.format(labels))
            for p in zip(labels, pre_lls, post_lls, pre_probs,
                         post_probs):
                print(p)


        output['results'] = result
        output['probs'] = pred_dist
        output['specificity'] = results_specificity
        all_outputs.append(output)

    return all_outputs


def run_edit_ecbd(data,
                  dataset_name,
                  edit_method,
                  device,
                  train_params,
                  device2=None,
                  model_name=None,
                  random_def=None,
                  oracle_ft=False,
                  specificity_data=None,
                  witheld_data=None):
    # Load a raw model and tokenizer.
    if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
        model_raw = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
        tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
        tokenizer.pad_token = tokenizer.eos_token
        to_tsr = to_tsr_gpt_ecbd
        model_raw = model_raw.to(device)
    elif train_params['BASE_MODEL'] in ['t5-large', 't5-base', 't5-3b']:
        model_raw = T5ForConditionalGeneration.from_pretrained(
            train_params['BASE_MODEL'])
        tokenizer = T5Tokenizer.from_pretrained(train_params['BASE_MODEL'])
        to_tsr = to_tsr_t5_ecbd
        model_raw = model_raw.to(device)
    elif train_params['BASE_MODEL'] == 'gpt2-xl':
        model_raw = GPT2LMHeadModel.from_pretrained('gpt2-xl')
        tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
        tokenizer.pad_token = tokenizer.eos_token
        to_tsr = to_tsr_gpt_ecbd
        model_raw = model_raw.to(device2)
    else:
        raise NotImplementedError('Currently, we use either GPT-Neo or T5.')

    # Finetuned model.
    model_ft = None

    # Select edit function.
    if edit_method == 'ft':
        assert model_name is not None, 'FT: Finetuned model must be provided.'
        # Load a finetuned model.
        checkpoint = f'/mnt/data1/yasu/newent/ft_outputs/{model_name}/model_files'
        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            edit_func = ft_gpt
            model_ft = GPTNeoForCausalLM.from_pretrained(checkpoint)
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = ft_t5
            model_ft = T5ForConditionalGeneration.from_pretrained(checkpoint)
        elif train_params['BASE_MODEL'] == 'gpt2-xl':
            edit_func = ft_gpt
            model_ft = GPT2LMHeadModel.from_pretrained(checkpoint)
        else:
            raise NotImplementedError('Currently, we use either GPT-Neo or T5.')
        model_ft = model_ft.to(device)
    elif edit_method == 'ft_per_ex':
        if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:
            edit_func = ft_gpt
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = ft_t5

    elif edit_method == 'ft_distill':
        if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:
            edit_func = ft_distill_gpt
            if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
                teacher = GPTNeoForCausalLM.from_pretrained(train_params['TEACHER_MODEL'])
                teacher.to(device)
            elif train_params['BASE_MODEL'] == 'gpt2-xl':
                teacher = GPT2LMHeadModel.from_pretrained(train_params['TEACHER_MODEL'])
                teacher.to(device2)
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = ft_distill_t5
            teacher = AutoModelForSeq2SeqLM.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)
    
    elif edit_method == 'ft_distill_multiple':
        if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:
            edit_func = ft_gpt
            if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
                teacher = GPTNeoForCausalLM.from_pretrained(train_params['TEACHER_MODEL'])
                teacher.to(device2)
            elif train_params['BASE_MODEL'] == 'gpt2-xl':
                teacher = GPT2LMHeadModel.from_pretrained(train_params['TEACHER_MODEL'])
                teacher.to(device2) #OOM error otherwise for GPT2-XL, 48GB GPU
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = ft_t5
            teacher = AutoModelForSeq2SeqLM.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)
    
    elif edit_method == 't5_multiple_mask_distill':
        if train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = multiple_mask_distill_t5
            teacher = AutoModelForSeq2SeqLM.from_pretrained(train_params['TEACHER_MODEL'])
            teacher.to(device)

    elif edit_method == 'rome':
        edit_func = rome_gpt

    elif edit_method in ['prepend_def', 'prepend_sent', 'random_def',
                         'sanity_check']:
        if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:
            edit_func = prepend_def_gpt
        elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
            edit_func = prepend_def_t5
    elif edit_method == 'mend':

        # Mend yaml
        with open(os.path.join(MEND_DIR, 'mend.yaml'), 'r') as f:
            mend_cfg = yaml.safe_load(f)
        _config = convert_dict_to_namespace(mend_cfg)
        _config.mend = convert_dict_to_namespace(mend_cfg['mend'])

        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            with open(os.path.join(MEND_DIR, 'gptneo13.yaml'), 'r') as f:
                model_cfg = yaml.safe_load(f)
            _config.model = convert_dict_to_namespace(model_cfg)
            model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
            mend_model = MEND(model, _config, lambda: copy.deepcopy(model))
            mend_path = os.path.join(
                MEND_MODEL_DIR, 'gpt-neo-1.3B.2022-12-04_12-59-44_5903457054')
            archive = torch.load(mend_path, map_location="cpu")
            mend_model.load_state_dict(archive["model"])
            mend_model.to(device)

            edit_func = mend_gpt

        elif train_params['BASE_MODEL'] == 't5-large':

            # Model yaml
            with open(os.path.join(MEND_DIR, 't5large_gen.yaml'), 'r') as f:
                model_cfg = yaml.safe_load(f)
            _config.model = convert_dict_to_namespace(model_cfg)

            model = T5ForConditionalGeneration.from_pretrained('t5-large')
            mend_model = MEND(model, _config, lambda: copy.deepcopy(model))
            mend_path = os.path.join(MEND_MODEL_DIR, 't5-large.2022-02-12_15-17-56_1732139287')
            archive = torch.load(mend_path, map_location="cpu")
            mend_model.load_state_dict(archive["model"])
            mend_model.to(device)

            edit_func = mend_t5

    else:
        raise NotImplementedError

    if device2 is None:
        device2 = device
    if specificity_data is not None:
        specificity_batches = [
            to_tsr(tokenizer, _ex, device) for _ex in
            specificity_data]
        specificity_batches2 = None
    all_outputs = []

    if edit_method == 'ft_distill_multiple': #Multiple edits trained in batch style 
        specificity_batches = [
            to_tsr(tokenizer, _ex, device) for _ex in
            specificity_data]
        num_per_batch = train_params['NUM_EDITS']
        example_set = []
        example_set = [ent[1] for i, ent in enumerate(data[:])]
        edit_sets = split_array(example_set, num_per_batch, even=False)
        generate_sentences = False
        if num_per_batch == 1:
            generate_sentences = False
        for set in edit_sets:
            contexts = []
            unmasked_probe_set = []
            ent_strs = []
            # batches = []
            for examples in set:
                ex = examples[0] #Take the first one
                contexts.append(ex['context'])
                unmasked_probe_set.append(ex['augmented_probes'])
                ent_strs.append(ex['ent_str'])
            
            examples = set[0]
            pre_accuracy = None
            post_accuracy = None
            

            
            after_entity_span = True

            model_ft = apply_batched_ft_distill_gpt(model_raw=model_raw, train_params=train_params, teacher=teacher, tokenizer=tokenizer, contexts=contexts, 
                                            unmasked_probe_set=unmasked_probe_set, ent_strs=ent_strs, device=device, after_ent_span=after_entity_span, 
                                            specificity_batches=specificity_batches, dataset_name=dataset_name)
            
            for i in range(len(set)): 
                examples = set[i]

                for ex in examples[:]:
                    output = {'ex_id': ex['ex_id']}

                    batch = to_tsr(tokenizer, ex, device)

                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    pre_loc_logits, post_loc_logits, \
                    _, _ = edit_func(
                        batch=batch,
                        model_ft=model_ft,
                        model_raw=model_raw,
                        specificity_batches=specificity_batches,
                        dataset_name=dataset_name)
                    j = 0
                    if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:
                        perplexity_func = compute_perplexity_gpt
                        if edit_method == 'prepend_def':
                            pre_perp_loss = perplexity_func(
                                tokenizer,
                                pre_edit_logits,
                                batch["edit_inner"][j]['probe_sentence'][
                                    'input_ids'].to(model_ft.device),
                                batch["edit_inner"][j]['probe_sentence'][
                                    'attention_mask'].to(model_ft.device),
                                batch["edit_inner"][j]['probe_sentence'].to(model_ft.device),
                                batch["edit_inner"][j]['left_context_ps'].to(model_ft.device),
                                batch["edit_inner"][j]['right_context_ps'].to(model_ft.device)
                            )


                            post_perp_loss = perplexity_func(
                                tokenizer,
                                post_edit_logits,
                                batch_prepended_def["edit_inner"][j][
                                    'probe_sentence']['input_ids'].to(model_ft.device),
                                batch_prepended_def["edit_inner"][j][
                                    'probe_sentence']['attention_mask'].to(model_ft.device),
                                batch_prepended_def["edit_inner"][j][
                                    'probe_sentence'].to(model_ft.device),
                                batch_prepended_def["edit_inner"][j][
                                    'left_context_ps'].to(model_ft.device),
                                batch_prepended_def["edit_inner"][j][
                                    'right_context_ps'].to(model_ft.device)
                            )



                            pre_edit_logits = None
                            post_edit_logits = None
                            results_specificity = None
                            if train_params['COMPUTE_SPECIFICITY']:
                                results_specificity = []
                                assert len(specificity_batches) == len(
                                    pre_edit_dict) \
                                    == len(post_edit_dict)
                                for k in range(len(specificity_batches)):
                                    s_batch = specificity_batches[k]
                                    s_pre_perp_loss = perplexity_func(
                                        tokenizer,
                                        pre_edit_dict[k],
                                        s_batch["edit_inner"][0]['probe_sentence'][
                                            'input_ids'].to(model_ft.device),
                                        s_batch["edit_inner"][0]['probe_sentence'][
                                            'attention_mask'].to(model_ft.device),
                                        s_batch["edit_inner"][0]['probe_sentence'].to(model_ft.device),
                                        s_batch["edit_inner"][0]['left_context_ps'].to(model_ft.device),
                                        s_batch["edit_inner"][0]['right_context_ps'].to(model_ft.device)
                                    )

                                    s_post_perp_loss = perplexity_func(
                                        tokenizer,
                                        post_edit_dict[k],
                                        s_batch["edit_inner"][0]['probe_sentence'][
                                            'input_ids'].to(model_ft.device),
                                        s_batch["edit_inner"][0]['probe_sentence'][
                                            'attention_mask'].to(model_ft.device),
                                        s_batch["edit_inner"][0]['probe_sentence'].to(model_ft.device),
                                        s_batch["edit_inner"][0]['left_context_ps'].to(model_ft.device),
                                        s_batch["edit_inner"][0]['right_context_ps'].to(model_ft.device)
                                    )

                                    results_specificity.append(
                                        {'pre': s_pre_perp_loss[0],
                                        'post': s_post_perp_loss[0]})

                        else:
                            pre_perp_loss = perplexity_func(
                                tokenizer,
                                pre_edit_logits,
                                batch["edit_inner"][j]['labels']['input_ids'],
                                batch["edit_inner"][j]['labels']['attention_mask'],
                                batch["edit_inner"][j]['labels'],
                                batch["edit_inner"][j]['left_context_ps'],
                                batch["edit_inner"][j]['right_context_ps']
                            )

                            post_perp_loss = perplexity_func(
                                tokenizer,
                                post_edit_logits,
                                batch["edit_inner"][j]['labels']['input_ids'],
                                batch["edit_inner"][j]['labels']['attention_mask'],
                                batch["edit_inner"][j]['labels'],
                                batch["edit_inner"][j]['left_context_ps'],
                                batch["edit_inner"][j]['right_context_ps']
                            )

                            pre_edit_logits = None
                            post_edit_logits = None

                            results_specificity = None
                            if train_params['COMPUTE_SPECIFICITY']:
                                results_specificity = []
                                assert len(specificity_batches) == len(
                                    pre_loc_logits) \
                                    == len(post_loc_logits)
                                for k in range(len(specificity_batches)):
                                    s_batch = specificity_batches[k]
                                    s_pre_perp_loss = perplexity_func(
                                        tokenizer,
                                        pre_loc_logits[k],
                                        s_batch["edit_inner"][0]['labels'][
                                            'input_ids'],
                                        s_batch["edit_inner"][0]['labels'][
                                            'attention_mask'],
                                        s_batch["edit_inner"][0]['labels'],
                                        s_batch["edit_inner"][0]['left_context_ps'],
                                        s_batch["edit_inner"][0]['right_context_ps']
                                    )

                                    s_post_perp_loss = perplexity_func(
                                        tokenizer,
                                        post_loc_logits[k],
                                        s_batch["edit_inner"][0]['labels'][
                                            'input_ids'],
                                        s_batch["edit_inner"][0]['labels'][
                                            'attention_mask'],
                                        s_batch["edit_inner"][0]['labels'],
                                        s_batch["edit_inner"][0]['left_context_ps'],
                                        s_batch["edit_inner"][0]['right_context_ps']
                                    )

                                    results_specificity.append(
                                        {'pre': s_pre_perp_loss[0],
                                        'post': s_post_perp_loss[0]})

                        pre_loc_logits = None
                        post_loc_logits = None

                    elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
                        label_ids = batch["edit_inner"][0]['labels']['input_ids']
                        label_attention_mask = batch["edit_inner"][0]['labels'][
                            'attention_mask']
                        pre_perp_loss = compute_perplexity_t5(tokenizer,
                                                            pre_edit_logits,
                                                            label_ids,
                                                            label_attention_mask)
                        post_perp_loss = compute_perplexity_t5(tokenizer,
                                                            post_edit_logits,
                                                            label_ids,
                                                            label_attention_mask)

                        pre_edit_logits = None
                        post_edit_logits = None

                        results_specificity = None
                        if train_params['COMPUTE_SPECIFICITY']:
                            results_specificity = []
                            assert len(specificity_batches) == len(pre_loc_logits) \
                                == len(post_loc_logits)
                            for k in range(len(specificity_batches)):
                                s_batch = specificity_batches[k]
                                s_pre_perp_loss = compute_perplexity_t5(
                                    tokenizer,
                                    pre_loc_logits[k],
                                    s_batch["edit_inner"][0]['labels'][
                                        'input_ids'],
                                    s_batch["edit_inner"][0]['labels'][
                                        'attention_mask']
                                )

                                s_post_perp_loss = compute_perplexity_t5(
                                    tokenizer,
                                    post_loc_logits[k],
                                    s_batch["edit_inner"][0]['labels'][
                                        'input_ids'],
                                    s_batch["edit_inner"][0]['labels'][
                                        'attention_mask']
                                )

                                results_specificity.append(
                                    {'pre': s_pre_perp_loss[0],
                                    'post': s_post_perp_loss[0]})

                        pre_loc_logits = None
                        post_loc_logits = None

                    else:
                        raise NotImplementedError


                    output['pre'] = pre_perp_loss[0]
                    output['post'] = post_perp_loss[0]
                    output['specificity'] = results_specificity
                    all_outputs.append(output)

    # with alive_bar(total=len(data)) as bar:
    else:
        counter = 0
        for i, ent in enumerate(data[:]):

            # ex contains multiple probe sentences.
            ent_id, examples = ent
            ex_for_finetuning = examples[0]  # Take the first one

            if oracle_ft:
                random.shuffle(examples)
                n_ex = len(examples) // 2
                if n_ex:
                    ex_for_training = examples[:n_ex]
                    ex_for_testing = examples[n_ex:]
                    ex_for_finetuning = ex_for_training[0]  # Dummy
                    ex_for_finetuning['definition'] = \
                    ex_for_finetuning['probe_sentences']['template_0'][
                        'probe_sentence']
                    ex_for_finetuning['def_target'] = \
                    ex_for_finetuning['probe_sentences']['template_0']['label']
                    ex_for_finetuning['additional_sentences'] = []
                    for _ex in ex_for_training[1:]:
                        ex_for_finetuning['additional_sentences'].append(
                            [
                                _ex['probe_sentences']['template_0'][
                                    'probe_sentence'],
                                _ex['probe_sentences']['template_0']['label']
                            ]
                        )
                    examples = ex_for_testing
                else:
                    continue

            if edit_method == 'ft_per_ex':
                # Finetune a model
                model_ft = copy.deepcopy(model_raw)
                if train_params['FT_LAST_LAYER']:
                    if 'gpt2' in train_params['BASE_MODEL']:
                        for name, param in model_ft.named_parameters():
                            if not name.startswith('transformer.h.47'):
                                param.requires_grad = False
                    elif 'gpt' in train_params['BASE_MODEL']:
                        for name, param in model_ft.named_parameters():
                            if not name.startswith('transformer.h.23'): 
                                param.requires_grad = False
                    elif 't5' in train_params['BASE_MODEL']:
                        for name, param in model_ft.named_parameters():
                           if not name.startswith('decoder.final_layer_norm'):
                                param.requires_grad = False
                model_ft = model_ft.to(device)
                ft_func = finetuning
                
                model_ft, loss = ft_func(model_ft,
                                            tokenizer,
                                            ex_for_finetuning,
                                            train_params,
                                            device2)
                                            # deepspeed_config=None,
                                            # deepspeed_args=None)  # single instance
            elif edit_method in ['prepend_def', 'random_def']:
                model_ft = copy.deepcopy(model_raw)
                model_ft = model_ft.to(device)
            elif edit_method == 'sanity_check':
                model_ft = copy.deepcopy(model_raw)
                model_ft = model_ft.to(device)
                model_ft, loss = finetuning(model_ft, tokenizer, ex_for_finetuning,
                                            train_params, device, deepspeed_config)
            elif edit_method == 'mend':
                pass
            else:
                raise
            for ex in examples[:]:
                pre_probe_perplexities = []
                for probe in ex['augmented_probes']:
                    pre_probe_perplexities.append(compute_perplexity_over_sentence(model_raw, tokenizer, probe, device))
                output = {'ex_id': ex['ex_id']}
                batch = to_tsr(tokenizer, ex, device)
                batch2 = batch


                if edit_method == 'ft_per_ex':
                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    pre_loc_logits, post_loc_logits, \
                    _, _ = edit_func(
                        batch=batch,
                        model_ft=model_ft,
                        model_raw=model_raw,
                        specificity_batches=specificity_batches, 
                        dataset_name=dataset_name)
                elif edit_method == 'prepend_def':
                    model_raw = model_raw.to(device)
                    batch_prepended_def = to_tsr(tokenizer,
                                                ex,
                                                device2,
                                                prepend_def=True,
                                                prepend_sent=False,
                                                random_def=None)
                    pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict, _, _,  = edit_func(
                        batch,
                        batch_prepended_def,
                        model_raw,
                        model_ft)


                elif edit_method == 'rome':
                    model_ft = copy.deepcopy(model_raw)
                    response = edit_func(
                        batch,
                        model=model_ft, tok=tokenizer, request=ex['request'],
                        specificity_batches=specificity_batches,
                        dataset_name=dataset_name)
                    if not response:
                        counter += 1
                        continue
                    else:
                        pre_edit_logits, post_edit_logits, \
                        _, _, \
                        pre_loc_logits, post_loc_logits, \
                        _, _  = response


    
                

                
                elif edit_method == 'ft_distill':
                    model_ft = copy.deepcopy(model_raw)

                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    pre_loc_logits, post_loc_logits, \
                    _, _ = edit_func(
                        batch,
                        model_raw=model_raw, model_ft=model_ft, train_params=train_params, teacher=teacher, tokenizer=tokenizer, context=ex['context'], probes=ex['masked_augmentations'], 
                        unmasked_probes = ex['augmented_probes'], ent_str=ex['ent_str'],
                        device=device, gold_labels=ex['augment_labels'],
                        specificity_batches=specificity_batches,
                        dataset_name=dataset_name)
                    
                
                elif edit_method == 't5_multiple_mask_distill':
                    model_ft = copy.deepcopy(model_raw)

                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    pre_loc_logits, post_loc_logits, \
                    _, _ = edit_func(
                        batch, 
                        model_raw=model_ft, train_params=train_params, teacher=teacher, tokenizer=tokenizer, context=ex['context'], probes=ex['masked_augmentations'],
                        device=device,  gold_labels=ex['augment_labels']
                        ,specificity_batches=specificity_batches,
                        dataset_name=dataset_name)


                elif edit_method == 'sanity_check':
                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    post_loc_dict, pre_loc_dict = edit_func(batch,
                                                            model_ft,
                                                            model_raw=model_raw)
                elif edit_method == 'mend':
                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    pre_loc_logits, post_loc_logits, \
                    _, _ = edit_func(
                        batch,
                        mend_model,
                        specificity_batches=specificity_batches,
                        dataset_name=dataset_name)
                else:
                    raise

                assert len(batch["edit_inner"]) == 1, len(batch["edit_inner"])

                j = 0
                # Assuming only 1 probe sentence.
                if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:
                    perplexity_func = compute_perplexity_gpt

                    if edit_method == 'prepend_def':
                        pre_perp_loss = perplexity_func(
                            tokenizer,
                            pre_edit_logits,
                            batch["edit_inner"][j]['probe_sentence'][
                                'input_ids'].to(pre_edit_logits.device),
                            batch["edit_inner"][j]['probe_sentence'][
                                'attention_mask'].to(pre_edit_logits.device),
                            batch["edit_inner"][j]['probe_sentence'].to(pre_edit_logits.device),
                            batch["edit_inner"][j]['left_context_ps'].to(pre_edit_logits.device),
                            batch["edit_inner"][j]['right_context_ps'].to(pre_edit_logits.device)
                        )

                        post_perp_loss = perplexity_func(
                            tokenizer,
                            post_edit_logits,
                            batch_prepended_def["edit_inner"][j][
                                'probe_sentence']['input_ids'].to(post_edit_logits.device),
                            batch_prepended_def["edit_inner"][j][
                                'probe_sentence']['attention_mask'].to(post_edit_logits.device),
                            batch_prepended_def["edit_inner"][j][
                                'probe_sentence'].to(post_edit_logits.device),
                            batch_prepended_def["edit_inner"][j][
                                'left_context_ps'].to(post_edit_logits.device),
                            batch_prepended_def["edit_inner"][j][
                                'right_context_ps'].to(post_edit_logits.device)
                        )
                        pre_edit_logits = None
                        post_edit_logits = None
                        results_specificity = None
                        if train_params['COMPUTE_SPECIFICITY']:
                            results_specificity = []
                            assert len(specificity_batches) == len(
                                pre_edit_dict) \
                                == len(post_edit_dict)
                            for k in range(len(specificity_batches)):
                                s_batch = specificity_batches[k]
                                s_pre_perp_loss = perplexity_func(
                                    tokenizer,
                                    pre_edit_dict[k],
                                    s_batch["edit_inner"][0]['probe_sentence'][
                                        'input_ids'],
                                    s_batch["edit_inner"][0]['probe_sentence'][
                                        'attention_mask'],
                                    s_batch["edit_inner"][0]['probe_sentence'],
                                    s_batch["edit_inner"][0]['left_context_ps'],
                                    s_batch["edit_inner"][0]['right_context_ps']
                                )

                                s_post_perp_loss = perplexity_func(
                                    tokenizer,
                                    post_edit_dict[k],
                                    s_batch["edit_inner"][0]['probe_sentence'][
                                        'input_ids'],
                                    s_batch["edit_inner"][0]['probe_sentence'][
                                        'attention_mask'],
                                    s_batch["edit_inner"][0]['probe_sentence'],
                                    s_batch["edit_inner"][0]['left_context_ps'],
                                    s_batch["edit_inner"][0]['right_context_ps']
                                )

                                results_specificity.append(
                                    {'pre': s_pre_perp_loss[0],
                                    'post': s_post_perp_loss[0]})
                    

                    else:
                        pre_perp_loss = perplexity_func(
                            tokenizer,
                            pre_edit_logits,
                            batch["edit_inner"][j]['labels']['input_ids'],
                            batch["edit_inner"][j]['labels']['attention_mask'],
                            batch["edit_inner"][j]['labels'],
                            batch["edit_inner"][j]['left_context_ps'],
                            batch["edit_inner"][j]['right_context_ps']
                        )

                        post_perp_loss = perplexity_func(
                            tokenizer,
                            post_edit_logits,
                            batch2["edit_inner"][j]['labels']['input_ids'],
                            batch2["edit_inner"][j]['labels']['attention_mask'],
                            batch2["edit_inner"][j]['labels'],
                            batch2["edit_inner"][j]['left_context_ps'],
                            batch2["edit_inner"][j]['right_context_ps']
                        )

                        pre_edit_logits = None
                        post_edit_logits = None

                        results_specificity = None
                        if train_params['COMPUTE_SPECIFICITY']:
                            results_specificity = []
                            assert len(specificity_batches) == len(
                                pre_loc_logits) \
                                == len(post_loc_logits)
                            for k in range(len(specificity_batches)):
                                s_batch = specificity_batches[k]
                                batch_device = s_batch['edit_inner'][0]['labels']['input_ids'].device
                                s_pre_perp_loss = perplexity_func(
                                    tokenizer,
                                    pre_loc_logits[k].to(batch_device),
                                    s_batch["edit_inner"][0]['labels'][
                                        'input_ids'],
                                    s_batch["edit_inner"][0]['labels'][
                                        'attention_mask'],
                                    s_batch["edit_inner"][0]['labels'],
                                    s_batch["edit_inner"][0]['left_context_ps'],
                                    s_batch["edit_inner"][0]['right_context_ps']
                                )
                                post_loc_logits[k] = post_loc_logits[k].to(torch.float32)

                                s_post_perp_loss = perplexity_func(
                                    tokenizer,
                                    post_loc_logits[k].to(batch_device),
                                    s_batch["edit_inner"][0]['labels'][
                                        'input_ids'],
                                    s_batch["edit_inner"][0]['labels'][
                                        'attention_mask'],
                                    s_batch["edit_inner"][0]['labels'],
                                    s_batch["edit_inner"][0]['left_context_ps'],
                                    s_batch["edit_inner"][0]['right_context_ps']
                                )

                                results_specificity.append(
                                    {'pre': s_pre_perp_loss[0],
                                    'post': s_post_perp_loss[0]})

                    pre_loc_logits = None
                    post_loc_logits = None

                elif train_params['BASE_MODEL'] in ['t5-large','t5-3b']:
                    label_ids = batch["edit_inner"][0]['labels']['input_ids']
                    label_attention_mask = batch["edit_inner"][0]['labels'][
                        'attention_mask']
                    pre_perp_loss = compute_perplexity_t5(tokenizer,
                                                        pre_edit_logits,
                                                        label_ids,
                                                        label_attention_mask)
                    post_perp_loss = compute_perplexity_t5(tokenizer,
                                                        post_edit_logits,
                                                        label_ids,
                                                        label_attention_mask)

                    pre_edit_logits = None
                    post_edit_logits = None

                    results_specificity = None
                    if train_params['COMPUTE_SPECIFICITY']:
                        results_specificity = []
                        assert len(specificity_batches) == len(pre_loc_logits) \
                            == len(post_loc_logits)
                        for k in range(len(specificity_batches)):
                            s_batch = specificity_batches[k]
                            s_pre_perp_loss = compute_perplexity_t5(
                                tokenizer,
                                pre_loc_logits[k],
                                s_batch["edit_inner"][0]['labels'][
                                    'input_ids'],
                                s_batch["edit_inner"][0]['labels'][
                                    'attention_mask']
                            )

                            s_post_perp_loss = compute_perplexity_t5(
                                tokenizer,
                                post_loc_logits[k],
                                s_batch["edit_inner"][0]['labels'][
                                    'input_ids'],
                                s_batch["edit_inner"][0]['labels'][
                                    'attention_mask']
                            )

                            results_specificity.append(
                                {'pre': s_pre_perp_loss[0],
                                'post': s_post_perp_loss[0]})

                    pre_loc_logits = None
                    post_loc_logits = None


                else:
                    raise NotImplementedError
                

                output['pre'] = pre_perp_loss[0]
                output['post'] = post_perp_loss[0]
                output['specificity'] = results_specificity


                all_outputs.append(output)
    return all_outputs


def group_examples(data):
    data_d = defaultdict(list)
    for ex in data:
        ent_id = '_'.join(ex['ex_id'].split('_')[:2])
        data_d[ent_id].append(ex)
    return list(data_d.items())


def run_experiment(ki_method,
                   ft_model_name,
                   dataset_name,
                   data_files,
                   device,
                   train_params,
                   device2=None, 
                   random_def=None,
                   oracle_ft=False):

    outputs_d = {}
    for data_file in data_files:
        data = load_json(data_file)
        print(data_file, len(data))

        if dataset_name == 'ecbd':
            specificity_data = None
            witheld_data = None
            specificity_data = load_json(SPECIFICITY_DATA_PATH)
            if 'gpt2' in train_params['BASE_MODEL']:
                data = [format_gpt2_data(ex) for ex in data]
                specificity_data = [
                    format_gpt2_data(ex) for ex in specificity_data]
            elif 'gpt' in train_params['BASE_MODEL']:
                data = [format_gpt_data(ex) for ex in data]
                specificity_data = [format_gpt_data(ex) for ex in specificity_data]
            elif 'llama' in train_params['BASE_MODEL']:
                data = [format_llama_data(ex) for ex in data]
                specificity_data = [format_llama_data(ex) for ex in specificity_data]

            # For ECBD, we group examples by entities and finetune only once per
            # entity. This is unnecessary for specificity data.
            data = group_examples(data)


            all_outputs = run_edit_ecbd(
                data,
                dataset_name,
                ki_method,
                device,
                train_params,
                device2,
                model_name=ft_model_name,
                random_def=random_def,
                oracle_ft=oracle_ft,
                specificity_data=specificity_data,
                witheld_data=witheld_data)

        else:  # Entity Inferences
            if 'gpt2' in train_params['BASE_MODEL']:
                data = [format_gpt2_data_entity_inferences(ex) for ex in data]
            elif 'gpt' in train_params['BASE_MODEL']:
                data = [format_gpt_data_entity_inferences(ex) for ex in data]
            all_outputs = run_edit_entity_inference(
                data,
                dataset_name,
                ki_method,
                device,
                train_params,
                model_name=ft_model_name,
                random_def=random_def)

        # Aggregate results
        outputs_d[data_file] = all_outputs

        torch.cuda.empty_cache()

    return outputs_d
