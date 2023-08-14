import os
import torch
import sys
import pickle
from scipy import stats

sys.path.append('/data/shankar/ping_knowledge_injection')
from transformers import set_seed
from src import metrics
from src import run_edit
from src import data_utils

ROOT_DIR = '/data/shankar/ping_knowledge_injection'

def main():

    # Choose from 'ft_per_ex', 'prepend_def', 'ft_distill_multiple (our distillation procedure)'
    ki_method = 'ft_distill_multiple'

    # Pretrained model. Use this together with 'ft'.
    ft_model_name = None

    # Choose a unique experiment name
    exp_name = 'ecbd/gpt/ft_distill_multiple'

    exp_dir = os.path.join(ROOT_DIR, 'output', exp_name)

    os.makedirs(exp_dir, exist_ok=True)

    data_dir = os.path.join(ROOT_DIR, 'data')

    data_files = [
        os.path.join(data_dir, 'ecbd/ecbd_2022_1k_augmented.json')
    ]

    train_params = {
        "EXP_NAME": exp_name,
        "EXP_DIR": exp_dir,
        "KI_METHOD": ki_method,
        "BASE_MODEL": 'gpt-neo-1.3B',  # model_type: gpt-neo-1.3B or gpt2-XL
        "TRAIN_BATCH_SIZE": 1,  # training batch size
        "VALID_BATCH_SIZE": 1,  # validation batch size
        "TRAIN_EPOCHS": 5,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": 3e-6,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 128,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 128,  # max length of target text
        "SEED": 2022,  # set seed for reproducibility
        "AUGMENT": False, # set to True for augmented fine-tuning 
        "NUM_AUGMENT": 1, # number of transfer sentences to fine-tune on
        "X_IS_Y": False,
        "MASKING_STRATEGY": 'random',
        "USE_NLL_MASKS": False,
        "TOPK": 1,
        "MEMORY_RETRIEVAL": False,
        "TRAIN_ON_PROBE": False,
        "COMPUTE_SPECIFICITY": True, #set to False if using prepend_def method 
        "DEVICE":"cuda:2",
        "DEVICE2":"cuda:2",
        "TEACHER_MODEL":'EleutherAI/gpt-neo-1.3B', #teacher model, same as base model
        "SAMPLE_TEMPERATURE":0.9,
        "SOFTMAX_TEMP":2.0,
        "FT_LAST_LAYER":False, #for fine-tuning, if set True then only the last layer of the model is edited
        "NUM_EDITS":1, # number of entities to edit information for at once
        "NUM_PROBES":5, # number of transfer sentences per entity
        "NUM_UPDATES":5, # number of gradient updates per transfer sentence
        "AFTER_ENT_SPAN":False # distill on logits after entity span in transfer sentence or distill on all logits
    }

    # Print params
    for k, v in train_params.items():
        print('{:>24}: {}'.format(k, v))

    device = torch.device(train_params["DEVICE"])
    device2 = torch.device(train_params['DEVICE2'])
    set_seed(train_params['SEED'])

    results_dict = run_edit.run_experiment(ki_method,
                                           ft_model_name,
                                           'ecbd',
                                           data_files,
                                           device,
                                           train_params,
                                           device2,
                                           random_def=None,
                                           oracle_ft=False)

    with open(os.path.join(exp_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results_dict, f)

    all_pre_results = []
    all_post_results = []
    all_pre_def_results = []
    all_post_def_results = []
    all_specificity_pre_res = []
    all_specificity_post_res = []
    for data_name, results in results_dict.items():
        pre_results = [res['pre'] for res in results]
        post_results = [res['post'] for res in results]
        pre_perp = metrics.compute_total_perplexity(pre_results)
        post_perp = metrics.compute_total_perplexity(post_results)

        all_pre_results.extend(pre_results)
        all_post_results.extend(post_results)


        diff = [
            post[0] - pre[0] for pre, post in zip(pre_results, post_results)]


        print('{}:\nPerplexity: Pre = {:.4f}, Post = {:.4f}'.format(
            data_name, pre_perp, post_perp))

        if train_params['COMPUTE_SPECIFICITY']:

            specificity_pre = [[r['pre'] for r in res['specificity']] for res
                               in results]
            specificity_post = [[r['post'] for r in res['specificity']] for res
                               in results]
            specificity_pre_perp = [metrics.compute_total_perplexity(s) for s
                                    in specificity_pre]
            specificity_post_perp =  [metrics.compute_total_perplexity(s) for s
                                    in specificity_post]
            all_specificity_pre_res.extend(specificity_pre)
            all_specificity_post_res.extend(specificity_post)
            for idx, o in enumerate(results):
                o['specificity_pre_acc'] = specificity_pre_perp[idx]
                o['specificity_post_acc'] = specificity_post_perp[idx]

        result_dict = {res['ex_id']: res for res in results}

        data_utils.write_results_ecbd(
            result_dict,
            data_name,
            os.path.join(
                exp_dir, data_name.split('/')[-1].rstrip('.json') + '.txt'),
            train_params['BASE_MODEL'])

    total_pre_perplexity = metrics.compute_total_perplexity(all_pre_results)
    total_post_perplexity = metrics.compute_total_perplexity(all_post_results)


    print('Total Perplexity: Pre = {:.4f}, Post = {:.4f}'.format(
        total_pre_perplexity, total_post_perplexity))
    

    total_specificity_pre_perplexity = metrics.compute_total_perplexity(
        [y for x in all_specificity_pre_res for y in x])
    total_specificity_post_perplexity = metrics.compute_total_perplexity(
        [y for x in all_specificity_post_res for y in x])

    print('Total Specificity: Pre {:.4f}, Post = {:.4f}'.format(
        total_specificity_pre_perplexity, total_specificity_post_perplexity))

    all_diff = [post[0] - pre[0] for pre, post in zip(all_pre_results,
                                                      all_post_results)]



if __name__ == '__main__':
    main()
