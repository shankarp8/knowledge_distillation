import copy
# import evaluate
import json
import random
import torch
import numpy as np
import pandas as pd
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


CATEGORY_MAP = {
    'tv_show': 'TV show',
    'fake_person': 'person',
    'disaster': 'disaster'
}

MEND_DIR = '/data/shankar/knowledge_injection_dev-master-8/src/mend'
MEND_MODEL_DIR = '/data/shankar/knowledge_injection_dev-master-8/src/mend/trained_models'
SPECIFICITY_DATA_PATH = '/data/shankar/ping_knowledge_injection/data/ecbd/specificity_popular_20np_20random.json'



def load_json(path):
    with open(path) as f:
        return [json.loads(l.strip()) for l in f]


def dict_to(d, device):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.to(device)
        elif isinstance(v, dict):
            new_dict[k] = dict_to(v, device)
        else:
            new_dict[k] = v

    return new_dict


class CustomDataSetClass(Dataset):

    def __init__(
            self,
            data,
            tokenizer,
            input_len,
            target_len,
            input_text="input_text",
            target_text="target_text"
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.input_len = input_len
        self.label_len = target_len
        self.target_text = self.data[target_text]
        self.input_text = self.data[input_text]

    def __len__(self):
        return len(self.input_text)

    def __getitem__(self, index):
        input_text = str(self.input_text[index])
        target_text = str(self.target_text[index])

        input_text = ' '.join(input_text.split())
        target_text = ' '.join(target_text.split())

        input_ = self.tokenizer.batch_encode_plus(
            [input_text],
            max_length=self.input_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors='pt'
        )

        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.label_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors='pt'
        )

        input_ids = input_['input_ids'].squeeze()
        input_mask = input_['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'input_ids': input_ids.to(dtype=torch.long),
            'input_mask': input_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'labels' : target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


def to_tsr_t5_ecbd(tokenizer, ex, device, prepend_def=False, pseudo_input=None, teacher_eval=False, prepend_sent=False,
              random_def=None):
    '''This function supports a single example only (i.e., bsize=1).'''

    definition = [ex['definition']]
    def_label = [ex['def_target']]

    if random_def is not None:
        fake_def = random.choice(random_def)
        probe_sentences = [fake_def + ' ' + v['probe_sentence'] for _, v in
                           ex['probe_sentences'].items()]
    elif prepend_def and not prepend_sent:
        probe_sentences = [definition[0].replace('<extra_id_0>',
                                                 ex['def_target'][
                                                 13:-13]) + ' ' + v[
                               'probe_sentence'] for _, v in
                           ex['probe_sentences'].items()]
    elif prepend_sent and not prepend_def:
        probe_sentences = [ex['additional_sent'] + ' ' + v['probe_sentence'] for
                           _, v in ex['probe_sentences'].items()]
    elif teacher_eval:
        probe_sentences = [v['probe_sentence'] for _, v in ex['probe_sentences'].items()]
        probe_sentences = [pseudo_input + ' ' + ps for ps in probe_sentences]

    else:
        probe_sentences = [v['probe_sentence'] for _, v in
                           ex['probe_sentences'].items()]

    probe_labels = [v['label'] for _, v in ex['probe_sentences'].items()]

    unmasked_definition = [
        def_.replace('<extra_id_0>', lbl_[13:-13]) for def_, lbl_ in zip(
            definition, def_label)]
    unmasked_probe_sentence = [
        v['probe_sentence'].replace('<extra_id_0>', v['label'][13:-13]) for _,
                                                                        v in ex[
            'probe_sentences'].items()]


    definition_tok = tokenizer(definition, padding=True, return_tensors="pt")
    def_label_tok = tokenizer(def_label, padding=True, return_tensors="pt")
    probe_sentences_tok = [
        tokenizer(ps, padding=True, return_tensors="pt").to(device) for
        ps in probe_sentences]
    probe_labels_tok = [
        tokenizer(pl, padding=True, return_tensors="pt").to(device) for
        pl in probe_labels]

    edit_inner = [{'probe_sentence': ps} for ps in probe_sentences_tok]
    for i, ps in enumerate(edit_inner):
        ps['labels'] = probe_labels_tok[i]

    def_ = {**definition_tok}
    def_["labels"] = def_label_tok["input_ids"]
    def_["decoder_attention_mask"] = def_label_tok["attention_mask"]

    batch = {
        "edit_inner": edit_inner,  # Edit examples
        "definition": def_,  # Locality
        "cond": None,
        "labels": None,
    }

    return dict_to(batch, device)


def to_tsr_gpt_ecbd(tokenizer, ex, device, prepend_def=False,
                    prepend_sent=False, random_def=None):
    '''This function supports a single example only (i.e., bsize=1).'''
    definition = [ex['definition']]
    # left_context = [ex['left_context']]
    # right_context = [ex['right_context']]

    if random_def is not None:
        fake_def = random.choice(random_def)
        probe_sentences = [fake_def + ' ' + v['probe_sentence'] for _, v in
                           ex['probe_sentences'].items()]
        left_context_ps = [fake_def + ' ' + v['left_context_ps'] for _, v
                           in ex['probe_sentences'].items()]
    elif prepend_def and not prepend_sent:
        probe_sentences = [definition[0] + ' ' + v['probe_sentence'] for _, v in
                           ex['probe_sentences'].items()]
        left_context_ps = [definition[0] + ' ' + v['left_context_ps'] for _, v
                           in ex['probe_sentences'].items()]
    elif prepend_sent and not prepend_def:
        probe_sentences = [ex['additional_sent'] + ' ' + v['probe_sentence'] for
                           _, v in ex['probe_sentences'].items()]
        left_context_ps = [ex['additional_sent'] + ' ' + v['left_context_ps']
                           for _, v in ex['probe_sentences'].items()]

    else:
        probe_sentences = [v['probe_sentence'] for _, v in
                           ex['probe_sentences'].items()]
        left_context_ps = [v['left_context_ps'] for _, v in
                           ex['probe_sentences'].items()]

    probe_labels = [v['gpt_labels'] for _, v in ex['probe_sentences'].items()]
    right_context_ps = [v['right_context_ps'] for _, v in
                        ex['probe_sentences'].items()]

    cleaned_probe_sentences = [ps.strip(' <|endoftext|>') for ps in
                               probe_sentences]


    definition_tok = tokenizer(definition, padding=True, return_tensors="pt")
    def_label_tok = tokenizer(definition, padding=True, return_tensors="pt")
    # left_context_tok = tokenizer(left_context, padding=True,
    #                              return_tensors="pt")
    # right_context_tok = tokenizer(right_context, padding=True,
    #                               return_tensors="pt")
    probe_sentences_tok = [
        tokenizer(ps, padding=True, return_tensors="pt").to(device) for
        ps in probe_sentences]
    probe_labels_tok = [
        tokenizer(pl, padding=True, return_tensors="pt").to(device) for
        pl in probe_labels]
    left_context_ps_tok = [
        tokenizer(lc, padding=True, return_tensors="pt").to(device) for
        lc in left_context_ps]
    right_context_ps_tok = [
        tokenizer(rc, padding=True, return_tensors="pt").to(device) for
        rc in right_context_ps]



    edit_inner = [{'probe_sentence': ps} for ps in probe_sentences_tok]
    for i, ps in enumerate(edit_inner):
        ps['labels'] = probe_labels_tok[i]
        ps['left_context_ps'] = left_context_ps_tok[i]
        ps['right_context_ps'] = right_context_ps_tok[i]

    def_ = {**definition_tok}
    def_["labels"] = def_label_tok["input_ids"]

    batch = {
        "edit_inner": edit_inner,  # Edit examples
        "definition": def_,  # Locality
        "cond": None,
        "labels": None,
    }

    return dict_to(batch, device)

def to_tsr_llama_ecbd(tokenizer, ex, device=None, prepend_def=False,
                    prepend_sent=False, random_def=None, half=False):
    '''This function supports a single example only (i.e., bsize=1).'''

    definition = [ex['definition']]
    left_context = [ex['left_context']]
    right_context = [ex['right_context']]

    if random_def is not None:
        fake_def = random.choice(random_def)
        probe_sentences = [fake_def + ' ' + v['probe_sentence'] for _, v in
                           ex['probe_sentences'].items()]
        left_context_ps = [fake_def + ' ' + v['left_context_ps'] for _, v
                           in ex['probe_sentences'].items()]
    elif prepend_def and not prepend_sent:
        probe_sentences = [definition[0] + ' ' + v['probe_sentence'] for _, v in
                           ex['probe_sentences'].items()]
        left_context_ps = [definition[0] + ' ' + v['left_context_ps'] for _, v
                           in ex['probe_sentences'].items()]
    elif prepend_sent and not prepend_def:
        probe_sentences = [ex['additional_sent'] + ' ' + v['probe_sentence'] for
                           _, v in ex['probe_sentences'].items()]
        left_context_ps = [ex['additional_sent'] + ' ' + v['left_context_ps']
                           for _, v in ex['probe_sentences'].items()]

    else:
        probe_sentences = [v['probe_sentence'] for _, v in
                           ex['probe_sentences'].items()]
        left_context_ps = [v['left_context_ps'] for _, v in
                           ex['probe_sentences'].items()]


    probe_labels = [v['gpt_labels'] for _, v in ex['probe_sentences'].items()]
    right_context_ps = [v['right_context_ps'] for _, v in
                        ex['probe_sentences'].items()]


    definition_tok = tokenizer(definition, padding=True, return_tensors="pt")
    def_label_tok = tokenizer(definition, padding=True, return_tensors="pt")

    # probe_sentences_tok = [
    #     tokenizer(ps, padding=True, return_tensors="pt").to(device) for
    #     ps in probe_sentences]
    # probe_labels_tok = [
    #     tokenizer(pl, padding=True, return_tensors="pt").to(device) for
    #     pl in probe_labels]
    # left_context_ps_tok = [
    #     tokenizer(lc, padding=True, return_tensors="pt").to(device) for
    #     lc in left_context_ps]
    # right_context_ps_tok = [
    #     tokenizer(rc, padding=True, return_tensors="pt").to(device) for
    #     rc in right_context_ps]
    # device = torch.device('cpu')

    probe_sentences_tok = [
    tokenizer(ps, padding=True, return_tensors="pt") for
    ps in probe_sentences]
    probe_labels_tok = [
        tokenizer(pl, padding=True, return_tensors="pt") for
        pl in probe_labels]
    left_context_ps_tok = [
        tokenizer(lc, padding=True, return_tensors="pt") for
        lc in left_context_ps]
    right_context_ps_tok = [
        tokenizer(rc, padding=True, return_tensors="pt") for
        rc in right_context_ps]

    

    edit_inner = [{'probe_sentence': ps} for ps in probe_sentences_tok]
    for i, ps in enumerate(edit_inner):
        ps['labels'] = probe_labels_tok[i]
        ps['left_context_ps'] = left_context_ps_tok[i]
        ps['right_context_ps'] = right_context_ps_tok[i]




    def_ = {**definition_tok}
    def_["labels"] = def_label_tok["input_ids"]

    batch = {
        "edit_inner": edit_inner,  # Edit examples
        "definition": def_,  # Locality
        "cond": None,
        "labels": None,
    }

    return batch


def random_mask(sentence):
    masked_sentence = sentence.split()
    n = len(masked_sentence)
    span_len = random.choice([1, 2, 3])
    span_start = random.choice(list(range(1, n)))  # exclude <ENT_NAME>
    span_str = ' '.join(masked_sentence[span_start: span_start + span_len])
    target = '<extra_id_0> ' + span_str + ' <extra_id_1>'
    masked_sentence = ' '.join(masked_sentence).replace(' ' + span_str,
                                                        ' <extra_id_0>', 1)
    return masked_sentence, target


def particle_mask(sentence):
    particles = ['a', 'an', 'the', 'is', 'can']
    masked_sentence = sentence.split()
    span_str = None
    for particle in particles:
        for i, word in enumerate(masked_sentence):
            if particle == word:
                span_str = word
                break
        if span_str:
            break
    assert span_str
    target = '<extra_id_0> ' + span_str + ' <extra_id_1>'
    masked_sentence = ' '.join(masked_sentence).replace(' ' + span_str,
                                                        ' <extra_id_0>', 1)
    return masked_sentence, target


def nll_mask(tokenizer, sentence, ent_str, model, device, topk=5):
    masked_sentence = sentence.split()
    ent_str = ent_str.split()
    ent_len = len(ent_str)
    # Find ent location
    ent_start, ent_end = 0, 0
    for i in range(len(masked_sentence)):
        if masked_sentence[i:i + ent_len] == ent_str:
            ent_start = i
            ent_end = i + ent_len
            break
    all_target = []
    for i, token in enumerate(masked_sentence):
        if i >= ent_end:
            input_ids = tokenizer(
                ' '.join(masked_sentence).replace(' ' + token, ' <extra_id_0>',
                                                  1),
                return_tensors="pt").input_ids.to(device,
                                                  dtype=torch.long)
            labels_ids = tokenizer('<extra_id_0> ' + token + ' <extra_id_1>',
                                   return_tensors="pt").input_ids.to(
                device, dtype=torch.long)
            lm_logits = model(input_ids=input_ids, labels=labels_ids).logits
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
            per_token_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)),
                                      labels_ids.view(-1)).view(1, -1)
            loss = per_token_loss[0, 1:-1].cpu().detach().numpy().mean()
            all_target.append((i, loss, token))

    all_target.sort(key=lambda x: x[1], reverse=True)
    masked_sentences, targets = [], []
    for j, loss, token in all_target[:topk]:
        assert masked_sentence[j] == token
        _masked_sentence = copy.deepcopy(masked_sentence)
        _masked_sentence[j] = '<extra_id_0>'
        masked_sentences.append(' '.join(_masked_sentence))
        targets.append('<extra_id_0> ' + token + ' <extra_id_1>')
    return masked_sentences, targets


def format_gpt_data(ex, pad_token='<|endoftext|>'):
    context = ex['definition'].split('<extra_id_0>')
    ex['original_def'] = ex['definition']
    assert len(context) == 2, context
    ex['left_context'] = context[0].strip()
    ex['right_context'] = context[1]
    ex['definition'] = ex['definition'].replace('<extra_id_0>', ex['def_target'][13:-13])
    for _, ps in ex['probe_sentences'].items():
        gpt_labels = []
        gpt_labels.append(ps['probe_sentence'].replace('<extra_id_0>', ps['label'][13:-13]) + ' ' + pad_token)
        ps_context = ps['probe_sentence'].split('<extra_id_0>')
        assert len(ps_context) == 2, ps_context
        ps['left_context_ps'] = ps_context[0].strip() + ' ' + pad_token
        ps['right_context_ps'] = ps_context[1] + ' ' + pad_token
        ps['original_ps'] = ps['probe_sentence']
        ps['probe_sentence'] = ps['probe_sentence'].replace('<extra_id_0>', ps['label'][13:-13]) + ' ' + pad_token
        ps['gpt_labels'] = gpt_labels
        ps['answer_str'] = ps['label'][13:-13]
    return ex

def format_llama_data(ex, pad_token='</s>'):
    context = ex['definition'].split('<extra_id_0>')
    ex['original_def'] = ex['definition']
    assert len(context) == 2, context
    ex['left_context'] = context[0].strip()
    ex['right_context'] = context[1]
    ex['definition'] = ex['definition'].replace('<extra_id_0>', ex['def_target'][13:-13])
    for _, ps in ex['probe_sentences'].items():
        llama_labels = []
        llama_labels.append(ps['probe_sentence'].replace('<extra_id_0>', ps['label'][13:-13]) + pad_token)
        ps_context = ps['probe_sentence'].split('<extra_id_0>')
        assert len(ps_context) == 2, ps_context
        ps['left_context_ps'] = ps_context[0].strip() + pad_token
        ps['right_context_ps'] = ps_context[1] + pad_token
        ps['original_ps'] = ps['probe_sentence']
        ps['probe_sentence'] = ps['probe_sentence'].replace('<extra_id_0>', ps['label'][13:-13]) + pad_token
        ps['gpt_labels'] = llama_labels
        ps['answer_str'] = ps['label'][13:-13]
    return ex

def format_gpt2_data(ex, pad_token='<|endoftext|>'):
    context = ex['definition'].split('<extra_id_0>')
    ex['original_def'] = ex['definition']
    assert len(context) == 2, context
    ex['left_context'] = context[0].strip()
    ex['right_context'] = context[1]
    ex['definition'] = ex['definition'].replace('<extra_id_0>', ex['def_target'][13:-13])
    for _, ps in ex['probe_sentences'].items():
        gpt_labels = []
        gpt_labels.append(ps['probe_sentence'].replace('<extra_id_0>', ps['label'][13:-13]) + pad_token)
        ps_context = ps['probe_sentence'].split('<extra_id_0>')
        assert len(ps_context) == 2, ps_context
        ps['left_context_ps'] = ps_context[0].strip() + pad_token
        ps['right_context_ps'] = ps_context[1] + pad_token
        ps['original_ps'] = ps['probe_sentence']
        ps['probe_sentence'] = ps['probe_sentence'].replace('<extra_id_0>', ps['label'][13:-13]) + pad_token
        ps['gpt_labels'] = gpt_labels
        ps['answer_str'] = ps['label'][13:-13]
    return ex


def format_gpt_data_entity_inferences(ex, pad_token='<|endoftext|>'):
    context = ex['definition'].split('<extra_id_0>')
    ex['original_def'] = ex['definition']
    assert len(context) == 2, context
    ex['left_context'] = context[0].strip()
    ex['right_context'] = context[1]
    ex['definition'] = ex['definition'].replace('<extra_id_0>', ex['def_target'][13:-13])
    label = ex['label']
    for _, ps in ex['probe_sentences'].items():
        gpt_labels = []
        for option in ps['labels']:
            gpt_labels.append(ps['probe_sentence'].replace(
                '<extra_id_0>', option[13:-13]) + ' ' + pad_token)
        ps_context = ps['probe_sentence'].split('<extra_id_0>')
        assert len(ps_context) == 2, ps_context
        ps['left_context_ps'] = ps_context[0].strip() + ' ' + pad_token
        ps['right_context_ps'] = ps_context[1] + ' ' + pad_token
        ps['original_ps'] = ps['probe_sentence']
        ps['probe_sentence'] = ps['probe_sentence'].replace('<extra_id_0>', label) + ' ' + pad_token
        ps['gpt_labels'] = gpt_labels
        ps['answer_str'] = label
    return ex

def format_gpt2_data_entity_inferences(ex, pad_token='<|endoftext|>'):
    context = ex['definition'].split('<extra_id_0>')
    ex['original_def'] = ex['definition']
    assert len(context) == 2, context
    ex['left_context'] = context[0].strip()
    ex['right_context'] = context[1]
    ex['definition'] = ex['definition'].replace('<extra_id_0>', ex['def_target'][13:-13])
    label = ex['label']
    for _, ps in ex['probe_sentences'].items():
        gpt_labels = []
        for option in ps['labels']:
            gpt_labels.append(ps['probe_sentence'].replace(
                '<extra_id_0>', option[13:-13]) + pad_token)
        ps_context = ps['probe_sentence'].split('<extra_id_0>')
        assert len(ps_context) == 2, ps_context
        ps['left_context_ps'] = ps_context[0].strip() + pad_token
        ps['right_context_ps'] = ps_context[1] + pad_token
        ps['original_ps'] = ps['probe_sentence']
        ps['probe_sentence'] = ps['probe_sentence'].replace('<extra_id_0>', label) + pad_token
        ps['gpt_labels'] = gpt_labels
        ps['answer_str'] = label
    return ex


def write_results_ecbd(result_dict, data_file, write_to, model_name):

    data = load_json(data_file)

    deltas = []

    with open(write_to, 'w') as f:
        for i, ex in enumerate(data):
            results = result_dict[ex['ex_id']]
            pre_edits = results['pre']
            post_edits = results['post']
            def_results = True
            generations = True
            try:
                pre_def_edits = results['pre_def']
                post_def_edits = results['post_def']
            except:
                def_results = False
            
            try:
                pre_edit_gen = results['pre_edit_gen']
                post_edit_gen = results['post_edit_gen']
            except:
                generations = False
            # sim_scores = results['sim_scores']

            f.write(f'----[{i}]' + '-' * 100 + '\n\n')
            f.write('ex_id         : {}\n'.format(ex['ex_id']))
            f.write('attribute     : {}\n'.format(ex['attribute']))
            if 'gpt' in model_name:
                f.write('definition    : {}\n\n'.format(
                    ex['definition'].replace('<extra_id_0>',
                                             ex['def_target'][13:-13])))
            else:
                f.write('definition    : {}\n\n'.format(ex['definition']))
                f.write(
                    'def target    : {}\n\n'.format(ex['def_target'][13:-13]))
            f.write('probe sentence: {}\n\n'.format(
                ex['probe_sentences']['template_0']['probe_sentence']))
            f.write('gold span     : {}\n\n'.format(
                ex['probe_sentences']['template_0']['label'][13:-13]))
            f.write('perplexity\n')
            f.write('pre-perp      : {:.2f}\n'.format(pre_edits[0]))
            f.write('per-token nll : {}\n\n'.format('  '.join(
                ['{} ({:.2f})'.format(token, nll) for token, nll in
                 pre_edits[1]])))
            f.write('post-perp     : {:.2f}\n'.format(post_edits[0]))
            f.write('per-token nll : {}\n\n'.format('  '.join(
                ['{} ({:.2f})'.format(token, nll) for token, nll in
                 post_edits[1]])))
            diff = post_edits[0] - pre_edits[0]
            
            f.write('Delta in perp : {:.2f}\n\n'.format(diff))
            if generations:
                f.write('Pre-edit Generation    : {}\n\n'.format(pre_edit_gen))
                f.write('Post-edit Generation    : {}\n\n'.format(post_edit_gen))
            
            f.write('augmented probes\n')
            for probe in ex['augmented_probes']:
                f.write(probe)
                f.write('\n')
            
            if def_results:
                f.write('perplexity\n')
                f.write('pre-perp on definition     : {:.2f}\n'.format(pre_def_edits[0]))
                f.write('per-token nll on definition: {}\n\n'.format('  '.join(
                    ['{} ({:.2f})'.format(token, nll) for token, nll in
                    pre_def_edits[1]])))
                f.write('post-perp on definition    : {:.2f}\n'.format(post_def_edits[0]))
                f.write('per-token nll on definition : {}\n\n'.format('  '.join(
                    ['{} ({:.2f})'.format(token, nll) for token, nll in
                    post_def_edits[1]])))
                diff_def = post_def_edits[0] - pre_def_edits[0]
                f.write('Delta in perp on definition : {:.2f}\n\n'.format(diff_def))

            if 'specificity_pre_acc' in results and \
                    'specificity_post_acc' in results:
                f.write('specificity  :\n')
                f.write('         pre = {:.4f}\n'.format(
                    results['specificity_pre_acc']))
                f.write('        post = {:.4f}\n\n'.format(
                    results['specificity_post_acc']))

            deltas.append(diff)

    fig, ax = plt.subplots()
    pd.DataFrame(deltas).hist(bins=100, range=(-10, 10), ax=ax)
    fig.savefig(write_to.strip('.txt') + '_diff.png')


def write_results_entity_inferences(result_dict, data_file, write_to,
                                    model_name):

    data = load_json(data_file)

    deltas = []

    with open(write_to, 'w') as f:
        for i, ex in enumerate(data):

            results = result_dict[ex['ex_id']]
            probs = results['probs']

            scores, label = probs

            # pre
            pre_probs = [s[-2] for s in scores]
            pre_id = np.argmax(pre_probs)
            pre_pred = scores[pre_id][0]

            # post
            post_probs = [s[-1] for s in scores]
            post_id = np.argmax(post_probs)
            post_pred = scores[post_id][0]

            pre_is_correct = 'Correct' if pre_pred == label else 'Wrong'
            post_is_correct = 'Correct' if post_pred == label else 'Wrong'

            f.write(f'----[{i}]' + '-' * 100 + '\n\n')
            f.write('ex_id         : {}\n'.format(ex['ex_id']))
            f.write('attribute     : {}\n'.format(ex['attribute']))
            if 'gpt' in model_name:
                f.write('definition    : {}\n\n'.format(
                    ex['definition'].replace('<extra_id_0>',
                                             ex['def_target'][13:-13])))
            else:
                f.write('definition    : {}\n\n'.format(ex['definition']))
                f.write(
                    'def target    : {}\n\n'.format(ex['def_target'][13:-13]))
            f.write('probe sentence: {}\n\n'.format(
                ex['probe_sentences']['template_0']['probe_sentence']))
            f.write('gold span     : {} ({})\n\n'.format(ex['label'], label))
            f.write('perplexity\n')
            f.write('pre-pred      : {} -- {}\n'.format(pre_pred,
                                                        pre_is_correct))
            f.write('post-pred     : {} -- {}\n\n'.format(post_pred,
                                                          post_is_correct))

            f.write('pre-dist      : {} (sum = {:.2f})\n'.format(
                ', '.join(['{}: {:.4f}'.format(s[0], s[-2]) for s in scores]),
                sum([s[-2] for s in scores])))
            f.write('post-dist     : {} (sum = {:.2f})\n\n'.format(
                ', '.join(['{}: {:.4f}'.format(s[0], s[-1]) for s in scores]),
                sum([s[-1] for s in scores])))

            if 'specificity_pre_acc' in results and \
                    'specificity_post_acc' in results:
                f.write('specificity  :\n')
                f.write('         pre = {:.4f}\n'.format(
                    results['specificity_pre_acc']))
                f.write('        post = {:.4f}\n\n'.format(
                    results['specificity_post_acc']))



def to_tsr_t5_entity_inference(tokenizer, ex, device, prepend_def=False, teacher_eval=False, pseudo_input=None,  
                               prepend_sent=False, random_def=False):
    '''This function supports a single example only (i.e., bsize=1).'''

    definition = [ex['definition']]

    def_label = [ex['def_target']]

    probe_sentences = [v['probe_sentence'] for _, v in
                       ex['probe_sentences'].items()]

    if prepend_sent:
        additional_sentence = ' '.join(
            [sent.replace('<ENT_NAME>', ex['ent_str']) for sent in
             ex['additional_sentences']])
        x_is_y = '{} is a {}.'.format(ex['ent_str'], ex['category'])

        additional_sentence = additional_sentence + ' ' + x_is_y
        # additional_sentence = additional_sentence
        # additional_sentence = x_is_y

        probe_sentences = [additional_sentence + ' ' + ps for ps in
                           probe_sentences]

    if prepend_def:
        probe_sentences = [
            definition[0].replace('<extra_id_0>',
                                  ex['def_target'][13:-13]) + '. ' + ps for ps in
            probe_sentences]
    if teacher_eval:
        probe_sentences = [pseudo_input + ' ' + ps for ps in probe_sentences]




    probe_labels = [v['labels'] for _, v in ex['probe_sentences'].items()]

    unmasked_definition = [
        def_.replace('<extra_id_0>', lbl_[13:-13]) for def_, lbl_ in zip(
            definition, def_label)]
    unmasked_probe_sentence = [
        v['probe_sentence'].replace('<extra_id_0>', ex['label'])
        for _, v in ex['probe_sentences'].items()]

    definition_tok = tokenizer(definition, padding=True, return_tensors="pt")
    def_label_tok = tokenizer(def_label, padding=True, return_tensors="pt")
    probe_sentences_tok = [
        tokenizer(ps, padding=True, return_tensors="pt").to(device) for
        ps in probe_sentences]
    probe_labels_tok = [
        tokenizer(pl, padding=True, return_tensors="pt").to(device) for
        pl in probe_labels]

    edit_inner = [{'probe_sentence': ps} for ps in probe_sentences_tok]
    for i, ps in enumerate(edit_inner):
        ps['labels'] = probe_labels_tok[i]

    def_ = {**definition_tok}
    def_["labels"] = def_label_tok["input_ids"]
    def_["decoder_attention_mask"] = def_label_tok["attention_mask"]

    batch = {
        "edit_inner": edit_inner,  # Edit examples
        "definition": def_,  # Locality
        "cond": None,
        "labels": None,
    }

    return dict_to(batch, device)


def to_tsr_gpt_entity_inference(tokenizer, ex, device, prepend_def=False,
                                prepend_sent=False, random_def=False):
    '''This function supports a single example only (i.e., bsize=1).'''

    definition = [ex['definition']]
    left_context = [ex['left_context']]
    right_context = [ex['right_context']]

    if prepend_def and not prepend_sent:
        probe_sentences = [definition[0] + ' ' + v['probe_sentence'] for _, v in
                           ex['probe_sentences'].items()]
        left_context_ps = [definition[0] + ' ' + v['left_context_ps'] for _, v
                           in ex['probe_sentences'].items()]
    elif prepend_sent and not prepend_def:
        probe_sentences = [ex['additional_sent'] + ' ' + v['probe_sentence'] for
                           _, v in ex['probe_sentences'].items()]
        left_context_ps = [ex['additional_sent'] + ' ' + v['left_context_ps']
                           for _, v in ex['probe_sentences'].items()]
    else:
        probe_sentences = [v['probe_sentence'] for _, v in
                           ex['probe_sentences'].items()]
        left_context_ps = [v['left_context_ps'] for _, v in
                           ex['probe_sentences'].items()]

    probe_labels = [v['gpt_labels'] for _, v in ex['probe_sentences'].items()]

    right_context_ps = [v['right_context_ps'] for _, v in
                        ex['probe_sentences'].items()]

    cleaned_probe_sentences = [ps.strip(' <|endoftext|>') for ps in
                               probe_sentences]


    definition_tok = tokenizer(definition, padding=True, return_tensors="pt")
    def_label_tok = tokenizer(definition, padding=True, return_tensors="pt")
    # left_context_tok = tokenizer(left_context, padding=True,
    #                              return_tensors="pt")
    # right_context_tok = tokenizer(right_context, padding=True,
    #                               return_tensors="pt")
    probe_sentences_tok = [
        tokenizer(ps, padding=True, return_tensors="pt").to(device) for
        ps in probe_sentences]
    probe_labels_tok = [
        tokenizer(pl, padding=True, return_tensors="pt").to(device) for
        pl in probe_labels]
    left_context_ps_tok = [
        tokenizer(lc, padding=True, return_tensors="pt").to(device) for
        lc in left_context_ps]
    right_context_ps_tok = [
        tokenizer(rc, padding=True, return_tensors="pt").to(device) for
        rc in right_context_ps]

    edit_inner = [{'probe_sentence': ps} for ps in probe_sentences_tok]
    for i, ps in enumerate(edit_inner):
        ps['labels'] = probe_labels_tok[i]
        ps['left_context_ps'] = left_context_ps_tok[i]
        ps['right_context_ps'] = right_context_ps_tok[i]

    def_ = {**definition_tok}
    def_["labels"] = def_label_tok["input_ids"]

    batch = {
        "edit_inner": edit_inner,  # Edit examples
        "definition": def_,  # Locality
        "cond": None,
        "labels": None,
    }

    return dict_to(batch, device)
