import numpy as np
import pandas as pd
import torch

from torch.nn import CrossEntropyLoss


def compute_dist_over_labels_t5(tokenizer, edit_dict, labels_str, labels_tsr):
    n_labels = len(edit_dict)
    assert labels_tsr['input_ids'].size(0) == labels_tsr['attention_mask'].size(
        0) == n_labels

    labels = []
    lls = []
    for i in range(n_labels):
        last_idx = labels_tsr['attention_mask'][
                       i].sum().item() - 2  # Ignore special tokens

        ll = edit_dict[i]['log_prob_all'][0, 1:last_idx]
        ll = ll.sum().item()
        lls.append(ll)
        label = labels_str[i][13:-13]
        labels.append(label)

    return labels, lognormalize(np.array(lls)), lls


def compute_perplexity_t5(tokenizer, logits, label_ids, label_attention_mask):
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_fct(logits.view(-1, logits.size(-1)), label_ids.view(-1))

    batch_size = logits.shape[0]
    perp_loss = []
    for i, l in enumerate(loss.view(batch_size, -1)):
        # Remove </s>, <pad>
        n_tokens = label_attention_mask[i].sum() - 2
        # Exclude <extra_id_0>, <extra_id_1>
        perplexity = torch.exp(
            (l * label_attention_mask[i])[1:n_tokens].mean()).item()
        loss_per_token = list(
            zip(tokenizer.convert_ids_to_tokens(
                label_ids[i].cpu().detach().numpy().tolist())[1:n_tokens],
                [float(s) for s in l.cpu().detach().numpy()[1:n_tokens]]
                )
        )
        perp_loss.append((perplexity, loss_per_token))
    return perp_loss


def compute_total_perplexity(all_outputs):
    nll_loss = []
    for output in all_outputs:
        nll_loss.append(np.mean([x[1] for x in output[1]]))
    perplexity = np.exp(np.mean(nll_loss))
    return perplexity


def compute_dist_over_labels_gpt(tokenizer, edit_dict, labels_str, labels_tsr,
                                 left_context_tsr, right_context_tsr):
    n_labels = len(edit_dict)
    assert labels_tsr['input_ids'].size(0) == labels_tsr['attention_mask'].size(
        0) == n_labels

    labels = []
    lls = []
    for i in range(n_labels):
        total_len = \
        (labels_tsr['input_ids'][i] == tokenizer.eos_token_id).nonzero(
            as_tuple=True)[0][0]
        # left and right contexts are the same for all labels
        left_len = \
        (left_context_tsr['input_ids'][0] == tokenizer.eos_token_id).nonzero(
            as_tuple=True)[0]
        right_len = \
        (right_context_tsr['input_ids'][0] == tokenizer.eos_token_id).nonzero(
            as_tuple=True)[0]
        start_loc = left_len
        span_len = total_len - left_len - right_len

        end_loc = start_loc + span_len
        ll = edit_dict[i]['log_prob_all'][0, start_loc-1:end_loc-1]
        ll = ll.sum().item()
        lls.append(ll)
        label = labels_str[i][13:-13]
        labels.append(label)

    return labels, lognormalize(np.array(lls)), lls
def compute_perplexity_over_sentence(model_raw, tokenizer, sentence, device):
    tokenizer.bos_token = tokenizer.eos_token
    device = model_raw.device
    input_ids = tokenizer(tokenizer.bos_token+' '+sentence, return_tensors='pt').input_ids.to(device)
    logits = model_raw(input_ids).logits
    label_ids = input_ids[:, 1:].contiguous()
    label_attention_mask = (input_ids != tokenizer.pad_token_id)[:, 1:].contiguous()
    
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
    shift_logits = logits[..., :-1, :].contiguous()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), label_ids.view(-1))
    
    # loss = loss.view(label_ids.shape)

    masked_loss = loss * label_attention_mask
    perplexity = torch.exp(masked_loss.sum() / label_attention_mask.sum()).item()
    
    token_strings = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    loss_per_token = list(zip(token_strings[1:], [float(l) for l in masked_loss.view(-1)]))

    return [(perplexity, loss_per_token)]

def compute_perplexity_bloom(tokenizer, logits, label_ids, label_attention_mask,
                           labels_tsr, left_context_tsr, right_context_tsr, full_probe=False, pseudo=None, label=None, pre=False):
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = label_ids[..., 1:].contiguous()
    shift_label_attention_mask = label_attention_mask[..., 1:].contiguous()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))

    batch_size = logits.shape[0]
    perp_loss = []
    for i, l in enumerate(loss.view(batch_size, -1)):

        total_len = \
        (labels_tsr['input_ids'][i] == tokenizer.pad_token_id).nonzero(
            as_tuple=True)[0]
        # left and right contexts are the same for all labels
        left_len = \
        (left_context_tsr['input_ids'][0] == tokenizer.pad_token_id).nonzero(
            as_tuple=True)[0]
        right_len = \
        (right_context_tsr['input_ids'][0] == tokenizer.pad_token_id).nonzero(
            as_tuple=True)[0]
        start_loc = left_len
        span_len = total_len - left_len - right_len

        end_loc = start_loc + span_len

        perplexity = torch.exp(
            (l * shift_label_attention_mask[i])[
            start_loc-1:end_loc-1].mean()).item()
        



        loss_per_token = list(
            zip(tokenizer.convert_ids_to_tokens(
                label_ids[i].cpu().detach().numpy().tolist())[
                start_loc:end_loc],
                [float(s) for s in l.cpu().detach().numpy()[
                                   start_loc-1:end_loc-1]]  # Shift back by 1
                )
        )


        if not loss_per_token:
            print(total_len, left_len, right_len, start_loc, end_loc)
            print(tokenizer.convert_ids_to_tokens(labels_tsr['input_ids'][i]))
            print(tokenizer.convert_ids_to_tokens(
                labels_tsr['input_ids'][i, start_loc:end_loc]))
            print(tokenizer.convert_ids_to_tokens(
                left_context_tsr['input_ids'][0]))
            print(tokenizer.convert_ids_to_tokens(
                right_context_tsr['input_ids'][0]))
            print()
            return False

        perp_loss.append((perplexity, loss_per_token))
    return perp_loss


def compute_perplexity_gpt(tokenizer, logits, label_ids, label_attention_mask,
                           labels_tsr, left_context_tsr, right_context_tsr, full_probe=False, pseudo=None, label=None, pre=False):
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = label_ids[..., 1:].contiguous()
    shift_label_attention_mask = label_attention_mask[..., 1:].contiguous()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))

    batch_size = logits.shape[0]
    perp_loss = []
    for i, l in enumerate(loss.view(batch_size, -1)):

        total_len = \
        (labels_tsr['input_ids'][i] == tokenizer.eos_token_id).nonzero(
            as_tuple=True)[0]
        # left and right contexts are the same for all labels
        left_len = \
        (left_context_tsr['input_ids'][0] == tokenizer.eos_token_id).nonzero(
            as_tuple=True)[0]
        right_len = \
        (right_context_tsr['input_ids'][0] == tokenizer.eos_token_id).nonzero(
            as_tuple=True)[0]
        start_loc = left_len
        span_len = total_len - left_len - right_len

        end_loc = start_loc + span_len



        perplexity = torch.exp(
            (l * shift_label_attention_mask[i])[
            start_loc-1:end_loc-1].mean()).item()
        


        loss_per_token = list(
            zip(tokenizer.convert_ids_to_tokens(
                label_ids[i].cpu().detach().numpy().tolist())[
                start_loc:end_loc],
                [float(s) for s in l.cpu().detach().numpy()[
                                   start_loc-1:end_loc-1]]  # Shift back by 1
                )
        )


        if not loss_per_token:
            print(total_len, left_len, right_len, start_loc, end_loc)
            print(tokenizer.convert_ids_to_tokens(labels_tsr['input_ids'][i]))
            print(tokenizer.convert_ids_to_tokens(
                labels_tsr['input_ids'][i, start_loc:end_loc]))
            print(tokenizer.convert_ids_to_tokens(
                left_context_tsr['input_ids'][0]))
            print(tokenizer.convert_ids_to_tokens(
                right_context_tsr['input_ids'][0]))
            print()
            return False
        else:
            print('successful!')

        perp_loss.append((perplexity, loss_per_token))
    return perp_loss

def compute_perplexity_llama(tokenizer, logits, label_ids, label_attention_mask,
                           labels_tsr, left_context_tsr, right_context_tsr, full_probe=False, pseudo=None, label=None, pre=False):
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = label_ids[..., 1:].contiguous()
    shift_label_attention_mask = label_attention_mask[..., 1:].contiguous()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))
    
    batch_size = logits.shape[0]
    perp_loss = []
    for i, l in enumerate(loss.view(batch_size, -1)):

        total_len = \
        (labels_tsr['input_ids'][0] == tokenizer.eos_token_id).nonzero(
            as_tuple=True)[0]
        left_len = \
        (left_context_tsr['input_ids'][0] == tokenizer.eos_token_id).nonzero(
            as_tuple=True)[0]
        right_len = \
        (right_context_tsr['input_ids'][0] == tokenizer.eos_token_id).nonzero(
            as_tuple=True)[0]
    
        start_loc = left_len
        span_len = total_len - left_len - right_len

        end_loc = start_loc + span_len
        end_loc+=2

        perplexity = torch.exp(
            (l * shift_label_attention_mask[i])[
            start_loc-1:end_loc-1].mean()).item()
        
        loss_per_token = list(
            zip(tokenizer.convert_ids_to_tokens(
                label_ids[i].cpu().detach().numpy().tolist())[
                start_loc:end_loc],
                [float(s) for s in l.cpu().detach().numpy()[
                                   start_loc-1:end_loc-1]]  # Shift back by 1
                )
        )

        if not loss_per_token:
            print(total_len, left_len, right_len, start_loc, end_loc)
            print(tokenizer.convert_ids_to_tokens(labels_tsr['input_ids'][i]))
            print(tokenizer.convert_ids_to_tokens(
                labels_tsr['input_ids'][i, start_loc:end_loc]))
            print(tokenizer.convert_ids_to_tokens(
                left_context_tsr['input_ids'][0]))
            print(tokenizer.convert_ids_to_tokens(
                right_context_tsr['input_ids'][0]))

            print()
            print(loss_per_token)
            print()
        

        perp_loss.append((perplexity, loss_per_token))
    return perp_loss


def lognormalize(x):
    a = np.logaddexp.reduce(x)
    return np.exp(x - a)


def plot_dist(labels, pre_probs, post_probs):
    df = pd.DataFrame({'label': labels, 'pre': pre_probs, 'post': post_probs})
    ax = df.plot.bar(x='label', y=['pre', 'post'], rot=90,
                     figsize=(2 * np.sqrt(len(labels)), 5))


def aggregate_results(scores):
    pos_count_s = 0
    pos_count_p = 0
    delta_s = []
    delta_p = []
    odds_s = []
    odds_p = []
    for p in scores:
        if isinstance(p, list):
            _, s1, s2, p1, p2 = p[0]
            for p_ in p[1:]:
                s1 = np.logaddexp(s1, p_[1])
                s2 = np.logaddexp(s2, p_[2])
                p1 += p_[3]
                p2 += p_[4]
        else:
            _, s1, s2, p1, p2 = p

        if s2 > s1:
            pos_count_s += 1
        if p2 > p1:
            pos_count_p += 1
        delta_s.append(np.exp(s2) - np.exp(s1))
        delta_p.append(p2 - p1)
        odds_s.append(np.exp(s2) / np.exp(s1))
        odds_p.append(p2 / p1)
    n = len(scores)

    return n, pos_count_s / n, np.mean(delta_s), np.mean(
        odds_s), pos_count_p / n, np.mean(delta_p), np.mean(odds_p)


def compute_top1_accuracy(pred_dist):
    pre_count = 0
    post_count = 0
    for ex in pred_dist:
        scores, label = ex
        if not isinstance(label, list):
            label = [label]

        # pre
        pre_probs = [s[-2] for s in scores]
        pre_id = np.argmax(pre_probs)
        if scores[pre_id][0] in label:
            pre_count += 1

        # post
        post_probs = [s[-1] for s in scores]
        post_id = np.argmax(post_probs)
        if scores[post_id][0] in label:
            post_count += 1

    n = len(pred_dist)

    return pre_count / n, post_count / n, n