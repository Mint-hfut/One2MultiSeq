import torch
from transformers.utils.versions import require_version
import numpy as np
from transformers import BartTokenizer, Seq2SeqTrainingArguments, BartConfig
import time
import json
import logging
import os
from collections import defaultdict
from typing import Optional
from dataclasses import field, dataclass
from nltk.stem import *
from seq2seq_trainer_ import Seq2SeqTrainer

from One2MultiSeq import CopyBartForConditionalGeneration,BartForConditionalGeneration_
from One2Set import One2SetBartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right
import copy
UNK_WORD = '<unk>'
require_version("transformers==4.21.2", "To fix: pip install transformers==4.21.2")
#require_version("torch==1.11.0+cu113", "To fix: pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113")
model_name = 'facebook/bart-base'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#"0,1"
os.environ["WANDB_DISABLED"] = "true"
@dataclass
class Seq2SeqTrainingArguments_(Seq2SeqTrainingArguments):
    topk_list: list = field(
        default=None, metadata={"help": "How many keywords are generated"}
    )
    encoder_input_length: int = field(
        default=None, metadata={"help": "the max length of encoder tokenizer"}
    )
    decoder_input_length: int = field(
        default=None, metadata={"help": "the max length of decoder tokenizer"}
    )
    data_type: str = field(
        default=None, metadata={"help": "One2Seq or One2MultiSeq"}
    )
    src_lines: str = field(
        default=None)
    singe_data_args: dict = field(
        default=None)
    test_dataset_name: str = field(
        default=None
    )
    meng_rui_precision: bool = field(
        default=None
    )

tokenizer = BartTokenizer.from_pretrained(model_name)
tokenizer.add_tokens(['<number>', '<url>', '<mention>', '<digit>', '<eos>', '<NULL>'], special_tokens=True)



class One2SetDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(self.data,list):
            return self.data[idx]



def fetch_datasets(training_args=None):
    print("loading valid dataset from {}".format(training_args.singe_data_args['train_data_path']+'/valid_One2Seq.pt'))
    valid_data = torch.load(training_args.singe_data_args['train_data_path']+'/valid_One2Seq.pt')

    if training_args.do_train:
        print("loading train dataset from {}".format(training_args.singe_data_args['train_data_path']+f'/train_{training_args.data_type}.pt'))
        train_data = torch.load(training_args.singe_data_args['train_data_path']+f'/train_{training_args.data_type}.pt')
    else:  # Reduce load time
        train_data = valid_data
    print("loading test dataset from {}".format(training_args.singe_data_args['test_data_path'][training_args.test_dataset_name]+'/test_One2Seq.pt'))
    test_data = torch.load(training_args.singe_data_args['test_data_path'][training_args.test_dataset_name]+'/test_One2Seq.pt')
    train_dataset = One2SetDataset(train_data)
    val_dataset = One2SetDataset(valid_data)
    test_dataset = One2SetDataset(test_data)

    return train_dataset, val_dataset, test_dataset





stemmer = PorterStemmer()
def postprocess_text(src_lines, preds, labels, sep_token):
    src_lines = [src_line.strip().split('<seg>')[0].strip().split() for src_line in src_lines]#attention: it is "<seg>", not <sep>, different from social media dataset
    src_lines = [[stemmer.stem(w.strip().lower()) for w in src_line] for src_line in src_lines]
    preds = [[pred.lower().replace('</s>', '').replace('<pad>', '').replace('<s>', '')] for pred in preds]
    labels = [[label.lower().replace('</s>', '').replace('<pad>', '').replace('<s>', '')] for label in labels]
    preds = [[' '.join([stemmer.stem(w) for w in p.split()]) for p in pred] for pred in preds]
    labels = [[' '.join([stemmer.stem(w) for w in p.split()]) for p in label] for label in labels]
    preds = [[p.strip() for p in pred if len(p.strip()) > 0] for pred in preds]
    labels = [[p.strip() for p in label if len(p.strip()) > 0] for label in labels]
    concatenated_preds = []
    concatenated_labels = []
    i = 0
    temp_pred = []
    temp_label = []
    for pred, label in zip(preds, labels):
        if len(pred)>0:
            if '<null>' not in pred[0]:
                temp_pred.append(pred[0])
        if '<null>' not in label[0]:
            temp_label.append(label[0])
        i += 1
        if i == 20:
            concatenated_preds.append(temp_pred)
            concatenated_labels.append(temp_label)
            temp_pred = []
            temp_label = []
            i = 0
        assert len(concatenated_preds) == len(concatenated_labels)
    return src_lines, concatenated_preds, concatenated_labels

def compute_precision(num_matches, num_predictions):
    return num_matches / num_predictions if num_predictions > 0 else 0.0
def compute_recall(num_matches, num_trgs):
    return num_matches / num_trgs if num_trgs > 0 else 0.0


def compute_f1(precision, recall):
    return float(2 * (precision * recall)) / (precision + recall) if precision + recall > 0 else 0.0
def average_precision_at_k(k_list, trg_list, pred_list):# Used to calculate the MAP@k
    num_trgs = len(trg_list)
    num_predictions = len(pred_list)
    is_match = np.zeros(num_predictions, dtype=bool)
    for pred_idx, pred in enumerate(pred_list):
        for trg_idx, trg in enumerate(trg_list):
            if pred == trg:
                is_match[pred_idx] = True
                break
    r=is_match
    if num_predictions == 0 or num_trgs == 0:
        return [0] * len(k_list)
    # k_max = max(k_list)
    k_max = -1
    for k in k_list:
        if k == 'M':
            k = num_predictions
        if k > k_max:
            k_max = k
    if num_predictions > k_max:
        num_predictions = k_max
        r = r[:num_predictions]
    r_cum_sum = np.cumsum(r, axis=0)
    precision_array = [compute_precision(r_cum_sum[k], k + 1) * r[k] for k in range(num_predictions)]
    precision_cum_sum = np.cumsum(precision_array, axis=0)
    average_precision_array = precision_cum_sum / num_trgs
    return_indices = []
    for k in k_list:
        if k == 'M':
            k = num_predictions
        return_indices.append((k - 1) if k <= num_predictions else (num_predictions - 1))
    return_indices = np.array(return_indices, dtype=int)
    return average_precision_array[return_indices]

def check_present_keyphrases(src_str, keyphrase_str_list):
    """
    :param src_str: stemmed word list of source text
    :param keyphrase_str_list: stemmed list of word list
    :return:
    """
    num_keyphrases = len(keyphrase_str_list)
    is_present = np.zeros(num_keyphrases, dtype=bool)

    for i, keyphrase_word_list in enumerate(keyphrase_str_list):
        joined_keyphrase_str = ' '.join(keyphrase_word_list)

        if joined_keyphrase_str.strip() == "":  # if the keyphrase is an empty string
            is_present[i] = False
        else:

            # check if it appears in source text
            match = False
            for src_start_idx in range(len(src_str) - len(keyphrase_word_list) + 1):
                match = True
                for keyphrase_i, keyphrase_w in enumerate(keyphrase_word_list):
                    src_w = src_str[src_start_idx + keyphrase_i]
                    if src_w != keyphrase_w:
                        match = False
                        break
                if match:
                    break
            if match:
                is_present[i] = True
            else:
                is_present[i] = False
    return is_present
def separate_present_absent_by_source(src_token_list_stemmed, keyphrase_token_2dlist_stemmed):
    is_present_mask = check_present_keyphrases(src_token_list_stemmed, keyphrase_token_2dlist_stemmed)
    present_keyphrase_token2dlist = []
    absent_keyphrase_token2dlist = []
    for keyphrase_token_list, is_present in zip(keyphrase_token_2dlist_stemmed, is_present_mask):
        if is_present:
            present_keyphrase_token2dlist.append(keyphrase_token_list)
        else:
            absent_keyphrase_token2dlist.append(keyphrase_token_list)
    return present_keyphrase_token2dlist, absent_keyphrase_token2dlist
def stem_word_list(word_list):
    return [stemmer.stem(w.strip().lower()) for w in word_list]
def stem_str_list(str_list):
    # stem every word in a list of word list
    # str_list is a list of word list
    stemmed_str_list = []
    for word_list in str_list:
        stemmed_word_list = stem_word_list(word_list)
        stemmed_str_list.append(stemmed_word_list)
    return stemmed_str_list
def check_duplicate_keyphrases(keyphrase_str_list):
    """
    :param keyphrase_str_list: a 2d list of tokens
    :return: a boolean np array indicate, 1 = unique, 0 = duplicate
    """
    num_keyphrases = len(keyphrase_str_list)
    not_duplicate = np.ones(num_keyphrases, dtype=bool)
    keyphrase_set = set()
    for i, keyphrase_word_list in enumerate(keyphrase_str_list):
        if '_'.join(keyphrase_word_list) in keyphrase_set:
            not_duplicate[i] = False
        else:
            not_duplicate[i] = True
        keyphrase_set.add('_'.join(keyphrase_word_list))
    return not_duplicate
def find_unique_target(trg_token_2dlist_stemmed):
    """
    Remove the duplicate targets
    :param trg_token_2dlist_stemmed:
    :return:
    """
    num_trg = len(trg_token_2dlist_stemmed)
    is_unique_mask = check_duplicate_keyphrases(trg_token_2dlist_stemmed)  # boolean array, 1=unqiue, 0=duplicate
    trg_filter = is_unique_mask
    filtered_stemmed_trg_str_list = [word_list for word_list, is_keep in
                                     zip(trg_token_2dlist_stemmed, trg_filter) if
                                     is_keep]
    num_duplicated_trg = num_trg - np.sum(is_unique_mask)
    return filtered_stemmed_trg_str_list, num_duplicated_trg
def compute_match_result(trg_str_list, pred_str_list, type='exact', dimension=1):
    assert type in ['exact', 'sub'], "Right now only support exact matching and substring matching"
    assert dimension in [1, 2], "only support 1 or 2"
    num_pred_str = len(pred_str_list)
    num_trg_str = len(trg_str_list)
    if dimension == 1:
        is_match = np.zeros(num_pred_str, dtype=bool)
        for pred_idx, pred_word_list in enumerate(pred_str_list):
            joined_pred_word_list = ' '.join(pred_word_list)
            for trg_idx, trg_word_list in enumerate(trg_str_list):
                joined_trg_word_list = ' '.join(trg_word_list)
                if type == 'exact':
                    if joined_pred_word_list == joined_trg_word_list:
                        is_match[pred_idx] = True
                        break
                elif type == 'sub':
                    if joined_pred_word_list in joined_trg_word_list:
                        is_match[pred_idx] = True
                        break
    else:
        is_match = np.zeros((num_trg_str, num_pred_str), dtype=bool)
        for trg_idx, trg_word_list in enumerate(trg_str_list):
            joined_trg_word_list = ' '.join(trg_word_list)
            for pred_idx, pred_word_list in enumerate(pred_str_list):
                joined_pred_word_list = ' '.join(pred_word_list)
                if type == 'exact':
                    if joined_pred_word_list == joined_trg_word_list:
                        is_match[trg_idx][pred_idx] = True
                elif type == 'sub':
                    if joined_pred_word_list in joined_trg_word_list:
                        is_match[trg_idx][pred_idx] = True
    return is_match
def compute_classification_metrics(num_matches, num_predictions, num_trgs):
    precision = compute_precision(num_matches, num_predictions)
    recall = compute_recall(num_matches, num_trgs)
    f1 = compute_f1(precision, recall)
    return precision, recall, f1
def compute_classification_metrics_at_ks(is_match, num_predictions, num_trgs, k_list=[5, 10], meng_rui_precision=False):
    """
    :param is_match: a boolean np array with size [num_predictions]
    :param predicted_list:
    :param true_list:
    :param topk:
    :return: {'precision@%d' % topk: precision_k, 'recall@%d' % topk: recall_k, 'f1_score@%d' % topk: f1, 'num_matches@%d': num_matches}
    """
    assert is_match.shape[0] == num_predictions
    # topk.sort()
    if num_predictions == 0:
        precision_ks = [0] * len(k_list)
        recall_ks = [0] * len(k_list)
        f1_ks = [0] * len(k_list)
        num_matches_ks = [0] * len(k_list)
        num_predictions_ks = [0] * len(k_list)
    else:
        num_matches = np.cumsum(is_match)
        num_predictions_ks = []
        num_matches_ks = []
        precision_ks = []
        recall_ks = []
        f1_ks = []
        for topk in k_list:
            if topk == 'M':
                topk = num_predictions
            elif topk == 'G':
                # topk = num_trgs
                if num_predictions < num_trgs:
                    topk = num_trgs
                else:
                    topk = num_predictions

            if meng_rui_precision:
                if num_predictions > topk:
                    num_matches_at_k = num_matches[topk - 1]
                    num_predictions_at_k = topk
                else:
                    num_matches_at_k = num_matches[-1]
                    num_predictions_at_k = num_predictions
            else:
                if num_predictions > topk:
                    num_matches_at_k = num_matches[topk - 1]
                else:
                    num_matches_at_k = num_matches[-1]
                num_predictions_at_k = topk

            precision_k, recall_k, f1_k = compute_classification_metrics(num_matches_at_k, num_predictions_at_k,
                                                                         num_trgs)
            precision_ks.append(precision_k)
            recall_ks.append(recall_k)
            f1_ks.append(f1_k)
            num_matches_ks.append(num_matches_at_k)
            num_predictions_ks.append(num_predictions_at_k)
    return precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks
def average_precision_at_ks(r, k_list, num_predictions, num_trgs):
    if num_predictions == 0 or num_trgs == 0:
        return [0] * len(k_list)
    # k_max = max(k_list)
    k_max = -1
    for k in k_list:
        if k == 'M':
            k = num_predictions
        elif k == 'G':
            # k = num_trgs
            if num_predictions < num_trgs:
                k = num_trgs
            else:
                k = num_predictions
        if k > k_max:
            k_max = k
    if num_predictions > k_max:
        num_predictions = k_max
        r = r[:num_predictions]
    r_cum_sum = np.cumsum(r, axis=0)
    precision_array = [compute_precision(r_cum_sum[k], k + 1) * r[k] for k in range(num_predictions)]
    precision_cum_sum = np.cumsum(precision_array, axis=0)
    average_precision_array = precision_cum_sum / num_trgs
    return_indices = []
    for k in k_list:
        if k == 'M':
            k = num_predictions
        elif k == 'G':
            # k = num_trgs
            if num_predictions < num_trgs:
                k = num_trgs
            else:
                k = num_predictions
        return_indices.append((k - 1) if k <= num_predictions else (num_predictions - 1))
    return_indices = np.array(return_indices, dtype=int)
    return average_precision_array[return_indices]
def update_score_dict(trg_token_2dlist_stemmed, pred_token_2dlist_stemmed, k_list, score_dict, tag):
    num_targets = len(trg_token_2dlist_stemmed)
    num_predictions = len(pred_token_2dlist_stemmed)

    is_match = compute_match_result(trg_token_2dlist_stemmed, pred_token_2dlist_stemmed,
                                    type='exact', dimension=1)
    # Classification metrics
    precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks = \
        compute_classification_metrics_at_ks(is_match, num_predictions, num_targets, k_list=k_list)

    ap_ks = average_precision_at_ks(is_match, k_list=k_list,
                                    num_predictions=num_predictions, num_trgs=num_targets)

    for topk, precision_k, recall_k, f1_k, num_matches_k, num_predictions_k, ap_k in \
            zip(k_list, precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks, ap_ks):
        score_dict['precision@{}_{}'.format(topk, tag)].append(precision_k)
        score_dict['recall@{}_{}'.format(topk, tag)].append(recall_k)
        score_dict['f1_score@{}_{}'.format(topk, tag)].append(f1_k)
        score_dict['num_matches@{}_{}'.format(topk, tag)].append(num_matches_k)
        score_dict['num_predictions@{}_{}'.format(topk, tag)].append(num_predictions_k)
        score_dict['num_targets@{}_{}'.format(topk, tag)].append(num_targets)
        score_dict['AP@{}_{}'.format(topk, tag)].append(ap_k)
    return score_dict
def filter_prediction(disable_valid_filter, disable_extra_one_word_filter, pred_token_2dlist_stemmed):
    """
    Remove the duplicate predictions, can optionally remove invalid predictions and extra one word predictions
    :param disable_valid_filter:
    :param disable_extra_one_word_filter:
    :param pred_token_2dlist_stemmed:
    :param pred_token_2d_list:
    :return:
    """
    num_predictions = len(pred_token_2dlist_stemmed)
    is_unique_mask = check_duplicate_keyphrases(pred_token_2dlist_stemmed)  # boolean array, 1=unqiue, 0=duplicate
    pred_filter = is_unique_mask
    if not disable_valid_filter:
        is_valid_mask = check_valid_keyphrases(pred_token_2dlist_stemmed)
        pred_filter = pred_filter * is_valid_mask
    if not disable_extra_one_word_filter:
        extra_one_word_seqs_mask, num_one_word_seqs = compute_extra_one_word_seqs_mask(pred_token_2dlist_stemmed)
        pred_filter = pred_filter * extra_one_word_seqs_mask
    filtered_stemmed_pred_str_list = [word_list for word_list, is_keep in
                                      zip(pred_token_2dlist_stemmed, pred_filter) if
                                      is_keep]
    num_duplicated_predictions = num_predictions - np.sum(is_unique_mask)
    return filtered_stemmed_pred_str_list, num_duplicated_predictions
def check_valid_keyphrases(str_list):
    num_pred_seq = len(str_list)
    is_valid = np.zeros(num_pred_seq, dtype=bool)
    for i, word_list in enumerate(str_list):
        keep_flag = True

        if len(word_list) == 0:
            keep_flag = False

        for w in word_list:
            if w == UNK_WORD or w == ',' or w == '.':
                keep_flag = False
                print(word_list)

        is_valid[i] = keep_flag

    return is_valid
def compute_extra_one_word_seqs_mask(str_list):
    num_pred_seq = len(str_list)
    mask = np.zeros(num_pred_seq, dtype=bool)
    num_one_word_seqs = 0
    for i, word_list in enumerate(str_list):
        if len(word_list) == 1:
            num_one_word_seqs += 1
            if num_one_word_seqs > 1:
                mask[i] = False
                continue
        mask[i] = True
    return mask, num_one_word_seqs
def remove_M(top_list):
    if 'M' in top_list:
        top_list.remove('M')
    return top_list

def compute_metrics(eval_preds, num_return_sequences=1, topk_list=[5,'M'], present_tags =['all','present','absent'],src_lines=None):
    score_dict = defaultdict(list)
    ignore_pad_token_for_loss = True
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    preds = preds.reshape(-1, preds.shape[-1])
    labels = labels.reshape(-1, labels.shape[-1])
    if isinstance(preds, tuple):
        preds = preds[0]
    if len(preds.shape) == 3:
        preds = preds.argmax(axis=-1)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)
    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)

    # Some simple post-processing
    if isinstance(src_lines,str):
        src_lines = open(src_lines, 'r', encoding='utf-8').readlines()
    elif isinstance(src_lines,list):
        src_lines = src_lines
    if src_lines is None:
        src_lines = eval_preds.srclines
        num_return_sequences=20
    src_text,decoded_preds, decoded_labels = postprocess_text(src_lines,decoded_preds, decoded_labels, tokenizer.sep_token)

    if len(decoded_preds) != len(decoded_labels):  # 将num_return_sequences个关键词序列进行拼接
        decoded_preds_for_cal_at_k = [[] for _ in range(len(decoded_labels))]
        decoded_preds_for_cal_at_M = [[] for _ in range(len(decoded_labels))]
        assert len(decoded_preds) // num_return_sequences == len(decoded_labels)
        for i in range(len(decoded_labels)):
            for j in range(num_return_sequences):
                index = i * num_return_sequences + j
                for k in decoded_preds[index]:
                    if j == 0:#Use only the keyphrases in the first sentence to calculate M
                        if k not in decoded_preds_for_cal_at_M[i] and len(k) > 1:
                            decoded_preds_for_cal_at_M[i].append(k)
                    if k not in decoded_preds_for_cal_at_k[i] and len(k) > 1:#Use all the keyphrases in the returned sentence (beam search)
                        decoded_preds_for_cal_at_k[i].append(k)
        decoded_preds = decoded_preds_for_cal_at_k
        decoded_preds_for_M = decoded_preds_for_cal_at_M
    else:
        decoded_preds_for_cal_at_k = [[] for _ in range(len(decoded_labels))]
        for i in range(len(decoded_labels)):
            for k in decoded_preds[i]:
                if k not in decoded_preds_for_cal_at_k[i] and len(k) > 1:#Use all the keyphrases in the returned sentence (beam search)
                    decoded_preds_for_cal_at_k[i].append(k)
        decoded_preds = decoded_preds_for_cal_at_k
        decoded_preds_for_M = decoded_preds

    for src, pred, pred_for_M, label in zip(src_text, decoded_preds,decoded_preds_for_M, decoded_labels):
        trg_token_2dlist = [trg_str.strip().split(' ') for trg_str in label]
        pred_token_2dlist = [pred_str.strip().split(' ') for pred_str in pred]
        pred_token_2dlist_for_M = [pred_str.strip().split(' ') for pred_str in pred_for_M]
        #pred_token_2dlist_for_M = [pred_str_for_M.strip().split(' ') for pred_str_for_M in pred_for_M]
        stemmed_trg_token_2dlist = stem_str_list(trg_token_2dlist)
        stemmed_pred_token_2dlist = stem_str_list(pred_token_2dlist)
        stemmed_pred_token_2dlist_for_M = stem_str_list(pred_token_2dlist_for_M)
        filtered_stemmed_pred_token_2dlist, num_duplicated_predictions = filter_prediction(False,
                                                                                           True,
                                                                                           stemmed_pred_token_2dlist)
        filtered_stemmed_pred_token_2dlist_for_M, num_duplicated_predictions_for_M = filter_prediction(False,
                                                                                           True,
                                                                                           stemmed_pred_token_2dlist_for_M)
        unique_stemmed_trg_token_2dlist, num_duplicated_trg = find_unique_target(stemmed_trg_token_2dlist)
        present_filtered_stemmed_pred, absent_filtered_stemmed_pred = separate_present_absent_by_source(
            src, filtered_stemmed_pred_token_2dlist)
        present_filtered_stemmed_pred_for_M, absent_filtered_stemmed_pred_for_M = separate_present_absent_by_source(
            src, filtered_stemmed_pred_token_2dlist_for_M)
        present_unique_stemmed_trg, absent_unique_stemmed_trg = separate_present_absent_by_source(
            src, unique_stemmed_trg_token_2dlist)
        topk_list_ = copy.deepcopy(topk_list)
        remove_M(topk_list_)
        if filtered_stemmed_pred_token_2dlist:
            score_dict = update_score_dict(unique_stemmed_trg_token_2dlist,
                                           filtered_stemmed_pred_token_2dlist,
                                           topk_list_, score_dict, 'all')
            if 'M' in topk_list:
                score_dict = update_score_dict(unique_stemmed_trg_token_2dlist,
                                               filtered_stemmed_pred_token_2dlist,
                                               ['M'], score_dict, 'all')
        if present_unique_stemmed_trg:
            score_dict = update_score_dict(present_unique_stemmed_trg,
                                           present_filtered_stemmed_pred,
                                           topk_list_, score_dict, 'present')
            if 'M' in topk_list:
                score_dict = update_score_dict(present_unique_stemmed_trg,
                                               present_filtered_stemmed_pred_for_M,
                                               ['M'], score_dict, 'present')
        if absent_unique_stemmed_trg:
            score_dict = update_score_dict(absent_unique_stemmed_trg,
                                           absent_filtered_stemmed_pred,
                                           topk_list_, score_dict, 'absent')
            if 'M' in topk_list:
                score_dict = update_score_dict(absent_unique_stemmed_trg,
                                               absent_filtered_stemmed_pred_for_M,
                                               ['M'], score_dict, 'absent')
    names = locals()
    result = {}
    for topk in topk_list:
        for present_tag in present_tags:
            names['total_predictions_%s_%s' % (topk,present_tag)] = sum(score_dict['num_predictions@{}_{}'.format(topk, present_tag)])

            names['total_targets_%s_%s' % (topk,present_tag)] = sum(score_dict['num_targets@{}_{}'.format(topk, present_tag)])

            names['total_num_matches_%s_%s' % (topk,present_tag)] = sum(score_dict['num_matches@{}_{}'.format(topk, present_tag)])
            # Compute the micro averaged recall, precision and F-1 score

            names['micro_avg_precision_%s_%s' % (topk,present_tag)], names['micro_avg_recall_%s_%s' % (topk,present_tag)], names['micro_avg_f1_score_%s_%s' % (topk,present_tag)] = compute_classification_metrics(
                names['total_num_matches_%s_%s' % (topk,present_tag)], names['total_predictions_%s_%s' % (topk,present_tag)],
                names['total_targets_%s_%s' % (topk,present_tag)])

            names['macro_avg_precision_%s_%s' % (topk,present_tag)] = sum(score_dict['precision@{}_{}'.format(topk, present_tag)]) / len(
                score_dict['precision@{}_{}'.format(topk, present_tag)]) if len(
                score_dict['precision@{}_{}'.format(topk, present_tag)]) > 0 else 0.0

            names['macro_avg_recall_%s_%s' % (topk,present_tag)] = sum(score_dict['recall@{}_{}'.format(topk, present_tag)]) / len(
                score_dict['recall@{}_{}'.format(topk, present_tag)]) if len(
                score_dict['recall@{}_{}'.format(topk, present_tag)]) > 0 else 0.0

            names['macro_avg_f1_score_%s_%s' % (topk,present_tag)] = float(2 * names['macro_avg_precision_%s_%s' % (topk,present_tag)] * names['macro_avg_recall_%s_%s' % (topk,present_tag)]) / (
                    names['macro_avg_precision_%s_%s' % (topk,present_tag)] + names['macro_avg_recall_%s_%s' % (topk,present_tag)]) if (
                    names['macro_avg_precision_%s_%s' % (topk,present_tag)] + names['macro_avg_recall_%s_%s' % (topk,present_tag)]) > 0 else 0.0

            names['MAP_%s_%s' % (topk,present_tag)] = sum(score_dict['AP@{}_{}'.format(topk, present_tag)]) / len(score_dict['AP@{}_{}'.format(topk, present_tag)]) if \
                                                                                                            len(score_dict['AP@{}_{}'.format(topk, present_tag)])>0 else 0

            result['macro_avg_f1_score_%s_%s' % (topk,present_tag)]=names['macro_avg_f1_score_%s_%s' % (topk,present_tag)]
            result['MAP_%s_%s' % (topk,present_tag)]=names['MAP_%s_%s' % (topk,present_tag)]
    result = {k: round(v, 4) for k, v in result.items()}
    return result, decoded_preds, decoded_labels


def main(seed, data_name, singe_data_args, test_dataset_name,do_train):
    training_args = Seq2SeqTrainingArguments_(
        output_dir='./models',
        num_train_epochs=10,
        per_device_train_batch_size=singe_data_args['per_device_train_batch_size'],
        per_device_eval_batch_size=16,
        warmup_steps=singe_data_args['warmup_steps'],
        weight_decay=0.01,
        logging_dir='./logs',
        do_train=do_train,
        do_eval=False,
        do_predict=True,
        save_strategy="no",
        learning_rate=singe_data_args['learning_rate'],
        fp16=True,
        generation_num_beams=1,
        generation_max_length=singe_data_args['generation_max_length'],
        predict_with_generate=True,
        seed=seed,  # random.randint(0,100)
        data_seed=seed,
        topk_list=singe_data_args['topk_list'],
        # generation max length is 8 by default.if you set n_keywords greater than 1, generation_max_length will automatically change to 16
        data_type='One2Set',  # select the training dataset, 'One2MultiSeq' or 'One2Seq'
        src_lines=singe_data_args['test_data_path'][test_dataset_name]+r'/test_src.txt',
        singe_data_args=singe_data_args,
        test_dataset_name=test_dataset_name,
        meng_rui_precision=singe_data_args['meng_rui_precision']
    )
    training_args.output_dir = f'models/temp_model/{data_name}/CopyBART_{training_args.data_type}_' \
                               f'{"base" if "base" in model_name else "large"}_epochs-{training_args.num_train_epochs}_' \
                               f'learning_rate-{training_args.learning_rate}_batch_size-{training_args.train_batch_size}_seed-{seed}'

    train_dataset, val_dataset, test_dataset = fetch_datasets(training_args)

    model = CopyBartForConditionalGeneration.from_pretrained(model_name,
        encoder_input_length=training_args.encoder_input_length, seed=training_args.seed, )
    if 'KP20K' in training_args.singe_data_args['train_data_path']:
        tokenizer.add_tokens(['<eos>'], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    model.loss_weight = torch.ones(model.vocab_size).to("cuda:0")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = len(train_dataset)  # 60
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    logger = logging.getLogger(__name__)
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(output_scores=True, output_attentions=True, return_dict_in_generate=True,
                                   metric_key_prefix="eval", num_return_sequences=training_args.generation_num_beams)
        max_eval_samples = len(val_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(val_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            test_dataset, metric_key_prefix="predict", output_attentions=True,
            output_scores=True, return_dict_in_generate=True, num_return_sequences=training_args.generation_num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = len(test_dataset)  # 60
        metrics["predict_samples"] = min(max_predict_samples, len(test_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = predict_results.predictions
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join([json.dumps(pred) for pred in predictions]))


if __name__ == "__main__":
    for seed in [100]:
        model_name ='facebook/bart-base' #facebook/bart-large,'facebook/bart-base'
        # When in the inference phase, you can choose your own trained model path, eg:
        # models/temp_model/kp20k/BART_One2Set_base_epoch10_learning_rate3e-4_batchsize-32-seed300
        all_data_args= {'CMKP':{'encoder_input_length':125,
                              'decoder_input_length':12,
                              'generation_max_length': 12,
                              'per_device_train_batch_size':32,
                              'warmup_steps': 2000,
                              'learning_rate': 5e-5 if 'base' in model_name else 4e-5,
                              'train_data_path':'data/CMKP_data',
                              'test_data_path':{'CMKP': 'data/CMKP_data'},
                               'meng_rui_precision':False,
                               'topk_list':[1, 3, 5]
                              },

                     'Twitter': {'encoder_input_length': 64,
                              'decoder_input_length': 12,
                              'generation_max_length': 12,
                             'per_device_train_batch_size': 32,
                             'warmup_steps': 2000,
                             'learning_rate': 5e-5 if 'base' in model_name else 4e-5,
                              'train_data_path': 'data/Twitter_data',
                              'test_data_path': {'Twitter': 'data/Twitter_data'},
                              'meng_rui_precision':False,
                                'topk_list': [1, 3, 5]
                                 },

                     'StackExchange': {'encoder_input_length': 128,
                              'decoder_input_length': 12,
                              'generation_max_length': 12,
                               'per_device_train_batch_size': 32,
                               'warmup_steps': 2000,
                               'learning_rate': 5e-5 if 'base' in model_name else 4e-5,
                              'train_data_path': 'data/StackExchange_data',
                              'test_data_path': {'StackExchange': 'data/StackExchange_data'},
                                'meng_rui_precision':False,
                                'topk_list': [1, 3, 5]
                                       },

                     'KP20K': {'encoder_input_length': 192,
                              'decoder_input_length': 12,
                              'generation_max_length': 12,
                               'per_device_train_batch_size': 32,
                               'warmup_steps': 8000,
                               'learning_rate': 5e-5 if 'base' in model_name else 4e-5,
                              'train_data_path': 'data/KP20K/kp20k_separated',
                              'test_data_path': {'KP20K': 'data/KP20K/testsets/kp20k',
                                                 'inspec': 'data/KP20K/testsets/inspec',
                                                 'krapivin': 'data/KP20K/testsets/krapivin',
                                                 'nus': 'data/KP20K/testsets/nus',
                                                 'semeval': 'data/KP20K/testsets/semeval',
                                                 },
                              'meng_rui_precision':False,
                               'topk_list': [5, 'M'],
                               },
             }

        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        tokenizer.add_tokens(['<number>', '<url>', '<mention>', '<digit>', '<eos>', '<NULL>'], special_tokens=True)
        train_dataset = 'CMKP'#CMKP,Twitter,StackExchange,KP20K
        test_dataset_name = 'CMKP'#CMKP,Twitter,StackExchange, or kp20k,inspec,nus,krapivin,semeval
        do_train = True
        main(seed,train_dataset, all_data_args[train_dataset],test_dataset_name, do_train)







