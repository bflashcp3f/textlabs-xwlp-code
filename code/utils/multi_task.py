

import math
import glob
import time
import json
import pickle
import os
import numpy as np
import operator
import time
import sys
import random

from collections import defaultdict, Counter, OrderedDict
from datetime import datetime



ENT_NAME = ['rg', 'convert-op', 'seal-op', 'spin-op', 'create-op', \
            'd', 'default-op', 'destroy-op', 'loc', 'm', 'measure-op', 'mix-op', \
            'mod', 'mth', 'remove-op', 's', 'sl', 'temp-treat-op', 'time-op', \
            'transfer-op', 'wash-op']

ID2LABEL = ['no_relation', 'ARG0', 'ARG1', 'ARG2', 'site', 'succ', 'co-ref-of', 'located-at',
            'measure', 'modifier', 'part-of', 'setting', 'usage']

LABEL2ID = dict([(label, id) for id, label in enumerate(ID2LABEL)])

NO_RELATION = "no_relation"

ALL_POS_RELS = ['ARG0', 'ARG1', 'ARG2', 'site', 'succ', 'co-ref-of', 'located-at',
            'measure', 'modifier', 'part-of', 'setting', 'usage']


WITHIN_SEN_RELS = ['setting', 'modifier', 'measure']

CORE_REL = ["ARG0", "ARG1", "ARG2"]
NON_CORE_REL = ['site', 'co-ref-of', 'located-at', 'measure', 'modifier', 'part-of', 'setting', 'usage']
TD_REL = ['succ']


def load_from_jsonl(file_name):
    data_list = []
    with open(file_name) as f:
        for line in f:
            data_list.append(json.loads(line))

    return data_list


def handle_doc_offset_ner(ner_tuple, doc_len, remove=True):
    # print(ner_tuple)
    ner_start, ner_end, ent_type = ner_tuple

    if remove:
        return [ner_start - doc_len, ner_end - doc_len, ent_type]
    else:
        return [ner_start + doc_len, ner_end + doc_len, ent_type]


def handle_doc_offset_rel(rel_tuple, doc_len, remove=True):
    ent1_start, ent1_end, ent2_start, ent2_end, rel_str = rel_tuple

    if remove:
        return [ent1_start - doc_len, ent1_end - doc_len, ent2_start - doc_len, ent2_end - doc_len, rel_str]
    else:
        return [ent1_start + doc_len, ent1_end + doc_len, ent2_start + doc_len, ent2_end + doc_len, rel_str]


def eveluate(TP, FP, FN, verbose=False):
    #     TP = len(set(gold_list) & set(pred_list))
    #     FP = len(set(pred_list)) - TP
    #     FN = len(set(gold_list)) - TP

    if TP + FP > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0

    if TP + FN > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0

    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    if verbose:
        print("Precision: {:.1f}".format(precision * 100))
        print("Recall: {:.1f}".format(recall * 100))
        print("F1: {:.1f}".format(f1 * 100))
        print(f"# gold: {TP + FN}")

    return precision, recall, f1


def find_word_sen_idx(doc_sen_len):
    doc_len = 0
    word2sen_idx = {}

    for sen_idx, sen_len in enumerate(doc_sen_len):

        for i in range(doc_len, doc_len + sen_len):
            word2sen_idx[i] = sen_idx

        doc_len += sen_len

    return word2sen_idx


def get_pre_sen_len(doc_data_sole, window_size, seq_len_limit=150):
    sen_list = doc_data_sole['sentences']
    doc_sen_num = len(sen_list)

    start_sen_idx = 1 - window_size
    end_sen_idx = doc_sen_num + window_size

    start_idx = start_sen_idx
    end_idx = start_idx + window_size

    start_idx_final_list = []
    end_idx_final_list = []
    offset_final_list = []

    pre_sen_len_list = []

    while end_idx < end_sen_idx:

        if start_idx < 0 and end_idx <= doc_sen_num:
            start_idx_final, end_idx_final = 0, end_idx
        elif start_idx < 0 and end_idx > doc_sen_num:
            start_idx_final, end_idx_final = 0, doc_sen_num
        elif start_idx >= 0 and end_idx <= doc_sen_num:
            start_idx_final, end_idx_final = start_idx, end_idx
        elif start_idx >= 0 and end_idx > doc_sen_num:
            start_idx_final, end_idx_final = start_idx, doc_sen_num
        else:
            raise

        if sum([len(sublist) for sublist in sen_list[start_idx_final:end_idx_final]]) <= seq_len_limit:
            #             print(sum([len(sublist) for sublist in sen_list[start_idx_final:end_idx_final]]))
            start_idx_final_list.append(start_idx_final)
            end_idx_final_list.append(end_idx_final)
            offset_final_list.append((start_idx_final, end_idx_final))

            pre_sen_len_list.append(sum([len(item) for item in sen_list[:start_idx_final]]))

        start_idx += 1
        end_idx = start_idx + window_size

    return pre_sen_len_list, offset_final_list


def decode_sliding_window(doc_data, pre_sen_length_list, window_size):
    sen_list = doc_data['sentences']
    ner_list = doc_data['predicted_ner']
    rel_list = doc_data['predicted_relations']

    overall_doc_size = 0
    assert len(pre_sen_length_list) == len(sen_list)

    ner_list_decoded_all = []
    rel_list_decoded_all = []

    for sen_idx, (sen_list_per_sen, ner_list_per_sen, rel_list_per_sen) in \
            enumerate(zip(sen_list, ner_list, rel_list)):
        pre_sen_length = pre_sen_length_list[sen_idx]

        ner_list_decoded = [handle_doc_offset_ner(item, overall_doc_size, remove=True) for item in ner_list_per_sen]
        rel_list_decoded = [handle_doc_offset_rel(item, overall_doc_size, remove=True) for item in rel_list_per_sen]

        overall_doc_size += len(sen_list_per_sen)

        ner_list_decoded = [handle_doc_offset_ner(item, pre_sen_length, remove=False) for item in ner_list_decoded]
        rel_list_decoded = [handle_doc_offset_rel(item, pre_sen_length, remove=False) for item in rel_list_decoded]

        ner_list_decoded_all.append(ner_list_decoded)
        rel_list_decoded_all.append(rel_list_decoded)

    return ner_list_decoded_all, rel_list_decoded_all