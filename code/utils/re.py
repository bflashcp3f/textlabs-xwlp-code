
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

import torch
from transformers import RobertaTokenizer, RobertaForTokenClassification, RobertaConfig, RobertaModel
from transformers import AutoModelForTokenClassification, AutoConfig, AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel
from transformers import BertForTokenClassification, BertPreTrainedModel, AdamW
from transformers import AdamW
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import KFold
from ipywidgets import IntProgress

from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from tqdm import tqdm, trange


ENT_NAME = ['rg', 'convert-op', 'seal-op', 'spin-op', 'create-op', \
            'd', 'default-op', 'destroy-op', 'loc', 'm', 'measure-op', 'mix-op', \
            'mod', 'mth', 'remove-op', 's', 'sl', 'temp-treat-op', 'time-op', \
            'transfer-op', 'wash-op']

ENT_NAME_START = []

for item in ENT_NAME:
    ENT_NAME_START.append("[ARG1-" + item + "-START]")

for item in ENT_NAME:
    ENT_NAME_START.append("[ARG2-" + item + "-START]")


ENT_NAME_END = []

for item in ENT_NAME:
    ENT_NAME_END.append("[ARG1-" + item + "-END]")

for item in ENT_NAME:
    ENT_NAME_END.append("[ARG2-" + item + "-END]")

ENT_ID = []

for i in range(300):
    ENT_ID.append("[T" + str(i) + "]")


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


def combine_data_in_doc(doc_data, tokenizer):
    sen_list = [item1 for item in doc_data['sentences'] for item1 in item]
    ner_list = sorted([item1 for item in doc_data['ner'] for item1 in item if item1[-1] != "ignored"],
                      key=lambda x: x[0])
    rel_list = [item1 for item in doc_data['relations'] for item1 in item]

    doc_sen_len_list = [len(item) for item in doc_data['sentences']]
    doc_sen_len_list_tkn = [len(tokenizer.tokenize(f"[CLS] {' '.join(item)} [SEP]")) - 2 for item in
                            doc_data['sentences']]

    return sen_list, ner_list, rel_list, doc_sen_len_list, doc_sen_len_list_tkn


def combine_data_in_doc_infer(doc_data):
    sen_list = [item1 for item in doc_data['sentences_tokenized'] for item1 in item]
    ner_list = sorted([item1 for item in doc_data['tl_ner_tokenized'] for item1 in item if item1[-1] != "ignored"],
                      key=lambda x: x[0])
    rel_list = [item1 for item in doc_data['relations_tokenized'] for item1 in item]

    doc_sen_len_list = [len(item) for item in doc_data['sentences_tokenized']]
    doc_sen_len_list_tkn = doc_sen_len_list


    return sen_list, ner_list, rel_list, doc_sen_len_list, doc_sen_len_list_tkn


def mask_all_entities(tokens, sorted_entity_list):
    ent_dict = dict([(item[2][0], item) for item in sorted_entity_list])

    word_idx = 0

    tagged_token_list = []

    while word_idx < len(tokens):

        if word_idx not in ent_dict:
            tagged_token_list.append(tokens[word_idx])
            word_idx += 1
        else:
            ent_id, ent_str, (ent_start, ent_end), ent_type, _ = ent_dict[word_idx]

            if ent_str.strip() != ' '.join(tokens[ent_start:ent_end]).strip():
                print(ent_str, ' '.join(tokens[ent_start:ent_end]))

            assert ent_str.strip() == ' '.join(tokens[ent_start:ent_end]).strip()

            tagged_token_list.append("[ARG1-" + ent_type + "-START]")
            tagged_token_list.append("[" + ent_id + "]")
            tagged_token_list.append(ent_str)
            tagged_token_list.append("[ARG1-" + ent_type + "-END]")
            word_idx = ent_end

    return tagged_token_list


def get_pos_rel_data(rel_list, sel_rel, sen_list, entstart2entid, entid2ent, rel2ep_type_dict, \
                     word2sen_idx, verbose=True):
    pos_rel_list = []
    pos_ep_dict = {}

    sel_ep_type = []

    for rel_str in sel_rel:
        sel_ep_type += list(rel2ep_type_dict[rel_str])

    sel_ep_type_dict = dict([(item, 0) for item in sel_ep_type])

    for arg1_start, arg1_end, arg2_start, arg2_end, rel_str in rel_list:

        if rel_str not in sel_rel:
            #             print(rel_str)
            continue

        arg1_type = entid2ent[entstart2entid[arg1_start]][-1]
        arg2_type = entid2ent[entstart2entid[arg2_start]][-1]
        #         print('-'.join([arg1_type, arg2_type]))

        # Adjust end index
        arg1_end = arg1_end + 1
        arg2_end = arg2_end + 1

        arg1_id = entstart2entid[arg1_start]
        arg2_id = entstart2entid[arg2_start]

        arg1_type = entid2ent[arg1_id][-1]
        arg2_type = entid2ent[arg2_id][-1]

        arg1_offset = (arg1_start, arg1_end)
        arg2_offset = (arg2_start, arg2_end)

        arg1_str = ' '.join(sen_list[arg1_start:arg1_end])
        arg2_str = ' '.join(sen_list[arg2_start:arg2_end])

        arg1 = (arg1_id, arg1_str, arg1_offset, arg1_type, word2sen_idx[arg1_start])
        arg2 = (arg2_id, arg2_str, arg2_offset, arg2_type, word2sen_idx[arg2_start])

        if verbose:
            print(rel_str, arg1, arg2)

        pos_rel_list.append((rel_str, arg1, arg2))
        pos_ep_dict['-'.join([arg1_id, arg2_id])] = 0

    return pos_rel_list, pos_ep_dict


def get_neg_rel_data(entid2ent, entstart2entid, sen_list, pos_ep_dict, sel_rel, rel2ep_type_dict, \
                     word2sen_idx, verbose=True):
    neg_rel_list = []

    sel_ep_type = []

    for rel_str in sel_rel:
        sel_ep_type += list(rel2ep_type_dict[rel_str])

    #     print(sel_ep_type)
    sel_ep_type_dict = dict([(item, 0) for item in sel_ep_type])
    #     print(sel_ep_type_dict)

    within_sen_ep_type = []
    for rel_str in WITHIN_SEN_RELS:
        within_sen_ep_type += list(rel2ep_type_dict[rel_str])

    within_sen_ep_type = dict([(item, 0) for item in within_sen_ep_type])

    for ent1_id in entid2ent.keys():
        for ent2_id in entid2ent.keys():
            if ent1_id != ent2_id and '-'.join([ent1_id, ent2_id]) not in pos_ep_dict:

                ent1_start, ent1_end, ent1_type = entid2ent[ent1_id]
                ent2_start, ent2_end, ent2_type = entid2ent[ent2_id]

                if '-'.join([ent1_type, ent2_type]) not in sel_ep_type_dict:
                    continue
                #                 print('-'.join([ent1_type, ent2_type]))

                if '-'.join([ent1_type, ent2_type]) in within_sen_ep_type and \
                        word2sen_idx[ent1_start] != word2sen_idx[ent2_start]:
                    continue
                #                 else:

                ent1_end = ent1_end + 1
                ent2_end = ent2_end + 1

                ent1_offset = (ent1_start, ent1_end)
                ent2_offset = (ent2_start, ent2_end)

                ent1_str = ' '.join(sen_list[ent1_start:ent1_end])
                ent2_str = ' '.join(sen_list[ent2_start:ent2_end])

                #                 ent1 = (ent1_id, ent1_str, ent1_offset, ent1_type, 0)
                #                 ent2 = (ent2_id, ent2_str, ent2_offset, ent2_type, 0)
                ent1 = (ent1_id, ent1_str, ent1_offset, ent1_type, word2sen_idx[ent1_start])
                ent2 = (ent2_id, ent2_str, ent2_offset, ent2_type, word2sen_idx[ent2_start])

                neg_rel_list.append(('no_relation', ent1, ent2))

    return neg_rel_list


def purify_word_list(tokens, arg1id, arg2id):
    word_idx = 0
    arg_tag = 0
    purified_token_list = []

    while word_idx < len(tokens):

        if tokens[word_idx][-len('-START]'):] == '-START]':
            if tokens[word_idx + 1] == '[' + arg1id + ']':
                purified_token_list.append(tokens[word_idx])
                arg_tag = 1
            elif tokens[word_idx + 1] == '[' + arg2id + ']':
                purified_token_list.append(tokens[word_idx][:len('[ARG')] + '2' + tokens[word_idx][len('[ARG1'):])
                arg_tag = 2
            word_idx += 2
        elif tokens[word_idx][-len('-END]'):] == '-END]':
            if arg_tag:
                purified_token_list.append(
                    tokens[word_idx][:len('[ARG')] + str(arg_tag) + tokens[word_idx][len('[ARG1'):])
                arg_tag = 0
            word_idx += 1
        else:
            purified_token_list.append(tokens[word_idx])
            word_idx += 1

    return purified_token_list


def get_data_per_mention(word_list, pos_pairs_list, neg_pairs_list, org_cxt_num=10, down_sample=False):
    purified_texts, relation_list = [], []
    arg1_ent_start_tkn_list = []
    arg2_ent_start_tkn_list = []
    arg_info_list = []

    if not down_sample:
        pairs_list_all = pos_pairs_list + neg_pairs_list
    else:
        random.seed(1234)
        neg_pairs_list_new = random.sample(neg_pairs_list, int(0.4 * len(neg_pairs_list)))
        pairs_list_all = pos_pairs_list * 2 + neg_pairs_list_new

    for relation, arg1, arg2 in pairs_list_all:


        arg1_id, arg1_str, arg1_offset, arg1_type, arg1_senid = arg1
        arg2_id, arg2_str, arg2_offset, arg2_type, arg2_senid = arg2

        assert "[{}]".format(arg1_id) in word_list
        assert "[{}]".format(arg2_id) in word_list

        arg1_ent_start_tkn = "[ARG1-" + arg1_type + "-START]"
        arg1_ent_end_tkn = "[ARG1-" + arg1_type + "-END]"

        arg2_ent_start_tkn = "[ARG2-" + arg2_type + "-START]"
        arg2_ent_end_tkn = "[ARG2-" + arg2_type + "-END]"

        arg_offset = [arg1_offset, arg2_offset]

        purified_word_list = purify_word_list(word_list, arg1_id, arg2_id)

        arg1_start_idx = purified_word_list.index(arg1_ent_start_tkn)
        arg1_end_idx = purified_word_list.index(arg1_ent_end_tkn)
        arg2_start_idx = purified_word_list.index(arg2_ent_start_tkn)
        arg2_end_idx = purified_word_list.index(arg2_ent_end_tkn)

        last_idx = max(arg1_end_idx, arg2_end_idx)
        first_idx = min(arg1_start_idx, arg2_start_idx)

        if last_idx - first_idx > 509:
            continue

        cxt_num = math.floor((510 - (last_idx - first_idx + 1)) / 2) \
            if org_cxt_num > math.floor((510 - (last_idx - first_idx + 1)) / 2) else org_cxt_num

        if first_idx < cxt_num:
            purified_word_list = [purified_word_list[0]] + purified_word_list[:last_idx + cxt_num] + \
                                 [purified_word_list[-1]]

        else:
            purified_word_list = [purified_word_list[0]] + purified_word_list[first_idx - cxt_num:last_idx + cxt_num] + \
                                 [purified_word_list[-1]]

        purified_texts.append(purified_word_list)
        arg1_ent_start_tkn_list.append(arg1_ent_start_tkn)
        arg2_ent_start_tkn_list.append(arg2_ent_start_tkn)
        arg_info_list.append([relation, arg1, arg2])

        relation_list.append(relation)

    assert len(purified_texts) == len(arg1_ent_start_tkn_list) and \
           len(purified_texts) == len(arg2_ent_start_tkn_list)

    return arg1_ent_start_tkn_list, arg2_ent_start_tkn_list, purified_texts, relation_list, arg_info_list


def find_word_sen_idx(doc_sen_len):
    doc_len = 0
    word2sen_idx = {}

    for sen_idx, sen_len in enumerate(doc_sen_len):

        for i in range(doc_len, doc_len + sen_len):
            word2sen_idx[i] = sen_idx

        doc_len += sen_len

    return word2sen_idx


def generate_rel_data_per_doc(sen_list, ner_list, rel_list, doc_sen_len_list, sel_rel, rel2ep_type_dict, \
                              ep_type2rel_dict, tokenizer, down_sample=False, verbose=True):
    # Build idx2ent dictionary
    entid2ent = dict([('T' + str(item_idx), item) for item_idx, item in enumerate(ner_list)])
    entstart2entid = dict([(item[0], 'T' + str(item_idx)) for item_idx, item in enumerate(ner_list)])

    word2sen_idx = find_word_sen_idx(doc_sen_len_list)

    ent_list = [(entstart2entid[ent_start], ' '.join(sen_list[ent_start:ent_end + 1]),
                 (ent_start, ent_end + 1), ent_type, '') for ent_start, ent_end, ent_type in ner_list]

    processed_word_list = mask_all_entities(sen_list, ent_list)
    #     print(' '.join(processed_word_list))

    # Get positive relations
    pos_rel_list, pos_ep_dict = get_pos_rel_data(rel_list, sel_rel, sen_list, entstart2entid, entid2ent, \
                                                 rel2ep_type_dict, word2sen_idx, False)

    # Get negative relations
    neg_rel_list = get_neg_rel_data(entid2ent, entstart2entid, sen_list, pos_ep_dict, sel_rel, \
                                    rel2ep_type_dict, word2sen_idx)

    tokenized_word_list = tokenizer.tokenize("[CLS] " + ' '.join(processed_word_list) + " [SEP]")

    arg1_ent_start_tkn_list, arg2_ent_start_tkn_list, \
    purified_texts, relation_list, arg_info_list = get_data_per_mention(tokenized_word_list, pos_rel_list, \
                                                                        neg_rel_list, down_sample=down_sample)

    return arg1_ent_start_tkn_list, arg2_ent_start_tkn_list, purified_texts, relation_list, arg_info_list


def prepare_data(doc_data_list, sel_rel, rel2ep_type_dict, ep_type2rel_dict, tokenizer, args, down_sample=False):
    start_time = time.time()

    arg1_ent_start_tkn_list_all = []
    arg2_ent_start_tkn_list_all = []
    purified_texts_all = []
    relation_list_all = []
    arg_info_list_all = []
    doc_name_list_all = []

    for doc_idx, doc_data in enumerate(doc_data_list):
        # print(doc_idx, doc_data["doc_key"])

        sen_list, ner_list, rel_list, \
        doc_sen_len_list, doc_sen_len_list_tkn = combine_data_in_doc(doc_data, tokenizer)

        arg1_ent_start_tkn_list_per_doc, arg2_ent_start_tkn_list_per_doc, \
        purified_texts_per_doc, relation_list_per_doc, \
        arg_info_list_per_doc = generate_rel_data_per_doc(sen_list, ner_list, rel_list, \
                                                          doc_sen_len_list, \
                                                          sel_rel, rel2ep_type_dict, ep_type2rel_dict, \
                                                          tokenizer, down_sample=down_sample)

        arg1_ent_start_tkn_list_all += arg1_ent_start_tkn_list_per_doc
        arg2_ent_start_tkn_list_all += arg2_ent_start_tkn_list_per_doc
        purified_texts_all += purified_texts_per_doc
        relation_list_all += relation_list_per_doc
        arg_info_list_all += arg_info_list_per_doc
        doc_name_list_all += [doc_data["doc_key"]] * len(arg_info_list_per_doc)


    # Get the input_ids and labels
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in purified_texts_all],
                              maxlen=args.max_len, value=tokenizer.pad_token_id, dtype="long", \
                              truncating="post", padding="post")

    attention_masks = [[float(i != tokenizer.pad_token_id) for i in ii] for ii in input_ids]


    labels = [LABEL2ID[l] for l in relation_list_all]

    arg1_idx_list = [purified_texts_all[sen_idx].index(arg1_ent_start_tkn_list_all[sen_idx])
                     if purified_texts_all[sen_idx].index(
        arg1_ent_start_tkn_list_all[sen_idx]) < args.max_len else args.max_len - 1
                     for sen_idx in range(len(purified_texts_all))
                     ]

    arg2_idx_list = [purified_texts_all[sen_idx].index(arg2_ent_start_tkn_list_all[sen_idx])
                     if purified_texts_all[sen_idx].index(
        arg2_ent_start_tkn_list_all[sen_idx]) < args.max_len else args.max_len - 1
                     for sen_idx in range(len(purified_texts_all))
                     ]

    assert len(input_ids) == len(attention_masks) and len(labels) == len(arg1_idx_list)


    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)
    arg1_idx = torch.tensor(arg1_idx_list)
    arg2_idx = torch.tensor(arg2_idx_list)

    print("--- %s seconds ---" % (time.time() - start_time))

    return inputs, masks, labels, arg1_idx, arg2_idx, arg_info_list_all, doc_name_list_all


def score(key, prediction, verbose=False):
        correct_by_relation = Counter()
        guessed_by_relation = Counter()
        gold_by_relation = Counter()

        # Loop over the data to compute a score
        for row in range(len(key)):
            gold = key[row]
            guess = prediction[row]

            if gold == NO_RELATION and guess == NO_RELATION:
                pass
            elif gold == NO_RELATION and guess != NO_RELATION:
                guessed_by_relation[guess] += 1
            elif gold != NO_RELATION and guess == NO_RELATION:
                gold_by_relation[gold] += 1
            elif gold != NO_RELATION and guess != NO_RELATION:
                guessed_by_relation[guess] += 1
                gold_by_relation[gold] += 1
                if gold == guess:
                    correct_by_relation[guess] += 1

        # Print verbose information
        if verbose:
            print("Per-relation statistics:")
            relations = gold_by_relation.keys()
            longest_relation = 0
            for relation in sorted(relations):
                longest_relation = max(len(relation), longest_relation)
            for relation in sorted(relations):
                # (compute the score)
                correct = correct_by_relation[relation]
                guessed = guessed_by_relation[relation]
                gold = gold_by_relation[relation]
                prec = 1.0
                if guessed > 0:
                    prec = float(correct) / float(guessed)
                recall = 0.0
                if gold > 0:
                    recall = float(correct) / float(gold)
                f1 = 0.0
                if prec + recall > 0:
                    f1 = 2.0 * prec * recall / (prec + recall)
                # (print the score)
                sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
                sys.stdout.write("  P: ")
                if prec < 0.1: sys.stdout.write(' ')
                if prec < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.1%}".format(prec))
                sys.stdout.write("  R: ")
                if recall < 0.1: sys.stdout.write(' ')
                if recall < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.1%}".format(recall))
                sys.stdout.write("  F1: ")
                if f1 < 0.1: sys.stdout.write(' ')
                if f1 < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.1%}".format(f1))
                sys.stdout.write("  #: %d" % gold)
                sys.stdout.write("\n")
            print("")

        # Print the aggregate score
        if verbose:
            print("Final Score:")
        prec_micro = 1.0
        if sum(guessed_by_relation.values()) > 0:
            prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
        recall_micro = 0.0
        if sum(gold_by_relation.values()) > 0:
            recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
        f1_micro = 0.0
        if prec_micro + recall_micro > 0.0:
            f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
        if verbose:
            print("Precision (micro): {:.1%}".format(prec_micro))
            print("   Recall (micro): {:.1%}".format(recall_micro))
            print("       F1 (micro): {:.1%}".format(f1_micro))
        return prec_micro, recall_micro, f1_micro


def generate_rel_mention(pred_data):
        gold_tags = pred_data['gold_tags']
        pred_tags = pred_data['pred_tags']
        arg_info_list = pred_data['arg_info_list']
        doc_name_list = pred_data['doc_name_list']

        doc_gold_rel_dict = defaultdict(list)
        doc_pred_rel_dict = defaultdict(list)

        for gold_tag, pred_tag, arg_info, doc_name in zip(gold_tags, pred_tags, arg_info_list, doc_name_list):

            rel_str, arg1_info, arg2_info = arg_info
            assert rel_str == gold_tag
            #     print(gold_tag == arg_info[0])

            arg1_id, arg1_str, arg1_offset, arg1_type, arg1_sen_idx = arg1_info
            arg2_id, arg2_str, arg2_offset, arg2_type, arg2_sen_idx = arg2_info

            arg1_offset = [arg1_offset[0], arg1_offset[1] - 1]
            arg2_offset = [arg2_offset[0], arg2_offset[1] - 1]

            if gold_tag != 'no_relation':
                #         print(arg1_offset + arg2_offset + [gold_tag])
                doc_gold_rel_dict[doc_name].append(arg1_offset + arg2_offset + [gold_tag])

            if pred_tag != 'no_relation':
                doc_pred_rel_dict[doc_name].append(arg1_offset + arg2_offset + [pred_tag])

        return doc_gold_rel_dict, doc_pred_rel_dict


def score_sel_rel(key, prediction, sel_rel, verbose=False):
        correct_by_relation = Counter()
        guessed_by_relation = Counter()
        gold_by_relation = Counter()

        # Loop over the data to compute a score
        for row in range(len(key)):
            gold = key[row]
            guess = prediction[row]

            if gold == NO_RELATION and guess == NO_RELATION:
                pass
            elif gold == NO_RELATION and guess != NO_RELATION:
                guessed_by_relation[guess] += 1
            elif gold != NO_RELATION and guess == NO_RELATION:
                gold_by_relation[gold] += 1
            elif gold != NO_RELATION and guess != NO_RELATION:
                guessed_by_relation[guess] += 1
                gold_by_relation[gold] += 1
                if gold == guess:
                    correct_by_relation[guess] += 1

        correct_all = 0
        guessed_all = 0
        gold_all = 0

        # Print verbose information
        if verbose:
            print("Per-relation statistics:")
            relations = gold_by_relation.keys()
            longest_relation = 0
            for relation in sorted(relations):
                longest_relation = max(len(relation), longest_relation)
            for relation in sorted(relations):
                # (compute the score)

                if not (relation in sel_rel):
                    # print(relations)
                    continue

                correct = correct_by_relation[relation]
                guessed = guessed_by_relation[relation]
                gold = gold_by_relation[relation]

                correct_all += correct
                guessed_all += guessed
                gold_all += gold

                prec = 1.0
                if guessed > 0:
                    prec = float(correct) / float(guessed)
                recall = 0.0
                if gold > 0:
                    recall = float(correct) / float(gold)
                f1 = 0.0
                if prec + recall > 0:
                    f1 = 2.0 * prec * recall / (prec + recall)
                # (print the score)
                sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
                sys.stdout.write("  P: ")
                if prec < 0.1: sys.stdout.write(' ')
                if prec < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.2%}".format(prec))
                sys.stdout.write("  R: ")
                if recall < 0.1: sys.stdout.write(' ')
                if recall < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.2%}".format(recall))
                sys.stdout.write("  F1: ")
                if f1 < 0.1: sys.stdout.write(' ')
                if f1 < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.2%}".format(f1))
                sys.stdout.write("  #: %d" % gold)

                sys.stdout.write("  TP: ")
                if f1 < 0.1: sys.stdout.write(' ')
                if f1 < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{}".format(correct))

                sys.stdout.write("  FP: ")
                if f1 < 0.1: sys.stdout.write(' ')
                if f1 < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{}".format(guessed - correct))

                sys.stdout.write("  FN: ")
                if f1 < 0.1: sys.stdout.write(' ')
                if f1 < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{}".format(gold - correct))

                sys.stdout.write("\n")
            print("")

        # Print the aggregate score
        if verbose:
            print("Final Score:")
        prec_micro = 1.0
        # if sum(guessed_by_relation.values()) > 0:
        if sum([guessed_by_relation[rel] for rel in sel_rel]) > 0:
            prec_micro = float(correct_all) / float(guessed_all)
        recall_micro = 0.0
        # if sum(gold_by_relation.values()) > 0:
        if sum([gold_by_relation[rel] for rel in sel_rel]) > 0:
            recall_micro = float(correct_all) / float(gold_all)
        f1_micro = 0.0
        if prec_micro + recall_micro > 0.0:
            f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
        if verbose:
            print("Precision (micro): {:.1%}".format(prec_micro))
            print("   Recall (micro): {:.1%}".format(recall_micro))
            print("       F1 (micro): {:.1%}".format(f1_micro))
            # print("TP: {}".format(correct_all))
            # print("FP: {}".format(guessed_all - correct_all))
            # print("FN: {}".format(gold_all - correct_all))
            print("# gold: {}".format(gold_all))
        return prec_micro, recall_micro, f1_micro


def combine_pred_data_in_doc(doc_data):
    sen_list = [item1 for item in doc_data['sentences_tokenized'] for item1 in item]
    ner_list = sorted(
        [item1 for item in doc_data['tl_ner_pred_tokenized'] for item1 in item if item1[-1] != "ignored"],
        key=lambda x: x[0])
    #     rel_list = [item1 for item in doc_data['relations_tokenized'] for item1 in item]
    rel_list = []

    doc_sen_len_list = [len(item) for item in doc_data['sentences_tokenized']]
    doc_sen_len_list_tkn = doc_sen_len_list


    return sen_list, ner_list, rel_list, doc_sen_len_list, doc_sen_len_list_tkn


def prepare_inference_data(doc_data_list, sel_rel, rel2ep_type_dict, ep_type2rel_dict, tokenizer, args,
                           down_sample=False):
    start_time = time.time()

    arg1_ent_start_tkn_list_all = []
    arg2_ent_start_tkn_list_all = []
    purified_texts_all = []
    relation_list_all = []
    arg_info_list_all = []
    doc_name_list_all = []

    for doc_idx, doc_data in enumerate(doc_data_list):
        # print(doc_idx, doc_data["doc_key"])

        sen_list, ner_list, rel_list, \
        doc_sen_len_list, doc_sen_len_list_tkn = combine_pred_data_in_doc(doc_data)

        arg1_ent_start_tkn_list_per_doc, arg2_ent_start_tkn_list_per_doc, \
        purified_texts_per_doc, relation_list_per_doc, \
        arg_info_list_per_doc = generate_rel_data_per_doc(sen_list, ner_list, rel_list, \
                                                          doc_sen_len_list, \
                                                          sel_rel, rel2ep_type_dict, ep_type2rel_dict, \
                                                          tokenizer, down_sample=down_sample)

        arg1_ent_start_tkn_list_all += arg1_ent_start_tkn_list_per_doc
        arg2_ent_start_tkn_list_all += arg2_ent_start_tkn_list_per_doc
        purified_texts_all += purified_texts_per_doc
        relation_list_all += relation_list_per_doc
        arg_info_list_all += arg_info_list_per_doc
        doc_name_list_all += [doc_data["doc_key"]] * len(arg_info_list_per_doc)

    # Get the input_ids and labels
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in purified_texts_all],
                              maxlen=args.max_len, value=tokenizer.pad_token_id, dtype="long", \
                              truncating="post", padding="post")

    attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]


    labels = [LABEL2ID[l] for l in relation_list_all]

    arg1_idx_list = [purified_texts_all[sen_idx].index(arg1_ent_start_tkn_list_all[sen_idx])
                     if purified_texts_all[sen_idx].index(
        arg1_ent_start_tkn_list_all[sen_idx]) < args.max_len else args.max_len - 1
                     for sen_idx in range(len(purified_texts_all))
                     ]

    arg2_idx_list = [purified_texts_all[sen_idx].index(arg2_ent_start_tkn_list_all[sen_idx])
                     if purified_texts_all[sen_idx].index(
        arg2_ent_start_tkn_list_all[sen_idx]) < args.max_len else args.max_len - 1
                     for sen_idx in range(len(purified_texts_all))
                     ]

    assert len(input_ids) == len(attention_masks) and len(labels) == len(arg1_idx_list)

    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)
    arg1_idx = torch.tensor(arg1_idx_list)
    arg2_idx = torch.tensor(arg2_idx_list)

    print("--- %s seconds ---" % (time.time() - start_time))

    return inputs, masks, labels, arg1_idx, arg2_idx, arg_info_list_all, doc_name_list_all


def score_by_relation(pred_rel_dict, gold_rel_dict, verbose=False):
        correct_by_relation = Counter()
        guessed_by_relation = Counter()
        gold_by_relation = Counter()

        # Loop over the data to compute a score
        for doc_name in pred_rel_dict.keys():

            gold_rel_list = gold_rel_dict[doc_name]
            pred_rel_list = pred_rel_dict[doc_name]

            gold_rel_str_list = [' '.join([str(item1) for item1 in item[:]]) \
                                 for item in gold_rel_list if item[-1] != 'no_relation']
            pred_rel_str_list = [' '.join([str(item1) for item1 in item[:]]) \
                                 for item in pred_rel_list if item[-1] != 'no_relation']

            #         print(len(set(gold_rel_str_list)), len(gold_rel_str_list))
            #         assert len(set(gold_rel_str_list)) == len(gold_rel_str_list)
            #         assert len(set(pred_rel_str_list)) == len(pred_rel_str_list)

            for item in set(gold_rel_str_list) & set(pred_rel_str_list):

                rel_str = item.split(" ")[-1]
                #             print(item, rel_str)

                if rel_str != 'no_relation':
                    guessed_by_relation[rel_str] += 1
                    gold_by_relation[rel_str] += 1
                    correct_by_relation[rel_str] += 1

            for item in set([item for item in gold_rel_str_list if item not in pred_rel_str_list]):

                rel_str = item.split(" ")[-1]
                #             print(item, rel_str)

                if rel_str != 'no_relation':
                    gold_by_relation[rel_str] += 1

            for item in set([item for item in pred_rel_str_list if item not in gold_rel_str_list]):

                rel_str = item.split(" ")[-1]
                #             print(item, rel_str)

                if rel_str != 'no_relation':
                    guessed_by_relation[rel_str] += 1

        # Print verbose information
        if verbose:
            print("Per-relation statistics:")
            relations = gold_by_relation.keys()
            longest_relation = 0
            for relation in sorted(relations):
                longest_relation = max(len(relation), longest_relation)
            for relation in sorted(relations):
                # (compute the score)
                correct = correct_by_relation[relation]
                guessed = guessed_by_relation[relation]
                gold = gold_by_relation[relation]
                prec = 1.0
                if guessed > 0:
                    prec = float(correct) / float(guessed)
                recall = 0.0
                if gold > 0:
                    recall = float(correct) / float(gold)
                f1 = 0.0
                if prec + recall > 0:
                    f1 = 2.0 * prec * recall / (prec + recall)
                # (print the score)
                sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
                sys.stdout.write("  P: ")
                if prec < 0.1: sys.stdout.write(' ')
                if prec < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.1%}".format(prec))
                sys.stdout.write("  R: ")
                if recall < 0.1: sys.stdout.write(' ')
                if recall < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.1%}".format(recall))
                sys.stdout.write("  F1: ")
                if f1 < 0.1: sys.stdout.write(' ')
                if f1 < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.1%}".format(f1))
                sys.stdout.write("  #: %d" % gold)

                sys.stdout.write("  TP: ")
                if f1 < 0.1: sys.stdout.write(' ')
                if f1 < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{}".format(correct))

                sys.stdout.write("  FP: ")
                if f1 < 0.1: sys.stdout.write(' ')
                if f1 < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{}".format(guessed - correct))

                sys.stdout.write("  FN: ")
                if f1 < 0.1: sys.stdout.write(' ')
                if f1 < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{}".format(gold - correct))

                sys.stdout.write("\n")
            print("")

        # Print the aggregate score
        if verbose:
            print("Final Score:")
        prec_micro = 1.0
        if sum(guessed_by_relation.values()) > 0:
            prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
        recall_micro = 0.0
        if sum(gold_by_relation.values()) > 0:
            recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
        f1_micro = 0.0
        if prec_micro + recall_micro > 0.0:
            f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
        if verbose:
            print("Precision (micro): {:.1%}".format(prec_micro))
            print("   Recall (micro): {:.1%}".format(recall_micro))
            print("       F1 (micro): {:.1%}".format(f1_micro))
        return prec_micro, recall_micro, f1_micro, \
               sum(correct_by_relation.values()), sum(guessed_by_relation.values()), \
               sum(gold_by_relation.values())


def score_by_distance(data_all, pred_rel_dict, gold_rel_dict, sel_rel, verbose=False):
        correct_by_distance = Counter()
        guessed_by_distance = Counter()
        gold_by_distance = Counter()

        # Loop over the data to compute a score
        for doc_data in data_all:

            doc_name = doc_data['doc_key']

            if not (doc_name in pred_rel_dict and doc_name in gold_rel_dict):
                raise

            sen_list, ner_list, rel_list, \
            doc_sen_len_list, doc_sen_len_list_tkn = combine_data_in_doc_infer(doc_data)

            word2sen_idx = find_word_sen_idx(doc_sen_len_list)

            gold_rel_list = gold_rel_dict[doc_name]
            pred_rel_list = pred_rel_dict[doc_name]

            # gold_rel_str_list = [' '.join([str(item1) for item1 in item[:]]) \
            #                      for item in gold_rel_list if item[-1] != 'no_relation']
            # pred_rel_str_list = [' '.join([str(item1) for item1 in item[:]]) \
            #                      for item in pred_rel_list if item[-1] != 'no_relation']

            # Only select core relations
            gold_rel_str_list = [' '.join([str(item1) for item1 in item[:]]) \
                                 for item in gold_rel_list if
                                 item[-1] != 'no_relation' and item[-1] in sel_rel]
            pred_rel_str_list = [' '.join([str(item1) for item1 in item[:]]) \
                                 for item in pred_rel_list if
                                 item[-1] != 'no_relation' and item[-1] in sel_rel]

            #         print(len(set(gold_rel_str_list)), len(gold_rel_str_list))
            #         assert len(set(gold_rel_str_list)) == len(gold_rel_str_list)
            #         assert len(set(pred_rel_str_list)) == len(pred_rel_str_list)

            for item in set(gold_rel_str_list) & set(pred_rel_str_list):

                rel_str = item.split(" ")[-1]
                arg1_start, arg1_end, arg2_start, arg2_end, rel_str = item.split(" ")
                arg1_start, arg1_end, arg2_start, arg2_end = int(arg1_start), int(arg1_end), \
                                                             int(arg2_start), int(arg2_end)
                #             print(arg1_start, arg1_end, arg2_start, arg2_end)
                #             print(abs(word2sen_idx[arg1_start] - word2sen_idx[arg2_start]))
                arg_dist = "within" if abs(word2sen_idx[arg1_start] - word2sen_idx[arg2_start]) == 0 else "cross"

                if rel_str != 'no_distance':
                    guessed_by_distance[arg_dist] += 1
                    gold_by_distance[arg_dist] += 1
                    correct_by_distance[arg_dist] += 1

            for item in set([item for item in gold_rel_str_list if item not in pred_rel_str_list]):

                rel_str = item.split(" ")[-1]
                arg1_start, arg1_end, arg2_start, arg2_end, rel_str = item.split(" ")
                arg1_start, arg1_end, arg2_start, arg2_end = int(arg1_start), int(arg1_end), \
                                                             int(arg2_start), int(arg2_end)
                #             print(arg1_start, arg1_end, arg2_start, arg2_end)
                #             print(abs(word2sen_idx[arg1_start] - word2sen_idx[arg2_start]))
                arg_dist = "within" if abs(word2sen_idx[arg1_start] - word2sen_idx[arg2_start]) == 0 else "cross"

                if rel_str != 'no_distance':
                    gold_by_distance[arg_dist] += 1

            for item in set([item for item in pred_rel_str_list if item not in gold_rel_str_list]):

                rel_str = item.split(" ")[-1]
                arg1_start, arg1_end, arg2_start, arg2_end, rel_str = item.split(" ")
                arg1_start, arg1_end, arg2_start, arg2_end = int(arg1_start), int(arg1_end), \
                                                             int(arg2_start), int(arg2_end)
                #             print(arg1_start, arg1_end, arg2_start, arg2_end)
                #             print(abs(word2sen_idx[arg1_start] - word2sen_idx[arg2_start]))
                arg_dist = "within" if abs(word2sen_idx[arg1_start] - word2sen_idx[arg2_start]) == 0 else "cross"

                if rel_str != 'no_distance':
                    guessed_by_distance[arg_dist] += 1

        # Print verbose information
        if verbose:
            # print("Per-distance statistics:")
            distances = gold_by_distance.keys()
            longest_distance = 0
            for distance in sorted(distances):
                longest_distance = max(len(distance), longest_distance)
            for distance in sorted(distances):
                # (compute the score)
                correct = correct_by_distance[distance]
                guessed = guessed_by_distance[distance]
                gold = gold_by_distance[distance]
                prec = 1.0
                if guessed > 0:
                    prec = float(correct) / float(guessed)
                recall = 0.0
                if gold > 0:
                    recall = float(correct) / float(gold)
                f1 = 0.0
                if prec + recall > 0:
                    f1 = 2.0 * prec * recall / (prec + recall)
                # (print the score)
                sys.stdout.write(("{:<" + str(longest_distance) + "}").format(distance))
                sys.stdout.write("  P: ")
                if prec < 0.1: sys.stdout.write(' ')
                if prec < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.1%}".format(prec))
                sys.stdout.write("  R: ")
                if recall < 0.1: sys.stdout.write(' ')
                if recall < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.1%}".format(recall))
                sys.stdout.write("  F1: ")
                if f1 < 0.1: sys.stdout.write(' ')
                if f1 < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.1%}".format(f1))
                sys.stdout.write("  #: %d" % gold)
                sys.stdout.write("\n")
            print("")

        # Print the aggregate score
        # if verbose:
        #     print("Final Score:")
        prec_micro = 1.0
        if sum(guessed_by_distance.values()) > 0:
            prec_micro = float(sum(correct_by_distance.values())) / float(sum(guessed_by_distance.values()))
        recall_micro = 0.0
        if sum(gold_by_distance.values()) > 0:
            recall_micro = float(sum(correct_by_distance.values())) / float(sum(gold_by_distance.values()))
        f1_micro = 0.0
        if prec_micro + recall_micro > 0.0:
            f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
        # if verbose:
        #     print("Precision (micro): {:.1%}".format(prec_micro))
        #     print("   Recall (micro): {:.1%}".format(recall_micro))
        #     print("       F1 (micro): {:.1%}".format(f1_micro))
        return prec_micro, recall_micro, f1_micro, \
               sum(correct_by_distance.values()), sum(guessed_by_distance.values()), \
               sum(gold_by_distance.values())


def score_verbose(key, prediction, verbose=False):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    if verbose:
        print("Precision (micro): {:.3%}".format(prec_micro))
        print("   Recall (micro): {:.3%}".format(recall_micro))
        print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro, \
           sum(correct_by_relation.values()), sum(guessed_by_relation.values()), \
           sum(gold_by_relation.values())


def score_sel_rel_infer(pred_rel_dict, gold_rel_dict, sel_rel, verbose=False):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for doc_name in pred_rel_dict.keys():

        gold_rel_list = gold_rel_dict[doc_name]
        pred_rel_list = pred_rel_dict[doc_name]

        gold_rel_str_list = [' '.join([str(item1) for item1 in item[:]]) \
                             for item in gold_rel_list if item[-1] != 'no_relation']
        pred_rel_str_list = [' '.join([str(item1) for item1 in item[:]]) \
                             for item in pred_rel_list if item[-1] != 'no_relation']

        for item in set(gold_rel_str_list) & set(pred_rel_str_list):

            rel_str = item.split(" ")[-1]
            #             print(item, rel_str)

            if rel_str != 'no_relation':
                guessed_by_relation[rel_str] += 1
                gold_by_relation[rel_str] += 1
                correct_by_relation[rel_str] += 1

        for item in set([item for item in gold_rel_str_list if item not in pred_rel_str_list]):

            rel_str = item.split(" ")[-1]
            #             print(item, rel_str)

            if rel_str != 'no_relation':
                gold_by_relation[rel_str] += 1

        for item in set([item for item in pred_rel_str_list if item not in gold_rel_str_list]):

            rel_str = item.split(" ")[-1]
            #             print(item, rel_str)

            if rel_str != 'no_relation':
                guessed_by_relation[rel_str] += 1

    correct_all = 0
    guessed_all = 0
    gold_all = 0

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)

            if not (relation in sel_rel):
                # print(relations)
                continue

            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold = gold_by_relation[relation]

            correct_all += correct
            guessed_all += guessed
            gold_all += gold

            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.1%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.1%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.1%}".format(f1))
            sys.stdout.write("  #: %d" % gold)

            # sys.stdout.write("  TP: ")
            # if f1 < 0.1: sys.stdout.write(' ')
            # if f1 < 1.0: sys.stdout.write(' ')
            # sys.stdout.write("{}".format(correct))
            #
            # sys.stdout.write("  FP: ")
            # if f1 < 0.1: sys.stdout.write(' ')
            # if f1 < 1.0: sys.stdout.write(' ')
            # sys.stdout.write("{}".format(guessed - correct))
            #
            # sys.stdout.write("  FN: ")
            # if f1 < 0.1: sys.stdout.write(' ')
            # if f1 < 1.0: sys.stdout.write(' ')
            # sys.stdout.write("{}".format(gold - correct))

            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if guessed_all > 0:
        prec_micro = float(correct_all) / float(guessed_all)
    recall_micro = 0.0
    if gold_all > 0:
        recall_micro = float(correct_all) / float(gold_all)
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    if verbose:
        print("Precision (micro): {:.1%}".format(prec_micro))
        print("   Recall (micro): {:.1%}".format(recall_micro))
        print("       F1 (micro): {:.1%}".format(f1_micro))
        # print("TP: {}".format(correct_all))
        # print("FP: {}".format(guessed_all - correct_all))
        # print("FN: {}".format(gold_all - correct_all))
        print("# gold: {}".format(gold_all))
    return prec_micro, recall_micro, f1_micro, correct_all, guessed_all, gold_all