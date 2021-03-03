

import math
import glob
import time
import json
import pickle
import os
import numpy as np
import operator
import time
import torch
import sys
import random
import argparse

from collections import defaultdict, Counter, OrderedDict
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from transformers import BertForTokenClassification, BertPreTrainedModel, AdamW
from sklearn.model_selection import KFold


WLP_ENT = ['Action']

ID2LABEL = ['ignored', 'rg', 'convert-op', 'seal-op', 'spin-op', 'create-op', \
            'd', 'default-op', 'destroy-op', 'loc', 'm', 'measure-op', 'mix-op', \
            'mod', 'mth', 'remove-op', 's', 'sl', 'temp-treat-op', 'time-op', \
            'transfer-op', 'wash-op']


LABEL2ID = dict((value, key) for key, value in enumerate(ID2LABEL))


NO_RELATION = 'ignored'


WLP_ENT_START = []

for item in WLP_ENT:
    WLP_ENT_START.append("[ENT-" + item + "-START]")

WLP_ENT_END = []

for item in WLP_ENT:
    WLP_ENT_END.append("[ENT-" + item + "-END]")


WLP2TL = {'Reagent': 'rg',
          'Location': 'loc',
          'Amount': 'm',
          'Modifier': 'mod',
          'Time': 's',
          'Temperature': 's',
          'Speed': 's',
          'Generic-Measure': 'm',
          'Device': 'd',
          'Concentration': 'm',
          'Seal': 'sl',
          'Method': 'mth',
          'Size': 'm',
          'Measure-Type': 'm',
          'pH': 'm'}

NONE_TYPE = ['Numerical', 'Mention', 'Misc']

def load_from_jsonl(file_name):
    data_list = []
    with open(file_name) as f:
        for line in f:
            data_list.append(json.loads(line))

    return data_list


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
    return prec_micro, recall_micro, f1_micro


def handle_doc_offset_ner(ner_tuple, doc_len, remove=True):
    # print(ner_tuple)
    ner_start, ner_end, ent_type = ner_tuple

    if remove:
        return [ner_start - doc_len, ner_end - doc_len, ent_type]
    else:
        return [ner_start + doc_len, ner_end + doc_len, ent_type]


def build_entity_typing_mention(tokens, sorted_entity_list):
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

            tagged_token_list.append("[ENT-" + ent_type + "-START]")
            tagged_token_list.append(ent_str)
            tagged_token_list.append("[ENT-" + ent_type + "-END]")
            word_idx = ent_end

    return tagged_token_list


def prepare_entity_typing_data(tl_data, tokenizer, args):
        start_time = time.time()

        wlp_ent_type_list = []
        tl_ent_type_list = []
        process_sen_list = []
        doc_name_list = []
        sen_idx_list = []
        ent_info_list = []

        for file_idx in range(len(tl_data)):

            doc_name = tl_data[file_idx]['doc_key']

            doc_data = tl_data[file_idx]
            doc_sen_len_list = [len(item) for item in doc_data['sentences']]
            ner_list = doc_data['wlp_labels']

            for sen_idx, (tl_sen, wlp_ner, tl_ner) in enumerate(zip(doc_data['sentences'], ner_list, doc_data['ner'])):

                pre_doc_len = sum(doc_sen_len_list[:sen_idx])
                tl_ner_remove_offset = [handle_doc_offset_ner(item, pre_doc_len, remove=True) for item in tl_ner]

                assert len(tl_ner) == len(wlp_ner)

                for tl_ent, wlp_label, tl_ent_old in zip(tl_ner_remove_offset, wlp_ner, tl_ner):

                    if wlp_label != 'Action':
                        continue

                    ent_start, ent_end, tl_label = tl_ent
                    ent_start_old, ent_end_old, tl_label_old = tl_ent_old
                    assert tl_label == tl_label_old
                    assert ent_start_old == ent_start + pre_doc_len

                    one_ent_list = [('ent_id', ' '.join(tl_sen[ent_start:ent_end + 1]), \
                                     (ent_start, ent_end + 1), wlp_label, '')]

                    process_sen = build_entity_typing_mention(tl_sen, one_ent_list)

                    wlp_ent_type_list.append("[ENT-" + wlp_label + "-START]")
                    tl_ent_type_list.append(tl_label)
                    # process_sen_list.append("[CLS] " + ' '.join(process_sen) + " [SEP]")
                    process_sen_list.append(f"{tokenizer.bos_token} " + ' '.join(process_sen) + f" {tokenizer.eos_token}")
                    doc_name_list.append(doc_name)
                    sen_idx_list.append(sen_idx)
                    ent_info_list.append((ent_start_old, ent_end_old, ent_start, ent_end, sen_idx, tl_label))

        tokenized_sen_list = [tokenizer.tokenize(sent) for sent in process_sen_list]
        print(max([len(item) for item in tokenized_sen_list]))

        # Get the input_ids and labels
        input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_sen_list],
                                  maxlen=args.max_len, value=tokenizer.pad_token_id, dtype="long", truncating="post", padding="post")

        attention_masks = [[float(i != tokenizer.pad_token_id) for i in ii] for ii in input_ids]

        # print(tl_ent_type_list)
        labels = [LABEL2ID[l] for l in tl_ent_type_list]

        start_tkn_idx_list = [tokenized_sen_list[sen_idx].index(wlp_ent_type_list[sen_idx])
                              if tokenized_sen_list[sen_idx].index(
            wlp_ent_type_list[sen_idx]) < args.max_len else args.max_len - 1
                              for sen_idx in range(len(tokenized_sen_list))
                              ]

        assert len(input_ids) == len(attention_masks) and len(labels) == len(start_tkn_idx_list)

        inputs = torch.tensor(input_ids)
        masks = torch.tensor(attention_masks)
        labels = torch.tensor(labels)
        start_idx = torch.tensor(start_tkn_idx_list)

        print("--- %s seconds ---" % (time.time() - start_time))

        return inputs, masks, labels, start_idx, tokenized_sen_list, wlp_ent_type_list, doc_name_list, ent_info_list


def prepare_ep_inference_data(tl_data, tokenizer, args):
    start_time = time.time()

    wlp_ent_type_list = []
    tl_ent_type_list = []
    process_sen_list = []
    doc_name_list = []
    sen_idx_list = []
    ent_info_list = []

    for file_idx in range(len(tl_data)):

        doc_name = tl_data[file_idx]['doc_key']
        # print(file_idx, doc_name)

        doc_data = tl_data[file_idx]
        doc_sen_len_list = [len(item) for item in doc_data['sentences_tokenized']]

        for sen_idx, (tl_sen, tl_ner, wlp_ner, wlp_ner_pred) in enumerate(zip(doc_data['sentences_tokenized'], \
                                                                              doc_data['tl_ner_tokenized'], \
                                                                              doc_data['wlp_ner_tokenized'], \
                                                                              doc_data['wlp_ner_pred_tokenized'])):


            pre_doc_len = sum(doc_sen_len_list[:sen_idx])
            wlp_ner_pred_remove_offset = [handle_doc_offset_ner(item, pre_doc_len, remove=True) for item in
                                          wlp_ner_pred]

            assert len(tl_ner) == len(wlp_ner)

            for wlp_ent, wlp_ent_old in zip(wlp_ner_pred_remove_offset, wlp_ner_pred):

                ent_start, ent_end, wlp_label_pred = wlp_ent
                ent_start_old, ent_end_old, wlp_label_pred_old = wlp_ent_old
                tl_label = 'ignored'

                if wlp_label_pred != 'Action':
                    continue

                one_ent_list = [('ent_id', ' '.join(tl_sen[ent_start:ent_end + 1]), \
                                 (ent_start, ent_end + 1), wlp_label_pred, '')]

                process_sen = build_entity_typing_mention(tl_sen, one_ent_list)

                wlp_ent_type_list.append("[ENT-" + wlp_label_pred + "-START]")
                tl_ent_type_list.append(tl_label)
                process_sen_list.append("[CLS] " + ' '.join(process_sen) + " [SEP]")
                doc_name_list.append(doc_name)
                sen_idx_list.append(sen_idx)
                ent_info_list.append((ent_start_old, ent_end_old, ent_start, ent_end, sen_idx, tl_label))

    tokenized_sen_list = [sent.split(' ') for sent in process_sen_list]

    # Get the input_ids and labels
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_sen_list],
                              maxlen=args.max_len, value=tokenizer.pad_token_id, dtype="long", truncating="post", padding="post")

    attention_masks = [[float(i != tokenizer.pad_token_id) for i in ii] for ii in input_ids]

    labels = [LABEL2ID[l] for l in tl_ent_type_list]

    start_tkn_idx_list = [tokenized_sen_list[sen_idx].index(wlp_ent_type_list[sen_idx])
                          if tokenized_sen_list[sen_idx].index(
        wlp_ent_type_list[sen_idx]) < args.max_len else args.max_len - 1
                          for sen_idx in range(len(tokenized_sen_list))
                          ]

    assert len(input_ids) == len(attention_masks) and len(labels) == len(start_tkn_idx_list)

    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)
    start_idx = torch.tensor(start_tkn_idx_list)

    print("--- %s seconds ---" % (time.time() - start_time))

    return inputs, masks, labels, start_idx, tokenized_sen_list, wlp_ent_type_list, doc_name_list, ent_info_list