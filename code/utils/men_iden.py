

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
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from keras.preprocessing.sequence import pad_sequences


ENT_NAME = ['Amount', 'Reagent', 'Device', 'Time', 'Speed', 'Action', 'Mention', 'Location', 'Numerical',
                     'Method', 'Temperature', 'Modifier', 'Concentration', 'Size', 'Generic-Measure', 'Seal',
                     'Measure-Type', 'Misc', 'pH', 'Unit']

TAG_NAME = ['O']

for item in ENT_NAME:
    TAG_NAME.append('B-' + item)
    TAG_NAME.append('I-' + item)

# print(TAG_NAME)

TAG2IDX = dict((value, key) for key, value in enumerate(TAG_NAME))
# print(TAG2IDX)

ENT_NAME_START = []

for item in ENT_NAME:
    ENT_NAME_START.append("[" + item + "-START]")

# print(ENT_NAME_START)

ENT_NAME_END = []

for item in ENT_NAME:
    ENT_NAME_END.append("[" + item + "-END]")

# print(ENT_NAME_END)


def load_from_txt(data_path, verbose=False, strip=True):
    examples = []

    with open(data_path, encoding='utf-8') as infile:
        while True:
            line = infile.readline()
            if len(line) == 0:
                break

            if strip:
                line = line.strip()

            examples.append(line)

    if verbose:
        print("{} examples read in {} .".format(len(examples), data_path))
    return examples


def load_from_jsonl(file_name):
    data_list = []
    with open(file_name) as f:
        for line in f:
            data_list.append(json.loads(line))

    return data_list


def get_processed_sentences(file_set, data_dir):

    all_label_list = []
    prepared_sen_over_files = []

    #     for file_idx in range(800):
    for file_idx in file_set:

        prepared_sen_list = []

        txt_file = '{}/protocol_{}.txt'.format(data_dir, str(file_idx))
        ann_file = '{}/protocol_{}.ann'.format(data_dir, str(file_idx))

        if not (os.path.isfile(txt_file) and os.path.isfile(ann_file)):
            continue
        #     else:
        #         print(txt_file)
        #         print(ann_file)
        #         print()

        sen_list = load_from_txt(txt_file, strip=False)
        sen_len_list = [len(item) for item in sen_list]

        ann_list = load_from_txt(ann_file)
        all_sen_str = ''.join(sen_list)

        ent_start_list = []

        for item in ann_list:
            if item[0] == 'T':
                try:
                    ent_id, label_offset, ent_str = item.split('\t')
                except:
                    print('item split problem')
                    print(ann_file)
                    print(item)

                try:
                    if ';' not in label_offset:
                        ent_label, ent_start, ent_end = label_offset.split(' ')
                        ent_start, ent_end = int(ent_start), int(ent_end)
                        all_label_list.append(ent_label)
                    else:
                        continue
                except:
                    print('label_offset split problem')
                    print(label_offset)
                #                 raise

                assert ent_str == all_sen_str[ent_start:ent_end] or \
                       ent_str == all_sen_str[ent_start:ent_end].strip()

                ent_start_list.append((ent_start, (ent_str, ent_start, ent_end, ent_label)))

        #     print(len(all_label_list), Counter(all_label_list))

        # Just to split entities by sentence
        sen_start_list = [sum(sen_len_list[:index]) for index in range(len(sen_len_list) + 1)]

        sen_idx = 0
        ent_idx = 0
        sen_ent_dict = defaultdict(list)

        sorted_ent_start_list = sorted(ent_start_list, key=lambda x: x[0])

        while ent_idx < len(sorted_ent_start_list) and sen_idx < len(sen_start_list):
            #     print(ent_idx, sen_idx)

            ent_start, ent_info = sorted_ent_start_list[ent_idx]

            if ent_start >= sen_start_list[sen_idx] and \
                    ent_start < sen_start_list[sen_idx + 1]:

                ent_str, ent_start, ent_end, ent_label = ent_info

                # Remove the sentence offset
                ent_start, ent_end = ent_start - sen_start_list[sen_idx], \
                                     ent_end - sen_start_list[sen_idx]

                sen_ent_dict[sen_idx].append((ent_str, ent_start, ent_end, ent_label))

                ent_idx += 1
                continue

            elif ent_start >= sen_start_list[sen_idx + 1]:
                sen_idx += 1
            else:
                print("Bug here")

        for sen_idx in range(len(sen_list)):

            sen_str = sen_list[sen_idx]

            if sen_idx not in sen_ent_dict:
                prepared_sen_list.append(sen_str.strip())
                continue

            ent_list = sen_ent_dict[sen_idx]

            span_start, span_end = 0, 0
            span_list = []
            label_list = []

            for ent_str, ent_start, ent_end, ent_label in ent_list:

                if ent_start > 0:
                    span_end = ent_start

                    if sen_str[span_start:span_end].strip():
                        span_list.append(sen_str[span_start:span_end])
                        label_list.append('O')

                span_list.append(sen_str[ent_start:ent_end])
                label_list.append(ent_label)

                span_start = ent_end

            # Add the last part of sentence
            if span_start != len(sen_str.strip()):
                span_end = len(sen_str.strip())
                span_list.append(sen_str[span_start:span_end])
                label_list.append('O')

            # Get the label for corresponding span
            span_label_list = list([item for item in zip(span_list, label_list) if item[0].strip()])

            # Add label to the sentence for bert tokenizer
            span_modified_list = []
            for span, span_label in span_label_list:
                if span_label != 'O':
                    span_modified_list += ['[{}-START]'.format(span_label), span, \
                                           '[{}-END]'.format(span_label)]
                else:
                    span_modified_list.append(span)

            prepared_sen = ' '.join(span_modified_list)

            #         print(prepared_sen)
            #         print()
            prepared_sen_list.append(prepared_sen)

        #     break

        prepared_sen_over_files.append(prepared_sen_list)

    # Remove the title since most of them are not annotated
    prepared_sen_aggregate = []

    for item in prepared_sen_over_files:
        prepared_sen_aggregate += item[1:]

    return prepared_sen_aggregate


def tokenize_txt(prepared_sen_aggregate, tokenizer):
    start_time = time.time()

    tokenized_word_tag = [tokenizer.tokenize(f'{tokenizer.cls_token} {sent} {tokenizer.eos_token}') for sent in prepared_sen_aggregate]

    print("--- %s seconds ---" % (time.time() - start_time))

    return tokenized_word_tag


def decompose_tokenized_text(token_list):
    """
    Split the output of BERT tokenizer into word_list and tag_list
    """

    # The initial tag is 'O'
    tag = 'O'
    word_list = []
    tag_list = []

    # A flag to indicate if it is in the entity now
    # 0 means out, 1 means start, 2 means in the middle
    in_ent = 0

    for token in token_list:
        # if the token ends with '-START]'
        # change the ent flag
        if token[-len('-START]'):] == '-START]':
            tag = token[1:-1].rsplit('-', 1)[0]
            in_ent = 1
        elif token[-len('-END]'):] == '-END]':
            tag = 'O'
            in_ent = 0
        else:
            word_list.append(token)

            if in_ent == 1:
                tag_list.append('B-' + tag)
                in_ent = 2
            elif in_ent == 2:
                tag_list.append('I-' + tag)
            else:
                tag_list.append(tag)

    #     print(list(zip(word_list, tag_list)))
    #     return(list(zip(word_list, tag_list)))
    return word_list, tag_list


def index_ent_in_prediction(word_list, tag_list):
    ent_queue, ent_idx_queue, ent_type_queue = [], [], []
    ent_list, ent_idx_list, ent_type_list = [], [], []

    for word_idx in range(len(word_list)):

        if 'B-' in tag_list[word_idx]:
            if ent_queue:

                if len(set(ent_type_queue)) != 1:
                    print(ent_queue)
                    print(ent_idx_queue)
                    print(ent_type_queue)
                    print(Counter(ent_type_queue).most_common())
                    print()

                else:
                    ent_list.append(' '.join(ent_queue).strip())
                    #                     ent_idx_list.append((ent_idx_queue[0], ent_idx_queue[-1]+1))
                    ent_idx_list.append((ent_idx_queue[0], ent_idx_queue[-1]))

                    assert len(set(ent_type_queue)) == 1
                    ent_type_list.append(ent_type_queue[0])

            ent_queue, ent_idx_queue, ent_type_queue = [], [], []
            ent_queue.append(word_list[word_idx])
            ent_idx_queue.append(word_idx)
            ent_type_queue.append(tag_list[word_idx][2:])

        if 'I-' in tag_list[word_idx]:
            if word_idx == 0 or (word_idx > 0 and tag_list[word_idx][2:] == tag_list[word_idx - 1][2:]):
                ent_queue.append(word_list[word_idx])
                ent_idx_queue.append(word_idx)
                ent_type_queue.append(tag_list[word_idx][2:])
            else:
                if ent_queue:

                    if len(set(ent_type_queue)) != 1:
                        print(ent_queue)
                        print(ent_idx_queue)
                        print(ent_type_queue)
                        print(Counter(ent_type_queue).most_common())
                        print()
                    else:
                        ent_list.append(' '.join(ent_queue).strip())
                        #                         ent_idx_list.append((ent_idx_queue[0], ent_idx_queue[-1]+1))
                        ent_idx_list.append((ent_idx_queue[0], ent_idx_queue[-1]))

                        assert len(set(ent_type_queue)) == 1
                        ent_type_list.append(ent_type_queue[0])

                ent_queue, ent_idx_queue, ent_type_queue = [], [], []
                ent_queue.append(word_list[word_idx])
                ent_idx_queue.append(word_idx)
                ent_type_queue.append(tag_list[word_idx][2:])

        if 'O' == tag_list[word_idx] or word_idx == len(word_list) - 1:
            if ent_queue:

                if len(set(ent_type_queue)) != 1:
                    print(ent_queue)
                    print(ent_idx_queue)
                    print(ent_type_queue)
                    print(Counter(ent_type_queue).most_common())
                    print()

                else:
                    ent_list.append(' '.join(ent_queue).strip())
                    #                     ent_idx_list.append((ent_idx_queue[0], ent_idx_queue[-1]+1))
                    ent_idx_list.append((ent_idx_queue[0], ent_idx_queue[-1]))

                    assert len(set(ent_type_queue)) == 1
                    ent_type_list.append(ent_type_queue[0])

            ent_queue, ent_idx_queue, ent_type_queue = [], [], []

    return ent_list, ent_idx_list, ent_type_list


def split_token_label(tokenized_word_tag):
    start_time = time.time()

    tokenized_texts = []
    labels = []

    for item in tokenized_word_tag:
        word_list, label_list = decompose_tokenized_text(item)
        assert len(word_list) == len(label_list)

        tokenized_texts.append(word_list)
        labels.append(label_list)

    print("--- %s seconds ---" % (time.time() - start_time))

    return tokenized_texts, labels


def generate_wlp_ner(ner_list, wlp_label_list):
    wlp_ner_list = []

    for ner_per_sen, wlp_label_per_sen in zip(ner_list, wlp_label_list):

        assert len(ner_per_sen) == len(wlp_label_per_sen)
        #         print(ner_per_sen)
        wlp_per_sen = []

        for ner_per, wlp_label_per in zip(ner_per_sen, wlp_label_per_sen):
            #             print(ner_per)
            #             print(ner_per[:2] + [wlp_label_per])
            wlp_per_sen.append(ner_per[:2] + [wlp_label_per])

        wlp_ner_list.append(wlp_per_sen)

    assert len(wlp_ner_list) == len(ner_list)

    return wlp_ner_list


def process_sentences(sen_list, ent_list):

    prepared_sen_list = []
    doc_len = 0

    for sen_idx in range(len(sen_list)):

        word_list = sen_list[sen_idx]

        if not ent_list[sen_idx]:
            prepared_sen_list.append(' '.join(word_list))
            doc_len += len(word_list)
            continue

        ent_per_sen = ent_list[sen_idx]

        span_start, span_end = 0, 0
        span_list = []
        label_list = []

        for ent_start_doc, ent_end_doc, ent_label in ent_per_sen:

            ent_start, ent_end = ent_start_doc - doc_len, ent_end_doc - doc_len
            #             print(ent_start, ent_end, ent_label)

            if ent_start > 0:
                span_end = ent_start

                if word_list[span_start:span_end]:
                    span_list.append(' '.join(word_list[span_start:span_end]))
                    label_list.append('O')

            span_list.append(' '.join(word_list[ent_start:ent_end + 1]))
            label_list.append(ent_label)

            span_start = ent_end + 1

        # Add the last part of sentence
        if span_start != len(word_list):
            span_end = len(word_list)
            span_list.append(' '.join(word_list[span_start:span_end]))
            label_list.append('O')

        # Get the label for corresponding span
        span_label_list = list([item for item in zip(span_list, label_list) if item[0].strip()])

        # Add label to the sentence for bert tokenizer
        span_modified_list = []
        for span, span_label in span_label_list:
            if span_label != 'O':
                span_modified_list += ['[{}-START]'.format(span_label), span, \
                                       '[{}-END]'.format(span_label)]
            else:
                span_modified_list.append(span)

        prepared_sen = ' '.join(span_modified_list)

        prepared_sen_list.append(prepared_sen)
        doc_len += len(word_list)

    #     print(prepared_sen_list)
    return prepared_sen_list


def process_set(process_sens, tokenizer, args, tag2idx, train_flag=False):
    print("Start tokenization")
    process_tokenized = tokenize_txt(process_sens, tokenizer)

    print("\nStart splitting tokens and labels")
    process_tokens, process_labels = split_token_label(process_tokenized)
    # print(max([len(item) for item in process_tokens]))

    process_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in process_tokens], \
                                    maxlen=args.max_len, value=tokenizer.pad_token_id, dtype="long", \
                                    truncating="post", padding="post")
    process_attention_masks = [[float(i != tokenizer.pad_token_id) for i in ii] for ii in process_input_ids]

    process_inputs = torch.tensor(process_input_ids)
    process_masks = torch.tensor(process_attention_masks)

    process_inputs_len = process_inputs.shape[1]

    process_tags = pad_sequences([[tag2idx[l] for l in lab] for lab in process_labels],
                               maxlen=process_inputs_len, value=tag2idx["O"], padding="post",
                               dtype="long", truncating="post")
    process_tags = torch.tensor(process_tags)

    print(process_inputs.shape, process_tags.shape)

    process_data = TensorDataset(process_inputs, process_masks, process_tags)

    if train_flag:
        process_sampler = RandomSampler(process_data)
    else:
        process_sampler = SequentialSampler(process_data)

    process_dataloader = DataLoader(process_data, sampler=process_sampler, batch_size=args.batch_size)

    return process_dataloader, process_inputs, process_masks, process_tags


def data_preprocess(data_list):
    processed_sen_list = []
    processed_wlp_ner_list = []
    processed_tl_ner_list = []
    processed_rel_list = []
    processed_doc_name_list = []

    for item in data_list:

        wlp_ner_list = generate_wlp_ner(item['ner'], item['wlp_labels'])
        processed_sen_list += process_sentences(item['sentences'], wlp_ner_list)
        processed_wlp_ner_list += wlp_ner_list
        processed_tl_ner_list += item['ner']
        processed_rel_list += item['relations']
        processed_doc_name_list += [item['doc_key']] * len(wlp_ner_list)

    return processed_sen_list, processed_wlp_ner_list, processed_tl_ner_list, \
           processed_rel_list, processed_doc_name_list
