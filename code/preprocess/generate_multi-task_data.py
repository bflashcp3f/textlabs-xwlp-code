
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
import argparse

from pathlib import Path


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


def generate_sliding_window(sen_list, ner_list, rel_list, window_size, seq_len_limit):
    sen_num = len(sen_list)

    start_sen_idx = 1 - window_size
    end_sen_idx = sen_num + window_size

    start_idx = start_sen_idx
    end_idx = start_idx + window_size

    overall_doc_size = 0
    sen_list_processed = []
    ner_list_processed = []
    rel_list_processed = []

    good_count = 0
    bad_count = 0

    while end_idx < end_sen_idx:

        def group_multiple_sens(sel_start_idx, sel_end_idx):

            sen_len_list = [len(item) for item in sen_list]
            previous_sens_len = sum(sen_len_list[:sel_start_idx])
            sel_sen_list = [item for sublist in sen_list[sel_start_idx:sel_end_idx] for item in sublist]
            sel_ner_list = [handle_doc_offset_ner(item, previous_sens_len, remove=True) for sublist in
                            ner_list[sel_start_idx:sel_end_idx] for item in sublist]
            sel_rel_list = [handle_doc_offset_rel(item, previous_sens_len, remove=True) for sublist in
                            rel_list[sel_start_idx:sel_end_idx] for item in sublist]

            # assert len(sel_sen_list) == 1 and len(sel_ner_list) == 1 and len(sel_rel_list) == 1

            return sel_sen_list, sel_ner_list, sel_rel_list

        if start_idx < 0 and end_idx <= sen_num:
            sen_list_grouped, ner_list_grouped, rel_list_grouped = group_multiple_sens(0, end_idx)
        elif start_idx < 0 and end_idx > sen_num:
            sen_list_grouped, ner_list_grouped, rel_list_grouped = group_multiple_sens(0, sen_num)
        elif start_idx >= 0 and end_idx <= sen_num:
            sen_list_grouped, ner_list_grouped, rel_list_grouped = group_multiple_sens(start_idx, end_idx)
        elif start_idx >= 0 and end_idx > sen_num:
            sen_list_grouped, ner_list_grouped, rel_list_grouped = group_multiple_sens(start_idx, sen_num)
        else:
            raise

        ner_list_grouped = [handle_doc_offset_ner(item, overall_doc_size, remove=False) for item in ner_list_grouped]
        rel_list_grouped = [handle_doc_offset_rel(item, overall_doc_size, remove=False) for item in rel_list_grouped if
                            item[0] > 0 and item[2] > 0]
        # print("ner_list_grouped: ", ner_list_grouped, '\n')

        if len(sen_list_grouped) <= seq_len_limit:

            sen_list_processed.append(sen_list_grouped)
            ner_list_processed.append(ner_list_grouped)
            rel_list_processed.append(rel_list_grouped)

            overall_doc_size += len(sen_list_grouped)
            # print(len(sen_list_grouped))
            good_count += 1
        else:
            # print("Exceed the input length limit: ", len(sen_list_grouped))
            bad_count += 1

        start_idx += 1
        end_idx = start_idx + window_size

    return sen_list_processed, ner_list_processed, rel_list_processed, good_count, bad_count


def process_doc_data(doc_data_list, window_size=6, seq_len_limit=150):
    doc_data_list_new = []
    num_short_seq_doc, num_long_seq_doc = 0, 0

    for doc_data in doc_data_list:
        sen_list = doc_data["sentences"]
        ner_list = doc_data["ner"]
        rel_list = doc_data["relations"]
        doc_name = doc_data["doc_key"]

        doc_data_new = {}

        # Remove "ignored" entities in ner_list
        ner_list_new = [[item1 for item1 in item if item1[-1] != 'ignored'] for item in ner_list]
        assert len(ner_list_new) == len(ner_list)

        # generate sliding window data
        final_sen_list, final_ner_list, final_rel_list, num_short_seq, num_long_seq = generate_sliding_window(sen_list, ner_list_new, \
                                                                                 rel_list, window_size, seq_len_limit)

        num_short_seq_doc += num_short_seq
        num_long_seq_doc += num_long_seq
        doc_data_new["clusters"] = [[] for i in range(len(final_sen_list))]
        doc_data_new["sentences"] = final_sen_list
        doc_data_new["ner"] = final_ner_list
        doc_data_new["relations"] = final_rel_list
        doc_data_new["doc_key"] = doc_name

        doc_data_list_new.append(doc_data_new)

    # # See how many sequences surpass the length limit
    # print(f"The number of regular input sequences {num_short_seq_doc}")
    # print(f"The number of over-limit input sequences {num_long_seq_doc}")

    return doc_data_list_new


def main(args):

    doc_data_all = load_from_jsonl(args.xwlp_path)

    with open("data/cross_validate.json", 'r') as f:
        cross_validate = json.load(f)

    for split_idx in range(5):

        print(f"For split {split_idx}: ")

        train_split = cross_validate[f"split_{split_idx}"]["train_list"]
        test_split = cross_validate[f"split_{split_idx}"]["test_list"]

        dygiepp_train = [doc_data for doc_data in doc_data_all if doc_data["doc_key"] in train_split]
        dygiepp_test = [doc_data for doc_data in doc_data_all if doc_data["doc_key"] in test_split]

        # print(dygiepp_train[0])
        print("Process training data")
        dygiepp_train_sliding = process_doc_data(dygiepp_train, window_size=args.window_size, seq_len_limit=args.seq_len_limit)
        print("Process test data")
        dygiepp_test_sliding = process_doc_data(dygiepp_test, window_size=args.window_size, seq_len_limit=args.seq_len_limit)

        saved_data_dir = Path(f"{args.output_dir}/dygiepp/split_{split_idx}")
        saved_data_dir.mkdir(parents=True, exist_ok=True)

        train_output_file = saved_data_dir / 'train.json'
        print(train_output_file)

        with open(train_output_file, 'w', encoding='utf-8') as output_file:
            for dic in dygiepp_train_sliding:
                json.dump(dic, output_file)
                output_file.write("\n")

        dev_output_file = saved_data_dir / 'dev.json'
        print(dev_output_file)

        with open(dev_output_file, 'w', encoding='utf-8') as output_file:
            for dic in dygiepp_test_sliding:
                json.dump(dic, output_file)
                output_file.write("\n")

        test_output_file = saved_data_dir / 'test.json'
        print(test_output_file)

        with open(test_output_file, 'w', encoding='utf-8') as output_file:
            for dic in dygiepp_test_sliding:
                json.dump(dic, output_file)
                output_file.write("\n")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--xwlp_path", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument('--window_size', type=int, default=5, required=True)
    parser.add_argument('--seq_len_limit', type=int, default=150, required=True)

    args = parser.parse_args()

    main(args)

