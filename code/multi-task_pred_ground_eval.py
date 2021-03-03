

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
import argparse
from pathlib import Path
from sklearn.model_selection import KFold

from collections import defaultdict, Counter

from utils.multi_task import *


def main(args):

    doc_data_all = load_from_jsonl(args.xwlp_path)

    with open("data/cross_validate.json", 'r') as f:
        cross_validate = json.load(f)

    window_size = args.window_size
    TF_over_threshold = [0] * window_size
    FP_over_threshold = [0] * window_size
    FN_over_threshold = [0] * window_size

    if args.eval_dev:
        print("DEV PERFORMANCE")

    for split_idx in range(2):

        test_split = cross_validate[f"split_{split_idx}"]["test_list"]
        xwlp_test_sole = [doc_data for doc_data in doc_data_all if doc_data["doc_key"] in test_split]
        sliding_window_dir = Path(f'{args.gold_path}/split_{split_idx}')
        test_file = sliding_window_dir / 'test.json'

        xwlp_test = []
        with open(test_file) as f:
            for line in f:
                xwlp_test.append(json.loads(line))

        assert [item['doc_key'] for item in xwlp_test] == [item['doc_key'] for item in xwlp_test_sole]

        pred_file = f'{args.pred_path}/xwlp-split-{split_idx}/test_pred.jsonl'

        xwlp_pred = []
        with open(pred_file) as f:
            for line in f:
                xwlp_pred.append(json.loads(line))


        for threshold in range(1, window_size + 1):


            TP_all = FP_all = FN_all = 0

            for doc_sole, doc_combined in zip(xwlp_test_sole, xwlp_pred):

                assert doc_sole['doc_key'] == doc_combined['doc_key']

                word2sen_idx = find_word_sen_idx([len(item) for item in doc_sole['sentences']])

                pre_sen_len_list, offset_list = get_pre_sen_len(doc_sole, window_size)

                ner_list_final, rel_list_final = decode_sliding_window(doc_combined, pre_sen_len_list, window_size)

                assert len(ner_list_final) == len(offset_list)
                assert len(rel_list_final) == len(offset_list)

                ner_dict = defaultdict(list)
                for item in ner_list_final:
                    for item1 in item:
                        ner_dict[word2sen_idx[item1[0]]].append('.'.join([str(item2) for item2 in item1]))


                ner_list_sole = doc_sole['ner']

                for sen_idx, ner_list_per_sen in enumerate(ner_list_sole):

                    ner_gold_list = ['.'.join([str(item2) for item2 in item]) \
                                     for item in ner_list_per_sen if item[-1].endswith('-op') and item[-1] != 'ignored']
                    ner_pred_list = [key for key, value in Counter(ner_dict[sen_idx]).items() \
                                     if value >= threshold and key.endswith('-op')]

                    TP = len(set(ner_gold_list) & set(ner_pred_list))
                    FP = len(set(ner_pred_list)) - TP
                    FN = len(set(ner_gold_list)) - TP

                    TP_all += TP
                    FP_all += FP
                    FN_all += FN

            TF_over_threshold[threshold - 1] += TP_all
            FP_over_threshold[threshold - 1] += FP_all
            FN_over_threshold[threshold - 1] += FN_all

    best_threshold, best_f1 = 1, -1
    for threshold in range(1, window_size + 1):

        precision, recall, f1 = eveluate(TF_over_threshold[threshold - 1], FP_over_threshold[threshold - 1], FN_over_threshold[threshold - 1])

        if f1 > best_f1:
            best_threshold = threshold

    if args.eval_dev:
        eveluate(TF_over_threshold[best_threshold - 1], FP_over_threshold[best_threshold - 1], FN_over_threshold[best_threshold - 1], verbose=True)
        print()


    # Test set
    print("TEST PERFORMANCE")
    TF_over_threshold = [0] * window_size
    FP_over_threshold = [0] * window_size
    FN_over_threshold = [0] * window_size

    for split_idx in range(2, 5):


        test_split = cross_validate[f"split_{split_idx}"]["test_list"]
        xwlp_test_sole = [doc_data for doc_data in doc_data_all if doc_data["doc_key"] in test_split]
        sliding_window_dir = Path(f'{args.gold_path}/split_{split_idx}')
        test_file = sliding_window_dir / 'test.json'

        xwlp_test = []
        with open(test_file) as f:
            for line in f:
                xwlp_test.append(json.loads(line))

        assert [item['doc_key'] for item in xwlp_test] == [item['doc_key'] for item in xwlp_test_sole]

        pred_file = f'{args.pred_path}/xwlp-split-{split_idx}/test_pred.jsonl'

        xwlp_pred = []
        with open(pred_file) as f:
            for line in f:
                xwlp_pred.append(json.loads(line))

        for threshold in range(1, window_size + 1):

            TP_all = FP_all = FN_all = 0

            for doc_sole, doc_combined in zip(xwlp_test_sole, xwlp_pred):

                assert doc_sole['doc_key'] == doc_combined['doc_key']

                word2sen_idx = find_word_sen_idx([len(item) for item in doc_sole['sentences']])

                pre_sen_len_list, offset_list = get_pre_sen_len(doc_sole, window_size)

                ner_list_final, rel_list_final = decode_sliding_window(doc_combined, pre_sen_len_list, window_size)

                assert len(ner_list_final) == len(offset_list)
                assert len(rel_list_final) == len(offset_list)


                ner_dict = defaultdict(list)
                for item in ner_list_final:
                    for item1 in item:
                        ner_dict[word2sen_idx[item1[0]]].append('.'.join([str(item2) for item2 in item1]))

                ner_list_sole = doc_sole['ner']

                for sen_idx, ner_list_per_sen in enumerate(ner_list_sole):

                    ner_gold_list = ['.'.join([str(item2) for item2 in item]) \
                                     for item in ner_list_per_sen if item[-1].endswith('-op') and item[-1] != 'ignored']
                    ner_pred_list = [key for key, value in Counter(ner_dict[sen_idx]).items() \
                                     if value >= threshold and key.endswith('-op')]

                    TP = len(set(ner_gold_list) & set(ner_pred_list))
                    FP = len(set(ner_pred_list)) - TP
                    FN = len(set(ner_gold_list)) - TP

                    TP_all += TP
                    FP_all += FP
                    FN_all += FN

            TF_over_threshold[threshold - 1] += TP_all
            FP_over_threshold[threshold - 1] += FP_all
            FN_over_threshold[threshold - 1] += FN_all


    eveluate(TF_over_threshold[best_threshold - 1], FP_over_threshold[best_threshold - 1], FN_over_threshold[best_threshold - 1], verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--xwlp_path", default="data/xwlp.json", type=str)
    parser.add_argument("--gold_path", default=None, type=str, required=True)
    parser.add_argument("--pred_path", default=None, type=str, required=True)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--seq_len_limit', type=int, default=150)
    parser.add_argument('--dist_size', type=int, default=25)
    parser.add_argument("--eval_dev", action='store_true', help="Show dev set performance")

    args = parser.parse_args()

    main(args)