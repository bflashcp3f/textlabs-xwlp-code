
import math
import glob
import time
import json
import pickle
import os
import numpy as np
import operator
import time
import argparse
import sys
from pathlib import Path

from collections import defaultdict, Counter, OrderedDict

from utils.multi_task import *


def eval_by_rel_type(sel_relations, args, within_cross_eval=False):

    sel_rel2id = dict((value, key) for key, value in enumerate(sel_relations))

    doc_data_all = load_from_jsonl(args.xwlp_path)

    with open("data/cross_validate.json", 'r') as f:
        cross_validate = json.load(f)

    threshold_size = args.window_size
    dist_size = args.dist_size

    TP_over_dist_thresh = [[0 for i in range(threshold_size)] for j in range(dist_size)]
    FP_over_dist_thresh = [[0 for i in range(threshold_size)] for j in range(dist_size)]
    FN_over_dist_thresh = [[0 for i in range(threshold_size)] for j in range(dist_size)]

    correct_over_dist_thresh = [[[0 for k in range(len(sel_relations))] \
                                 for i in range(threshold_size)] \
                                for j in range(dist_size)]

    gold_over_dist_thresh = [[[0 for k in range(len(sel_relations))] \
                              for i in range(threshold_size)] \
                             for j in range(dist_size)]

    pred_over_dist_thresh = [[[0 for k in range(len(sel_relations))] \
                              for i in range(threshold_size)] \
                             for j in range(dist_size)]

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

        rel_dict_sole_all = []
        rel_dict_all = []

        for doc_sole, doc_combined in zip(xwlp_test_sole, xwlp_pred):

            word2sen_idx = find_word_sen_idx([len(item) for item in doc_sole['sentences']])

            pre_sen_len_list, offset_list = get_pre_sen_len(doc_sole, threshold_size, args.seq_len_limit)

            ner_list_final, rel_list_final = decode_sliding_window(doc_combined, pre_sen_len_list, threshold_size)

            assert len(ner_list_final) == len(offset_list)
            assert len(rel_list_final) == len(offset_list)

            rel_dict = defaultdict(list)

            for item in rel_list_final:
                for item1 in item:
                    rel_dict[abs(word2sen_idx[item1[0]] - word2sen_idx[item1[2]])].append(
                        '.'.join([str(item2) for item2 in item1]))

            rel_list_sole = doc_sole['relations']
            rel_dict_sole = defaultdict(list)

            for sen_idx, rel_list_per_sen in enumerate(rel_list_sole):

                for item in rel_list_per_sen:
                    rel_dict_sole[abs(word2sen_idx[item[0]] - word2sen_idx[item[2]])].append(
                        '.'.join([str(item1) for item1 in item]))

            rel_dict_sole_all.append(rel_dict_sole)
            rel_dict_all.append(rel_dict)

        # print("Max dist in gold labels: ", max([max(item.keys()) for item in rel_dict_sole_all]))
        # max_dist_gold = max([max(item.keys()) for item in rel_dict_sole_all])

        for arg_dist in range(dist_size):

            for threshold in range(1, threshold_size + 1):

                TP_all = FP_all = FN_all = 0

                for rel_dict_sole, rel_dict in zip(rel_dict_sole_all, rel_dict_all):

                    rel_gold_list = rel_dict_sole[arg_dist] if arg_dist in rel_dict_sole else []
                    rel_pred_list = [key for key, value in Counter(rel_dict[arg_dist]).items() if value >= threshold] if arg_dist in rel_dict else []

                    rel_gold_list = [item for item in rel_gold_list if item.rsplit('.')[-1] in sel_relations]
                    rel_pred_list = [item for item in rel_pred_list if item.rsplit('.')[-1] in sel_relations]

                    TP = len([item for item in rel_gold_list if item in rel_pred_list])
                    FP = len(rel_pred_list) - TP
                    FN = len(rel_gold_list) - TP

                    TP_all += TP
                    FP_all += FP
                    FN_all += FN

                    for item in [item for item in rel_gold_list if item in rel_pred_list]:
                        correct_over_dist_thresh[arg_dist][threshold - 1][sel_rel2id[item.split('.')[-1]]] += 1

                    for item in rel_gold_list:
                        gold_over_dist_thresh[arg_dist][threshold - 1][sel_rel2id[item.split('.')[-1]]] += 1

                    for item in rel_pred_list:
                        pred_over_dist_thresh[arg_dist][threshold - 1][sel_rel2id[item.split('.')[-1]]] += 1

                TP_over_dist_thresh[arg_dist][threshold - 1] += TP_all
                FP_over_dist_thresh[arg_dist][threshold - 1] += FP_all
                FN_over_dist_thresh[arg_dist][threshold - 1] += FN_all


    dev_dist_threshold = [0] * dist_size
    for arg_dist in range(args.window_size):
        dist_f1 = []
        for threshold in range(1, args.window_size + 1 - arg_dist):
            # print(f"arg_dist-{arg_dist}, threshold-{threshold}")
            precision, recall, f1 = eveluate(TP_over_dist_thresh[arg_dist][threshold - 1], \
                                             FP_over_dist_thresh[arg_dist][threshold - 1], \
                                             FN_over_dist_thresh[arg_dist][threshold - 1], verbose=False)
            dist_f1.append(f1)
            # print()

        max_dist_f1 = max(dist_f1)
        best_threshold = [item_idx for item_idx, item in enumerate(dist_f1) if item ==max_dist_f1][0]
        # print(f"Best threshold for distance {arg_dist} is {best_threshold+1} with f1 {max_dist_f1*100:.1f}\n")

        dev_dist_threshold[arg_dist] = best_threshold

    if args.eval_dev:
        print("OVERALL PERFORMANCE:")
        TP_final = 0
        FP_final = 0
        FN_final = 0

        for arg_dist, threshold in enumerate(dev_dist_threshold):
            TP_final += TP_over_dist_thresh[arg_dist][threshold - 1]
            FP_final += FP_over_dist_thresh[arg_dist][threshold - 1]
            FN_final += FN_over_dist_thresh[arg_dist][threshold - 1]

        eveluate(TP_final, FP_final, FN_final)
        print()

    print("TEST PERFORMANCE")

    TP_over_dist_thresh = [[0 for i in range(threshold_size)] for j in range(dist_size)]
    FP_over_dist_thresh = [[0 for i in range(threshold_size)] for j in range(dist_size)]
    FN_over_dist_thresh = [[0 for i in range(threshold_size)] for j in range(dist_size)]

    correct_over_dist_thresh = [[[0 for k in range(len(sel_relations))] \
                                 for i in range(threshold_size)] \
                                for j in range(dist_size)]

    gold_over_dist_thresh = [[[0 for k in range(len(sel_relations))] \
                              for i in range(threshold_size)] \
                             for j in range(dist_size)]

    pred_over_dist_thresh = [[[0 for k in range(len(sel_relations))] \
                              for i in range(threshold_size)] \
                             for j in range(dist_size)]


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

        rel_dict_sole_all = []
        rel_dict_all = []

        for doc_sole, doc_combined in zip(xwlp_test_sole, xwlp_pred):

            word2sen_idx = find_word_sen_idx([len(item) for item in doc_sole['sentences']])

            pre_sen_len_list, offset_list = get_pre_sen_len(doc_sole, threshold_size, args.seq_len_limit)

            ner_list_final, rel_list_final = decode_sliding_window(doc_combined, pre_sen_len_list, threshold_size)

            assert len(ner_list_final) == len(offset_list)
            assert len(rel_list_final) == len(offset_list)

            rel_dict = defaultdict(list)

            for item in rel_list_final:
                for item1 in item:
                    rel_dict[abs(word2sen_idx[item1[0]] - word2sen_idx[item1[2]])].append(
                        '.'.join([str(item2) for item2 in item1]))

            rel_list_sole = doc_sole['relations']
            rel_dict_sole = defaultdict(list)

            for sen_idx, rel_list_per_sen in enumerate(rel_list_sole):

                for item in rel_list_per_sen:
                    rel_dict_sole[abs(word2sen_idx[item[0]] - word2sen_idx[item[2]])].append(
                        '.'.join([str(item1) for item1 in item]))

            rel_dict_sole_all.append(rel_dict_sole)
            rel_dict_all.append(rel_dict)

        for arg_dist in range(dist_size):

            for threshold in range(1, threshold_size + 1):

                TP_all = FP_all = FN_all = 0

                for rel_dict_sole, rel_dict in zip(rel_dict_sole_all, rel_dict_all):

                    rel_gold_list = rel_dict_sole[arg_dist] if arg_dist in rel_dict_sole else []
                    rel_pred_list = [key for key, value in Counter(rel_dict[arg_dist]).items() if value >= threshold] if arg_dist in rel_dict else []

                    rel_gold_list = [item for item in rel_gold_list if item.rsplit('.')[-1] in sel_relations]
                    rel_pred_list = [item for item in rel_pred_list if item.rsplit('.')[-1] in sel_relations]

                    TP = len([item for item in rel_gold_list if item in rel_pred_list])
                    FP = len(rel_pred_list) - TP
                    FN = len(rel_gold_list) - TP

                    TP_all += TP
                    FP_all += FP
                    FN_all += FN

                    for item in [item for item in rel_gold_list if item in rel_pred_list]:
                        correct_over_dist_thresh[arg_dist][threshold - 1][sel_rel2id[item.split('.')[-1]]] += 1

                    for item in rel_gold_list:
                        gold_over_dist_thresh[arg_dist][threshold - 1][sel_rel2id[item.split('.')[-1]]] += 1

                    for item in rel_pred_list:
                        pred_over_dist_thresh[arg_dist][threshold - 1][sel_rel2id[item.split('.')[-1]]] += 1

                TP_over_dist_thresh[arg_dist][threshold - 1] += TP_all
                FP_over_dist_thresh[arg_dist][threshold - 1] += FP_all
                FN_over_dist_thresh[arg_dist][threshold - 1] += FN_all

    print("OVERALL PERFORMANCE:")
    TP_final = 0
    FP_final = 0
    FN_final = 0

    for arg_dist, threshold in enumerate(dev_dist_threshold):
        TP_final += TP_over_dist_thresh[arg_dist][threshold - 1]
        FP_final += FP_over_dist_thresh[arg_dist][threshold - 1]
        FN_final += FN_over_dist_thresh[arg_dist][threshold - 1]

    eveluate(TP_final, FP_final, FN_final)
    print()

    if within_cross_eval:
        print("WITHIN-SENTENCE PERFORMANCE:")
        TP_final = 0
        FP_final = 0
        FN_final = 0

        for arg_dist, threshold in enumerate(dev_dist_threshold):
            if arg_dist != 0:
                continue

            TP_final += TP_over_dist_thresh[arg_dist][threshold - 1]
            FP_final += FP_over_dist_thresh[arg_dist][threshold - 1]
            FN_final += FN_over_dist_thresh[arg_dist][threshold - 1]

        eveluate(TP_final, FP_final, FN_final)
        print()

        print("CROSS-SENTENCE PERFORMANCE:")
        TP_final = 0
        FP_final = 0
        FN_final = 0

        for arg_dist, threshold in enumerate(dev_dist_threshold):

            if arg_dist == 0:
                continue

            TP_final += TP_over_dist_thresh[arg_dist][threshold - 1]
            FP_final += FP_over_dist_thresh[arg_dist][threshold - 1]
            FN_final += FN_over_dist_thresh[arg_dist][threshold - 1]

        eveluate(TP_final, FP_final, FN_final)
        print()

    print("BY-RELATION PERFORMANCE:")
    longest_relation = 0
    for relation in sorted(sel_relations):
        longest_relation = max(len(relation), longest_relation)

    for rel in sel_relations:

        TP_final = 0
        TP_FP_final = 0
        TP_FN_final = 0

        # for arg_dist, threshold in dist_arg_list:
        for arg_dist, threshold in enumerate(dev_dist_threshold):
            TP_final += correct_over_dist_thresh[arg_dist][threshold - 1][sel_rel2id[rel]]
            TP_FP_final += pred_over_dist_thresh[arg_dist][threshold - 1][sel_rel2id[rel]]
            TP_FN_final += gold_over_dist_thresh[arg_dist][threshold - 1][sel_rel2id[rel]]

        prec, recall, f1 = eveluate(TP_final, TP_FP_final - TP_final, TP_FN_final - TP_final, verbose=False)

        sys.stdout.write(("{:<" + str(longest_relation) + "}").format(rel))
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
        sys.stdout.write("  #: %d" % TP_FN_final)
        sys.stdout.write("\n")
    print()


def main(args):

    print("\nCore relations: ")
    eval_by_rel_type(CORE_REL, args, within_cross_eval=True)

    print("\nNon-Core relations: ")
    eval_by_rel_type(NON_CORE_REL, args)

    print("\nTemporal Ordering relations: ")
    eval_by_rel_type(TD_REL, args)


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

