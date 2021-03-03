
import sys
import random
import argparse

from collections import Counter
from utils.re import *

def main(args):

    print("The Evaluation of Argument Role Labeling + Temporal Ordering.")

    print("GOLD INPUT:")
    # Gold Dev
    print("Dev set:")

    gold_all = []
    pred_all = []

    for split_idx in range(2):
        gold_input_pred_file = f'{args.gold_input_pred_path}/split_{split_idx}/predictions.json'

        # Reading data back
        with open(gold_input_pred_file, 'r') as f:
            gold_input_pred_data = json.load(f)

        gold_all += gold_input_pred_data['gold_tags']
        pred_all += gold_input_pred_data['pred_tags']

    precision, recall, f1 = score(gold_all, pred_all, True)
    # print("Precision-Score: {:.1f}".format(precision * 100))
    # print("Recall-Score: {:.1f}".format(recall * 100))
    # print("F1-Score: {:.1f}".format(f1 * 100))
    print()

    # Gold Test
    print("Test set:")

    gold_all = []
    pred_all = []

    for split_idx in range(2, 5):
        gold_input_pred_file = f'{args.gold_input_pred_path}/split_{split_idx}/predictions.json'

        # Reading data back
        with open(gold_input_pred_file, 'r') as f:
            gold_input_pred_data = json.load(f)

        gold_all += gold_input_pred_data['gold_tags']
        pred_all += gold_input_pred_data['pred_tags']

    precision, recall, f1 = score(gold_all, pred_all, True)
    # print("Precision-Score: {:.1f}".format(precision * 100))
    # print("Recall-Score: {:.1f}".format(recall * 100))
    # print("F1-Score: {:.1f}".format(f1 * 100))
    print()


    print("\nCore relations: ")
    score_sel_rel(gold_all, pred_all, CORE_REL, True)

    print("\nNon-Core relations: ")
    score_sel_rel(gold_all, pred_all, NON_CORE_REL, True)

    print("\nTemporal Ordering relations: ")
    score_sel_rel(gold_all, pred_all, TD_REL, True)


    print("\nPREDICTED INPUT:")

    # Pred Dev
    print("Dev set:")
    TP = 0
    FP = 0
    FN = 0

    for split_idx in range(2):

        gold_input_pred_file = f"{args.pred_input_pred_path}/split_{split_idx}.json"
        xwlp_test = load_from_jsonl(gold_input_pred_file)

        for doc_data in xwlp_test:
            doc_name = doc_data['doc_key']
            gold_rel_list = doc_data['tl_rel_gold_tokenized']
            pred_rel_list = doc_data['tl_rel_pred_tokenized']

            gold_rel_str_list = [' '.join([str(item1) for item1 in item[:]]) \
                                 for item in gold_rel_list if item[-1] != 'no_relation']
            pred_rel_str_list = [' '.join([str(item1) for item1 in item[:]]) \
                                 for item in pred_rel_list if item[-1] != 'no_relation']

            TP += len(set(gold_rel_str_list) & set(pred_rel_str_list))
            FP += (len(set(pred_rel_str_list)) - len(set(gold_rel_str_list) & set(pred_rel_str_list)))
            FN += (len(set(gold_rel_str_list)) - len(set(gold_rel_str_list) & set(pred_rel_str_list)))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    # print(f"\nFor split {split_idx}")
    # print(f"TP:{TP}, FP:{FP}, FN:{FN}")
    print("Precision-Score: {:.1f}".format(precision * 100))
    print("Recall-Score: {:.1f}".format(recall * 100))
    print("F1-Score: {:.1f}".format(f1 * 100))
    print()

    # Pred Test
    print("Test set:")
    TP = 0
    FP = 0
    FN = 0

    xwlp_test_all = []
    gold_rel_dict_all = {}
    pred_rel_dict_all = {}

    for split_idx in range(2, 5):

        gold_input_pred_file = f"{args.pred_input_pred_path}/split_{split_idx}.json"
        xwlp_test = load_from_jsonl(gold_input_pred_file)
        xwlp_test_all += xwlp_test

        for doc_data in xwlp_test:
            doc_name = doc_data['doc_key']
            gold_rel_list = doc_data['tl_rel_gold_tokenized']
            pred_rel_list = doc_data['tl_rel_pred_tokenized']

            gold_rel_str_list = [' '.join([str(item1) for item1 in item[:]]) \
                                 for item in gold_rel_list if item[-1] != 'no_relation']
            pred_rel_str_list = [' '.join([str(item1) for item1 in item[:]]) \
                                 for item in pred_rel_list if item[-1] != 'no_relation']

            gold_rel_dict_all[doc_name] = gold_rel_list
            pred_rel_dict_all[doc_name] = pred_rel_list

            TP += len(set(gold_rel_str_list) & set(pred_rel_str_list))
            FP += (len(set(pred_rel_str_list)) - len(set(gold_rel_str_list) & set(pred_rel_str_list)))
            FN += (len(set(gold_rel_str_list)) - len(set(gold_rel_str_list) & set(pred_rel_str_list)))

            if len(set(gold_rel_str_list)) != len(gold_rel_str_list):
                print("Duplicate happens")
                print(doc_data['doc_key'])
                print(doc_data.keys())
                print(doc_data['relations_tokenized'])
                print(Counter([' '.join([str(item2) for item2 in item1]) for item in doc_data['relations_tokenized'] for item1 in item]))
                print(Counter(gold_rel_str_list))
                print('\n')
                break

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    print("Precision-Score: {:.1f}".format(precision * 100))
    print("Recall-Score: {:.1f}".format(recall * 100))
    print("F1-Score: {:.1f}".format(f1 * 100))
    print()

    print("Performance by relation:")
    score_by_relation(pred_rel_dict_all, gold_rel_dict_all, True)

    print("\nCore relations: ")
    score_sel_rel_infer(pred_rel_dict_all, gold_rel_dict_all, sel_rel=CORE_REL, verbose=True)

    print("\nPerformance by distance:")
    score_by_distance(xwlp_test_all, pred_rel_dict_all, gold_rel_dict_all, sel_rel=CORE_REL, verbose=True)

    print("\nNon-Core relations: ")
    score_sel_rel_infer(pred_rel_dict_all, gold_rel_dict_all, sel_rel=NON_CORE_REL, verbose=True)

    print("\nTemporal Ordering relations: ")
    score_sel_rel_infer(pred_rel_dict_all, gold_rel_dict_all, sel_rel=TD_REL, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--xwlp_path", default=None, type=str, required=True)
    parser.add_argument("--gold_input_pred_path", default=None, type=str, required=True)
    parser.add_argument("--pred_input_pred_path", default=None, type=str, required=True)

    args = parser.parse_args()

    main(args)