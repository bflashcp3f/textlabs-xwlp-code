
import sys

from utils.pred_ground import *


def main(args):

    print("The Evaluation of Predicate Grounding.")

    # Gold Dev
    print("The input is gold mentions:")
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

    precision, recall, f1 = score(gold_all, pred_all)
    print("Precision-Score: {:.1f}".format(precision * 100))
    print("Recall-Score: {:.1f}".format(recall * 100))
    print("F1-Score: {:.1f}".format(f1 * 100))
    print()

    # Gold Test
    print("The input is gold mentions:")
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

    precision, recall, f1 = score(gold_all, pred_all)
    print("Precision-Score: {:.1f}".format(precision * 100))
    print("Recall-Score: {:.1f}".format(recall * 100))
    print("F1-Score: {:.1f}".format(f1 * 100))
    print()


    # Pred Dev
    print("The input is predicted mentions:")
    print("Dev set:")
    TP = 0
    FP = 0
    FN = 0

    for split_idx in range(2):

        gold_input_pred_file = f"{args.pred_input_pred_path}/split_{split_idx}.json"
        xwlp_test = load_from_jsonl(gold_input_pred_file)

        test_gold_ent_dict = defaultdict(list)
        test_pred_ent_dict = defaultdict(list)

        for doc_data in xwlp_test:
            doc_name = doc_data['doc_key']
            test_gold_ent_dict[doc_name] = [item1 for item in doc_data['tl_ner_tokenized']
                                            for item1 in item if item1[-1].endswith('-op')]
            test_pred_ent_dict[doc_name] = [item1 for item in doc_data['tl_ner_pred_tokenized']
                                            for item1 in item if item1[-1].endswith('-op')]

        assert sorted(test_pred_ent_dict.keys()) == sorted(test_gold_ent_dict.keys())

        for doc_name in test_pred_ent_dict.keys():
            #     print(doc_name)

            gold_ent_list = test_gold_ent_dict[doc_name]
            pred_ent_list = test_pred_ent_dict[doc_name]

            gold_ent_str_list = [' '.join([str(item1) for item1 in item[:3]]) \
                                 for item in gold_ent_list if item[-1] != 'ignored']
            pred_ent_str_list = [' '.join([str(item1) for item1 in item[:3]]) \
                                 for item in pred_ent_list if item[-1] != 'ignored']

            TP += len(set(gold_ent_str_list) & set(pred_ent_str_list))
            FP += (len(set(pred_ent_str_list)) - len(set(gold_ent_str_list) & set(pred_ent_str_list)))
            FN += (len(set(gold_ent_str_list)) - len(set(gold_ent_str_list) & set(pred_ent_str_list)))

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

    for split_idx in range(2, 5):

        gold_input_pred__file = f"{args.pred_input_pred_path}/split_{split_idx}.json"
        xwlp_test = load_from_jsonl(gold_input_pred__file)

        test_gold_ent_dict = defaultdict(list)
        test_pred_ent_dict = defaultdict(list)

        for doc_data in xwlp_test:
            doc_name = doc_data['doc_key']
            test_gold_ent_dict[doc_name] = [item1 for item in doc_data['tl_ner_tokenized']
                                            for item1 in item if item1[-1].endswith('-op')]
            test_pred_ent_dict[doc_name] = [item1 for item in doc_data['tl_ner_pred_tokenized']
                                            for item1 in item if item1[-1].endswith('-op')]

        assert sorted(test_pred_ent_dict.keys()) == sorted(test_gold_ent_dict.keys())

        for doc_name in test_pred_ent_dict.keys():
            #     print(doc_name)

            gold_ent_list = test_gold_ent_dict[doc_name]
            pred_ent_list = test_pred_ent_dict[doc_name]

            gold_ent_str_list = [' '.join([str(item1) for item1 in item[:3]]) \
                                 for item in gold_ent_list if item[-1] != 'ignored']
            pred_ent_str_list = [' '.join([str(item1) for item1 in item[:3]]) \
                                 for item in pred_ent_list if item[-1] != 'ignored']

            TP += len(set(gold_ent_str_list) & set(pred_ent_str_list))
            FP += (len(set(pred_ent_str_list)) - len(set(gold_ent_str_list) & set(pred_ent_str_list)))
            FN += (len(set(gold_ent_str_list)) - len(set(gold_ent_str_list) & set(pred_ent_str_list)))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    print("Precision-Score: {:.1f}".format(precision * 100))
    print("Recall-Score: {:.1f}".format(recall * 100))
    print("F1-Score: {:.1f}".format(f1 * 100))
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_input_pred_path", default=None, type=str, required=True)
    parser.add_argument("--pred_input_pred_path", default=None, type=str, required=True)

    args = parser.parse_args()

    main(args)