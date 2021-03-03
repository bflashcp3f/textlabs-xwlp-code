

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


from utils.re import *
from models.re import *


def main(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(n_gpu, torch.cuda.get_device_name(0))

    doc_data_all = load_from_jsonl(args.xwlp_path)

    with open("data/cross_validate.json", 'r') as f:
        cross_validate = json.load(f)

    test_pred_rel_dict_list = []
    test_gold_rel_dict_list = []

    for split_idx in range(5):

        model_path = f"{args.model_path}/split_{split_idx}"
        print(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)

        train_split = cross_validate[f"split_{split_idx}"]["train_list"]
        test_split = cross_validate[f"split_{split_idx}"]["test_list"]

        xwlp_train = [doc_data for doc_data in doc_data_all if doc_data["doc_key"] in train_split]
        # xwlp_test = [doc_data for doc_data in doc_data_all if doc_data["doc_key"] in test_split]

        data_file = f"{args.data_path}/split_{split_idx}.json"
        xwlp_test = load_from_jsonl(data_file)


        rel2ep_type = defaultdict(set)
        ep_type2rel = defaultdict(set)
        rel2arg_dis = defaultdict(list)
        ep_type2arg_dis = defaultdict(list)

        for doc_data in xwlp_train:

            ner_start_label_dict = dict([(item1[0], item1[-1]) for item in doc_data['ner'] for item1 in item])
            ner_start_sen_idx_dict = dict(
                [(item1[0], item_idx) for item_idx, item in enumerate(doc_data['ner']) for item1 in item])

            for rel_list in doc_data['relations']:
                for rel_men in rel_list:
                    #             print(rel_men[-1], ner_start_idx_dict[rel_men[0]], ner_start_idx_dict[rel_men[2]], )
                    rel2ep_type[rel_men[-1]].add('-'.join([ner_start_label_dict[rel_men[0]], \
                                                           ner_start_label_dict[rel_men[2]]]))
                    ep_type2rel['-'.join([ner_start_label_dict[rel_men[0]], ner_start_label_dict[rel_men[2]]])].add(
                        rel_men[-1])
                    rel2arg_dis[rel_men[-1]].append(ner_start_sen_idx_dict[rel_men[0]] - \
                                                    ner_start_sen_idx_dict[rel_men[2]])
                    ep_type2arg_dis[
                        '-'.join([ner_start_label_dict[rel_men[0]], ner_start_label_dict[rel_men[2]]])].append(
                        ner_start_sen_idx_dict[rel_men[0]] - \
                        ner_start_sen_idx_dict[rel_men[2]])


        test_input_ids, test_masks, test_labels, \
        test_arg1_idx, test_arg2_idx, test_arg_info_list_all, \
        test_doc_name_list_all = prepare_inference_data(xwlp_test[:], ALL_POS_RELS, rel2ep_type, ep_type2rel,
                                                        tokenizer, args)

        print("Test: ", test_input_ids.shape[0])
        print(test_input_ids.shape[0] - Counter([item.tolist() for item in test_labels[:]])[0], test_input_ids.shape[0])

        test_data = TensorDataset(test_input_ids, test_masks, test_labels, test_arg1_idx, test_arg2_idx)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)

        re_model = BertSimpleEMES.from_pretrained(model_path)

        if n_gpu > 1:
            re_model.to(device)
            re_model = torch.nn.DataParallel(re_model)
        else:
            re_model.cuda()

        # Put model in evaluation mode to evaluate loss on the dev set
        re_model.eval()

        test_gold = []
        test_pred = []

        # Evaluate data for one epoch
        for step, batch in enumerate(test_dataloader):

            if step % 100 == 0:
                print(step, datetime.now())

            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels, b_subj_idx, b_obj_idx = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up test
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = re_model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask,
                                  subj_ent_start=b_subj_idx, obj_ent_start=b_obj_idx
                                  )

            # Move logits and labels to CPU
            logits = logits[0].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            test_pred += np.argmax(logits, axis=1).tolist()
            test_gold += label_ids.tolist()

        assert len(test_arg_info_list_all) == len(test_doc_name_list_all)
        assert [ID2LABEL[gold_id] for gold_id in test_gold] == [item[0] for item in test_arg_info_list_all]

        test_gold_rel_dict = defaultdict(list)

        for doc_data in xwlp_test:
            doc_name = doc_data['doc_key']
            test_gold_rel_dict[doc_name] = [item1 for item in doc_data['relations_tokenized']
                                            for item1 in item if item1[-1] in ALL_POS_RELS]

        test_pred_tl_labels = [ID2LABEL[pred_id] for pred_id in test_pred]

        test_pred_rel_dict = defaultdict(list)

        for pred_tl_label, doc_name, arg_info in zip(test_pred_tl_labels, test_doc_name_list_all,
                                                     test_arg_info_list_all):

            _, arg1_info, arg2_info = arg_info
            arg1_id, arg1_str, arg1_offset, arg1_type, arg1_senid = arg1_info
            arg2_id, arg2_str, arg2_offset, arg2_type, arg2_senid = arg2_info

            arg1_start, arg1_end = arg1_offset
            arg2_start, arg2_end = arg2_offset

            arg1_end = arg1_end - 1
            arg2_end = arg2_end - 1

            test_pred_rel_dict[doc_name].append([arg1_start, arg1_end, arg2_start, arg2_end, pred_tl_label])

        TP = 0
        FP = 0
        FN = 0

        assert sorted(test_pred_rel_dict.keys()) == sorted(test_gold_rel_dict.keys())

        for doc_name in test_pred_rel_dict.keys():
            #     print(doc_name)

            gold_rel_list = test_gold_rel_dict[doc_name]
            pred_rel_list = test_pred_rel_dict[doc_name]

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

        print(f"\nFor split {split_idx}")
        print(f"TP:{TP}, FP:{FP}, FN:{FN}")
        print("Precision-Score: {:.1f}".format(precision * 100))
        print("Recall-Score: {:.1f}".format(recall * 100))
        print("F1-Score: {:.1f}".format(f1 * 100))
        print()

        test_gold_rel_dict_list.append(test_gold_rel_dict)
        test_pred_rel_dict_list.append(test_pred_rel_dict)

        xwlp_test_new = []

        for doc_data in xwlp_test:

            doc_name = doc_data['doc_key']
            gold_rel_list = test_gold_rel_dict[doc_name]
            pred_rel_list = test_pred_rel_dict[doc_name]

            gold_rel_list = [item for item in gold_rel_list if item[-1] != 'no_relation']
            pred_rel_list = [item for item in pred_rel_list if item[-1] != 'no_relation']

            doc_data_new = doc_data.copy()
            doc_data_new['tl_rel_gold_tokenized'] = gold_rel_list
            doc_data_new['tl_rel_pred_tokenized'] = pred_rel_list

            xwlp_test_new.append(doc_data_new)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        output_path = f"{args.output_dir}/split_{split_idx}.json"
        print(output_path)

        with open(output_path, 'w', encoding='utf-8') as output_file:
            for dic in xwlp_test_new:
                json.dump(dic, output_file)
                output_file.write("\n")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--xwlp_path", default=None, type=str, required=True)
    parser.add_argument("--data_path", default=None, type=str, required=True)
    parser.add_argument("--model_path", default=None, type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16, required=True)
    parser.add_argument('--max_len', type=int, default=512, required=True)
    parser.add_argument("--gpu_ids", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)

    args = parser.parse_args()

    main(args)