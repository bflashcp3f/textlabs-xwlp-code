

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

from datetime import datetime

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from tqdm import tqdm, trange

from utils.pred_ground import *
from models.pred_ground import *


def main(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(n_gpu, torch.cuda.get_device_name(0))

    for split_idx in range(5):

        data_file = f"{args.data_path}/split_{split_idx}.json"
        xwlp_test = load_from_jsonl(data_file)

        test_file_set = [int(item['doc_key'].split('.')[0].split('_')[-1]) for item in xwlp_test]

        print(data_file, len(xwlp_test))

        model_path = f"{args.model_path}/split_{split_idx}"
        print(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)

        test_inputs, test_masks, test_labels, test_start_idx, \
        test_tokenized_sen_list, test_wlp_ent_type_list, test_doc_name_list, \
        test_ent_info_list = prepare_ep_inference_data(xwlp_test, tokenizer, args)

        print(test_inputs.shape)

        test_data = TensorDataset(test_inputs, test_masks, test_labels, test_start_idx)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)

        model = BertSimpleEntityMaskTyping.from_pretrained(model_path)

        if n_gpu > 1:
            model.to(device)
            model = torch.nn.DataParallel(model)
        else:
            model.cuda()

        # Put model in evaluation mode to evaluate loss on the dev set
        model.eval()

        test_gold = []
        test_pred = []

        # Evaluate data for one epoch
        for batch in test_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels, b_start_idx = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up test
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask,
                               ent_start=b_start_idx
                               )

            # Move logits and labels to CPU
            logits = logits[0].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            test_pred += np.argmax(logits, axis=1).tolist()
            test_gold += label_ids.tolist()

        test_gold_ent_dict = defaultdict(list)

        for doc_data in xwlp_test:
            doc_name = doc_data['doc_key']
            test_gold_ent_dict[doc_name] = [item1 for item in doc_data['tl_ner_tokenized']
                                            for item1 in item if item1[-1].endswith('-op')]

        test_pred_tl_labels = [ID2LABEL[pred_id] for pred_id in test_pred]

        test_pred_ent_dict = defaultdict(list)

        for pred_tl_label, doc_name, ent_info in zip(test_pred_tl_labels, test_doc_name_list, test_ent_info_list):

            ent_start_old, ent_end_old, ent_start, ent_end, sen_idx, _ = ent_info
            test_pred_ent_dict[doc_name].append([ent_start_old, ent_end_old, pred_tl_label])

        assert test_pred_ent_dict.keys() == test_gold_ent_dict.keys()

        xwlp_test_new = []

        for doc_data in xwlp_test:
            doc_name = doc_data['doc_key']
            wlp_ner_pred_list = doc_data['wlp_ner_pred_tokenized']
            pred_ops_list = test_pred_ent_dict[doc_name]

            wlp_ner_pred_idx_list = ['-'.join(str(e) for e in item1[:2]) for item in wlp_ner_pred_list for item1 in item]
            pred_ops_idx_dict = dict([('-'.join(str(e) for e in item[:2]), item[2]) for item in pred_ops_list])

            assert len(set(wlp_ner_pred_idx_list) & set(pred_ops_idx_dict.keys())) == len(set(pred_ops_idx_dict.keys()))

            tl_ner_pred_list = [[[ent_start, ent_end, WLP2TL[wlp_ner_label] if wlp_ner_label != 'Action' else pred_ops_idx_dict['-'.join([str(ent_start), str(ent_end)])]]
                                 for ent_start, ent_end, wlp_ner_label in item if wlp_ner_label not in NONE_TYPE] for item in wlp_ner_pred_list]

            assert len(tl_ner_pred_list) == len(wlp_ner_pred_list)

            doc_data_new = doc_data.copy()
            doc_data_new['tl_ner_pred_tokenized'] = tl_ner_pred_list

            xwlp_test_new.append(doc_data_new)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        output_path = f"{args.output_dir}/split_{split_idx}.json"
        print(output_path)

        with open(output_path, 'w', encoding='utf-8') as output_file:
            for dic in xwlp_test_new:
                json.dump(dic, output_file)
                output_file.write("\n")

        TP = 0
        FP = 0
        FN = 0

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

        print(f"\nFor split {split_idx}")
        print(f"TP:{TP}, FP:{FP}, FN:{FN}")
        print("Precision-Score: {:.1f}".format(precision * 100))
        print("Recall-Score: {:.1f}".format(recall * 100))
        print("F1-Score: {:.1f}".format(f1 * 100))
        print()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None, type=str, required=True)
    parser.add_argument("--model_path", default=None, type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16, required=True)
    parser.add_argument('--max_len', type=int, default=512, required=True)
    parser.add_argument("--gpu_ids", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)

    args = parser.parse_args()

    main(args)