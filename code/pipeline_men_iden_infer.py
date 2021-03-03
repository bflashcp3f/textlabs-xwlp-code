
from utils.conlleval import evaluate    # Credits to conlleval: https://github.com/sighsmile/conlleval/

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


from utils.men_iden import *


def main(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    doc_data_all = load_from_jsonl(args.xwlp_path)

    with open("data/cross_validate.json", 'r') as f:
        cross_validate = json.load(f)

    for split_idx in range(5):

        train_split = cross_validate[f"split_{split_idx}"]["train_list"]
        test_split = cross_validate[f"split_{split_idx}"]["test_list"]

        mi_train = [doc_data for doc_data in doc_data_all if doc_data["doc_key"] in train_split]
        mi_test = [doc_data for doc_data in doc_data_all if doc_data["doc_key"] in test_split]

        test_sens, test_wlp_ner, test_tl_ner, test_rel, test_doc_name = data_preprocess(mi_test)

        tokenizer = BertTokenizer.from_pretrained(args.model_path)

        ner_model = BertForTokenClassification.from_pretrained(args.model_path)

        # model.cuda();
        if n_gpu > 1:
            ner_model.to(device)
            ner_model = torch.nn.DataParallel(ner_model)
        else:
            ner_model.cuda()

        ner_model.eval()

        print("Start tokenization")
        test_tokenized = tokenize_txt(test_sens, tokenizer)

        print("\nStart splitting tokens and labels")
        test_tokens, test_labels = split_token_label(test_tokenized)
        print(max([len(item) for item in test_tokens]))

        test_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in test_tokens], \
                                       maxlen=args.max_len, value=tokenizer.pad_token_id, dtype="long", \
                                       truncating="post", padding="post")
        test_attention_masks = [[float(i != tokenizer.pad_token_id) for i in ii] for ii in test_input_ids]

        test_inputs = torch.tensor(test_input_ids)
        test_masks = torch.tensor(test_attention_masks)
        test_inputs_len = test_inputs.shape[1]
        test_tags = pad_sequences([[TAG2IDX[l] for l in lab] for lab in test_labels],
                                  maxlen=test_inputs_len, value=TAG2IDX["O"], padding="post",
                                  dtype="long", truncating="post")
        test_tags = torch.tensor(test_tags)

        for tmp_idx in range(len(test_tokens)):
            #     tmp_idx = 1
            tmp_ent_list, tmp_ent_idx_list, tmp_ent_type_list = index_ent_in_prediction(test_tokens[tmp_idx],
                                                                                        test_labels[tmp_idx])
            assert tmp_ent_type_list == [item[-1] for item in test_wlp_ner[tmp_idx]]

        test_data = TensorDataset(test_inputs, test_masks, test_tags)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)

        predictions, true_labels = [], []
        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                logits = ner_model(b_input_ids, token_type_ids=None,
                                   attention_mask=b_input_mask)
            logits = logits[0].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)

        #     print("Remove padding:")
        pred_tags_str = [TAG_NAME[p_i] for p_idx, p in enumerate(predictions)
                         for p_i_idx, p_i in enumerate(p)
                         if test_masks[p_idx][p_i_idx]
                         ]
        test_tags_str = [TAG_NAME[l_i.tolist()] for l_idx, l in enumerate(test_tags)
                         for l_i_idx, l_i in enumerate(l)
                         if test_masks[l_idx][l_i_idx]
                         ]

        # https://github.com/sighsmile/conlleval
        prec_test, rec_test, f1_test = evaluate(test_tags_str, pred_tags_str, verbose=True)
        print("\nOn test set: ")
        print("Precision-Score: {}".format(prec_test))
        print("Recall-Score: {}".format(rec_test))
        print("F1-Score: {}".format(f1_test))
        print()

        pred_labels = [[TAG_NAME[p_i] for p_i_idx, p_i in enumerate(p) if test_masks[p_idx][p_i_idx]]
                       for p_idx, p in enumerate(predictions)]

        doc_pred_label_dict = defaultdict(list)
        doc_test_label_dict = defaultdict(list)
        doc_token_dict = defaultdict(list)

        for pred_label_sen, test_label_sen, test_token_sen, doc_name in zip(pred_labels, test_labels, test_tokens,
                                                                            test_doc_name):

            doc_pred_label_dict[doc_name].append(pred_label_sen[1:-1])
            doc_test_label_dict[doc_name].append(test_label_sen[1:-1])
            doc_token_dict[doc_name].append(test_token_sen[1:-1])

            assert len(pred_label_sen) == len(test_label_sen)
            assert len(test_label_sen) == len(test_token_sen)

        mi_test_new = []

        for doc_data in mi_test:

            doc_name = doc_data['doc_key']

            wlp_ner_list = doc_data['wlp_labels']
            tl_ner_list = doc_data['ner']
            rel_list = doc_data['relations']
            sen_list = doc_data['sentences']

            doc_pred_labels = doc_pred_label_dict[doc_name]
            doc_test_labels = doc_test_label_dict[doc_name]
            doc_test_tokens = doc_token_dict[doc_name]

            #     print(tl_ner_list)
            doc_gold_ent_list, doc_gold_ent_idx_list, \
            doc_gold_ent_type_list = index_ent_in_prediction([item1 for item in doc_test_tokens for item1 in item],
                                                             [item1 for item in doc_test_labels for item1 in item])

            doc_pred_ent_list, doc_pred_ent_idx_list, \
            doc_pred_ent_type_list = index_ent_in_prediction([item1 for item in doc_test_tokens for item1 in item],
                                                             [item1 for item in doc_pred_labels for item1 in item])

            assert [item1 for item in wlp_ner_list for item1 in item] == doc_gold_ent_type_list

            tl_ner_list_doc = [item1 for item in tl_ner_list for item1 in item]
            assert len(tl_ner_list_doc) == len(doc_gold_ent_idx_list)

            doc_sen_len_list = [len(item) for item in doc_test_tokens]

            def find_word_sen_idx(doc_sen_len):
                doc_len = 0
                word2sen_idx = {}

                for sen_idx, sen_len in enumerate(doc_sen_len):

                    for i in range(doc_len, doc_len + sen_len):
                        word2sen_idx[i] = sen_idx

                    doc_len += sen_len

                return word2sen_idx

            doc_word2sen_idx = find_word_sen_idx(doc_sen_len_list)

            idx_trans_dict = {}
            tl_ner_list_new = [[] for _ in range(len(doc_test_tokens))]
            wlp_ner_list_new = [[] for _ in range(len(doc_test_tokens))]

            for old_offset_tl_type, new_offset, wlp_type in zip(tl_ner_list_doc, doc_gold_ent_idx_list, \
                                                                doc_gold_ent_type_list):
                #         print(old_offset_tl_type)
                old_start, old_end, tl_type = old_offset_tl_type
                new_start, new_end = new_offset

                sen_idx = doc_word2sen_idx[new_start]

                idx_trans_dict['-'.join([str(old_start), str(old_end)])] = [new_start, new_end]
                tl_ner_list_new[sen_idx].append([new_start, new_end, tl_type])
                wlp_ner_list_new[sen_idx].append([new_start, new_end, wlp_type])

            for sen_idx in range(len(tl_ner_list_new)):
                assert len(tl_ner_list_new[sen_idx]) == len(tl_ner_list[sen_idx])

            wlp_ner_list_pred = [[] for _ in range(len(doc_test_tokens))]
            for new_offset_pred, wlp_type_pred in zip(doc_pred_ent_idx_list, doc_pred_ent_type_list):

                new_start_pred, new_end_pred = new_offset_pred

                sen_idx = doc_word2sen_idx[new_start_pred]

                # Doesn't count the title
                if sen_idx == 0:
                    continue

                wlp_ner_list_pred[sen_idx].append([new_start_pred, new_end_pred, wlp_type_pred])

            assert rel_list[0] == []
            rel_list_new = [[idx_trans_dict['-'.join([str(arg1_start), str(arg1_end)])] + \
                             idx_trans_dict['-'.join([str(arg2_start), str(arg2_end)])] + [rel_str] \
                             for arg1_start, arg1_end, arg2_start, arg2_end, rel_str in rel_per_sen] \
                            for rel_per_sen in rel_list]

            for sen_idx in range(len(rel_list_new)):
                assert len(rel_list_new[sen_idx]) == len(rel_list[sen_idx])

            assert len(rel_list_new) == len(tl_ner_list_new)
            assert len(rel_list_new) == len(wlp_ner_list_new)

            doc_data_new = {}

            doc_data_new['doc_key'] = doc_name
            doc_data_new['sentences_tokenized'] = doc_test_tokens
            doc_data_new['tl_ner_tokenized'] = tl_ner_list_new
            doc_data_new['wlp_ner_tokenized'] = wlp_ner_list_new
            doc_data_new['relations_tokenized'] = rel_list_new
            doc_data_new['wlp_ner_pred_tokenized'] = wlp_ner_list_pred

            mi_test_new.append(doc_data_new)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        output_path = f"{args.output_dir}/split_{split_idx}.json"
        print(output_path)

        with open(output_path, 'w', encoding='utf-8') as output_file:
            for dic in mi_test_new:
                json.dump(dic, output_file)
                output_file.write("\n")

        for doc_data in mi_test_new:
            assert doc_data['tl_ner_tokenized'][0] == []
            assert doc_data['wlp_ner_tokenized'][0] == []
            assert doc_data['relations_tokenized'][0] == []
            assert doc_data['wlp_ner_pred_tokenized'][0] == []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--xwlp_path", default=None, type=str, required=True)
    parser.add_argument("--model_path", default=None, type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16, required=True)
    parser.add_argument('--max_len', type=int, default=512, required=True)
    parser.add_argument("--gpu_ids", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)

    args = parser.parse_args()

    main(args)