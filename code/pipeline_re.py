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

from collections import defaultdict, Counter, OrderedDict
from datetime import datetime

import torch
from transformers import RobertaTokenizer, RobertaForTokenClassification, RobertaConfig, RobertaModel
from transformers import AutoModelForTokenClassification, AutoConfig, AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel
from transformers import BertForTokenClassification, BertPreTrainedModel, AdamW
from transformers import AdamW
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import KFold
from ipywidgets import IntProgress

from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from tqdm import tqdm, trange

from utils.re import *
from models.re import *


def main(args):

    print(vars(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(n_gpu, torch.cuda.get_device_name(0))

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.random_seed)

    tokenizer = BertTokenizer.from_pretrained(args.lm)
    tokenizer.bos_token = '[CLS]'
    tokenizer.eos_token = '[SEP]'
    tokenizer.unk_token = '[UNK]'
    tokenizer.sep_token = '[SEP]'
    tokenizer.cls_token = '[CLS]'
    tokenizer.mask_token = '[MASK]'
    tokenizer.pad_token = '[PAD]'

    print(len(tokenizer))
    tokenizer.add_tokens(ENT_NAME_END)
    tokenizer.add_tokens(ENT_NAME_START)
    tokenizer.add_tokens(ENT_ID)
    print(len(tokenizer))

    doc_data_all = load_from_jsonl(args.xwlp_path)

    with open("data/cross_validate.json", 'r') as f:
        cross_validate = json.load(f)

    for split_idx in range(5):

        train_split = cross_validate[f"split_{split_idx}"]["train_list"]
        test_split = cross_validate[f"split_{split_idx}"]["test_list"]

        xwlp_train = [doc_data for doc_data in doc_data_all if doc_data["doc_key"] in train_split]
        xwlp_test = [doc_data for doc_data in doc_data_all if doc_data["doc_key"] in test_split]

        rel2ep_type = defaultdict(set)
        ep_type2rel = defaultdict(set)
        rel2arg_dis = defaultdict(list)
        ep_type2arg_dis = defaultdict(list)

        for doc_data in xwlp_train:

            ner_start_label_dict = dict([(item1[0], item1[-1]) for item in doc_data['ner'] for item1 in item])
            ner_start_sen_idx_dict = dict([(item1[0], item_idx) for item_idx, item in enumerate(doc_data['ner']) for item1 in item])

            for rel_list in doc_data['relations']:
                for rel_men in rel_list:
                    rel2ep_type[rel_men[-1]].add('-'.join([ner_start_label_dict[rel_men[0]], ner_start_label_dict[rel_men[2]]]))
                    ep_type2rel['-'.join([ner_start_label_dict[rel_men[0]], ner_start_label_dict[rel_men[2]]])].add(rel_men[-1])
                    rel2arg_dis[rel_men[-1]].append(ner_start_sen_idx_dict[rel_men[0]] - \
                                                    ner_start_sen_idx_dict[rel_men[2]])
                    ep_type2arg_dis['-'.join([ner_start_label_dict[rel_men[0]], ner_start_label_dict[rel_men[2]]])].append(
                        ner_start_sen_idx_dict[rel_men[0]] - \
                        ner_start_sen_idx_dict[rel_men[2]])

        train_input_ids, train_masks, train_labels, train_arg1_idx, train_arg2_idx, train_arg_info_list_all, \
        train_doc_name_list_all = prepare_data(xwlp_train[:], ALL_POS_RELS, rel2ep_type, ep_type2rel, \
                                               tokenizer, args, down_sample=True)

        print("Train cases: ", train_input_ids.shape[0])
        print(Counter([ID2LABEL[item.tolist()] for item in train_labels[:]]))

        test_input_ids, test_masks, test_labels, test_arg1_idx, test_arg2_idx, test_arg_info_list_all, \
        test_doc_name_list_all = prepare_data(xwlp_test[:], ALL_POS_RELS, rel2ep_type, ep_type2rel, tokenizer, args)

        print("Test cases: ", test_input_ids.shape[0])
        print(Counter([ID2LABEL[item.tolist()] for item in test_labels[:]]))

        train_data = TensorDataset(train_input_ids, train_masks, train_labels, train_arg1_idx, train_arg2_idx)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

        test_data = TensorDataset(test_input_ids, test_masks, test_labels, test_arg1_idx, test_arg2_idx)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)

        re_model = BertSimpleEMES.from_pretrained(args.lm, num_labels=len(LABEL2ID) + 1)
        re_model.resize_token_embeddings(len(tokenizer))

        if n_gpu > 1:
            re_model.to(device)
            re_model = torch.nn.DataParallel(re_model)
        else:
            re_model.cuda()

        optimizer = AdamW(re_model.parameters(), lr=args.learning_rate, correct_bias=False)

        train_loss_set = []
        best_f1 = 0
        saved_model_dir = f"{args.output_dir}/split_{split_idx}"
        print(saved_model_dir)

        # trange is a tqdm wrapper around the normal python range
        for num_epoch in trange(args.epochs, desc="Epoch"):

            # Training
            # Set our model to training mode (as opposed to evaluation mode)
            re_model.train()

            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # Train the data for one epoch
            for step, batch in enumerate(train_dataloader):

                #         if step % 100 == 0 and step > 0:
                if step % 100 == 0:
                    print(step, datetime.now())

                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels, b_subj_idx, b_obj_idx = batch
                #         print(b_input_ids.size(), b_input_mask.size(), b_labels.size(), b_subj_idx.size(), b_obj_idx.size())
                # Clear out the gradients (by default they accumulate)
                optimizer.zero_grad()
                # Forward pass
                loss, _ = re_model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels,
                                   subj_ent_start=b_subj_idx, obj_ent_start=b_obj_idx
                                   )
                if n_gpu > 1:
                    loss = loss.mean()

                train_loss_set.append(loss)
                # Backward pass
                loss.backward()
                optimizer.step()

                # Update tracking variables
                tr_loss += loss
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
            #         break

            print("Train loss: {}".format(tr_loss / nb_tr_steps))

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

            print("\nThe performance on test set: ")
            prec_test, rec_test, f1_test = score([ID2LABEL[gold_id] for gold_id in test_gold],
                                                 [ID2LABEL[pred_id] for pred_id in test_pred],
                                                 verbose=True
                                                 )
            print("Precision-Score: {}".format(prec_test))
            print("Recall-Score: {}".format(rec_test))
            print("F1-Score: {}".format(f1_test))
            print(Counter([ID2LABEL[gold_id] for gold_id in test_gold]))
            print(Counter([ID2LABEL[pred_id] for pred_id in test_pred]))
            print()

            pred_dict = {}
            pred_dict["pred_tags"] = [ID2LABEL[pred_id] for pred_id in test_pred]
            pred_dict["gold_tags"] = [ID2LABEL[gold_id] for gold_id in test_gold]
            pred_dict["arg_info_list"] = test_arg_info_list_all
            pred_dict["doc_name_list"] = test_doc_name_list_all

            assert len(test_arg_info_list_all) == len(test_doc_name_list_all)
            assert [ID2LABEL[gold_id] for gold_id in test_gold] == [item[0] for item in test_arg_info_list_all]

            # Check if the current model is the best model
            if f1_test > best_f1:
                best_f1 = f1_test

                config = vars(args).copy()

                config['saved_model_dir'] = saved_model_dir

                config['precision_test'] = prec_test
                config['recall_test'] = rec_test
                config['f1_test'] = f1_test

                config['best_epoch'] = num_epoch

                print(config, '\n')

                # Save the model
                if args.save_model:

                    if not os.path.exists(saved_model_dir):
                        os.makedirs(saved_model_dir)

                    print("\nSave the best model to {}".format(saved_model_dir))
                    if n_gpu > 1:
                        re_model.module.save_pretrained(save_directory=saved_model_dir)
                    else:
                        re_model.save_pretrained(save_directory=saved_model_dir)
                    tokenizer.save_pretrained(save_directory=saved_model_dir)

                    # Save hyper-parameters (lr, batch_size, epoch, precision, recall, f1)
                    config_path = os.path.join(saved_model_dir, 'self_config.json')
                    with open(config_path, 'w') as json_file:
                        json.dump(config, json_file)

                    predictions_path = os.path.join(saved_model_dir, 'predictions.json')
                    with open(predictions_path, 'w') as json_file:
                        json.dump(pred_dict, json_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--xwlp_path", default=None, type=str)
    parser.add_argument("--lm", default=None, type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16, required=True)
    parser.add_argument('--max_len', type=int, default=512, required=True)
    parser.add_argument('--patient', type=int, default=30)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument("--gpu_ids", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument('--random_seed', type=int, default=1234,
                        help="random seed for random library")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--save_model", action='store_true', help="Save trained checkpoints.")

    args = parser.parse_args()

    main(args)



