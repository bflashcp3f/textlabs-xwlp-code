
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

    print(vars(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(n_gpu, device)

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
    print(len(tokenizer))

    all_file_set = [int(item.split('.')[0].split('_')[-1])
                    for item in os.listdir(args.wlp_path)
                    if '.txt' == item[-4:]]

    with open("data/cross_validate.json", 'r') as f:
        cross_validate = json.load(f)

    dev_file_set = [int(item.split('.')[0].split('_')[-1]) for
                    item in cross_validate["split_0"]["test_list"] + cross_validate["split_1"]["test_list"]]
    test_file_set = [int(item.split('.')[0].split('_')[-1]) for
                     item in cross_validate["split_2"]["test_list"] + cross_validate["split_3"]["test_list"] + cross_validate["split_4"]["test_list"]]
    train_file_set = [item for item in all_file_set if item not in dev_file_set + test_file_set]

    print("TRAIN/DEV/TEST split: ", f"{len(train_file_set)}/{len(dev_file_set)}/{len(test_file_set)}")

    train_sens = get_processed_sentences(train_file_set, args.wlp_path)
    print("\n# TRAIN sentences: ", len(train_sens))

    text = ' '.join(train_sens[0].split(' '))
    # marked_text = "[CLS] " + text + " [SEP]"
    print(text)
    print('----------------------------')
    tokenized_text = tokenizer.tokenize(text)
    print(tokenized_text)
    # print(tokenized_text.index("[tl-transfer-enum-START]"))
    print('----------------------------')
    # print(tokenizer.convert_tokens_to_ids(tokenized_text) == tokenizer.encode(tokenized_text))
    print(tokenizer.convert_tokens_to_ids(tokenized_text))

    train_dataloader, train_inputs, train_masks, train_tags = process_set(train_sens, tokenizer, args, TAG2IDX,
                                                                               train_flag=True)

    dev_sens = get_processed_sentences(dev_file_set, args.wlp_path)
    print("\n# DEV sentences: ", len(dev_sens))

    dev_dataloader, dev_inputs, dev_masks, dev_tags = process_set(dev_sens, tokenizer, args, TAG2IDX,
                                                                       train_flag=False)

    test_sens = get_processed_sentences(test_file_set, args.wlp_path)
    print("\n# TEST sentences: ", len(test_sens))

    test_dataloader, test_inputs, test_masks, test_tags = process_set(test_sens, tokenizer, args, TAG2IDX,
                                                                           train_flag=False)

    ner_model = BertForTokenClassification.from_pretrained(args.lm, num_labels=len(TAG2IDX))

    # model.cuda();
    if n_gpu > 1:
        ner_model.to(device)
        ner_model = torch.nn.DataParallel(ner_model)
    else:
        ner_model.cuda()

    optimizer = AdamW(ner_model.parameters(), lr=args.learning_rate, eps=1e-8)

    best_f1 = 0
    bad_count = 0

    for epoch_num in trange(args.epochs, desc="Epoch"):

        # TRAIN loop
        ner_model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu

            if step % 100 == 0 and step > 0:
                print("The number of steps: {}, {}".format(step, datetime.now()))

            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # forward pass
            loss, _ = ner_model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
            #         print(loss)

            if n_gpu > 1:
                loss = loss.mean()

            # backward pass
            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=ner_model.parameters(), max_norm=args.max_grad_norm)
            # update parameters
            optimizer.step()
            ner_model.zero_grad()
        # print train loss per epoch
        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        # on dev set
        ner_model.eval()

        predictions, true_labels = [], []
        for batch in dev_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                #             tmp_eval_loss = ner_model(b_input_ids, token_type_ids=None,
                #                                   attention_mask=b_input_mask, labels=b_labels)
                logits = ner_model(b_input_ids, token_type_ids=None,
                                   attention_mask=b_input_mask)
            logits = logits[0].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)

        #     print("Remove padding:")
        pred_tags_str = [TAG_NAME[p_i] for p_idx, p in enumerate(predictions)
                         for p_i_idx, p_i in enumerate(p)
                         if dev_masks[p_idx][p_i_idx]
                         ]
        dev_tags_str = [TAG_NAME[l_i.tolist()] for l_idx, l in enumerate(dev_tags)
                        for l_i_idx, l_i in enumerate(l)
                        if dev_masks[l_idx][l_i_idx]
                        ]

        # https://github.com/sighsmile/conlleval
        prec_dev, rec_dev, f1_dev = evaluate(dev_tags_str, pred_tags_str, verbose=False)
        print("\nOn dev set: ")
        print("Precision-Score: {}".format(prec_dev))
        print("Recall-Score: {}".format(rec_dev))
        print("F1-Score: {}".format(f1_dev))
        print()

        # on test set
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
        prec_test, rec_test, f1_test = evaluate(test_tags_str, pred_tags_str, verbose=False)
        print("\nOn test set: ")
        print("Precision-Score: {}".format(prec_test))
        print("Recall-Score: {}".format(rec_test))
        print("F1-Score: {}".format(f1_test))
        print()

        # Check if the current model is the best model
        if f1_dev > best_f1:
            best_f1 = f1_dev

            config = vars(args).copy()

            config['output_dir'] = args.output_dir
            config['best_performed_epoch'] = epoch_num

            config['precision_dev'] = prec_dev
            config['recall_dev'] = rec_dev
            config['f1_dev'] = f1_dev

            config['precision_test'] = prec_test
            config['recall_test'] = rec_test
            config['f1_test'] = f1_test

            print(config, '\n')

            # Save the model
            if args.save_model:

                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)

                print("\nSave the best model to {}".format(args.output_dir))
                if n_gpu > 1:
                    ner_model.module.save_pretrained(save_directory=args.output_dir)
                else:
                    ner_model.save_pretrained(save_directory=args.output_dir)

                tokenizer.save_pretrained(save_directory=args.output_dir)

                # Save hyper-parameters (lr, batch_size, epoch, precision, recall, f1)
                config_path = os.path.join(args.output_dir, 'self_config.json')
                with open(config_path, 'w') as json_file:
                    json.dump(config, json_file)

            bad_count = 0
        else:
            bad_count += 1

            if bad_count == args.patient:
                sys.exit("Reach the PATIENT!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wlp_path", default=None, type=str)
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