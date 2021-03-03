

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
    tokenizer.add_tokens(WLP_ENT_START)
    tokenizer.add_tokens(WLP_ENT_END)
    print(len(tokenizer))

    doc_data_all = load_from_jsonl(args.xwlp_path)

    with open("data/cross_validate.json", 'r') as f:
        cross_validate = json.load(f)

    for split_idx in range(5):

        train_split = cross_validate[f"split_{split_idx}"]["train_list"]
        test_split = cross_validate[f"split_{split_idx}"]["test_list"]

        xwlp_train = [doc_data for doc_data in doc_data_all if doc_data["doc_key"] in train_split]
        xwlp_test = [doc_data for doc_data in doc_data_all if doc_data["doc_key"] in test_split]

        train_inputs, train_masks, train_labels, train_start_idx, \
        train_tokenized_sen_list, train_wlp_ent_type_list, train_doc_name_list, \
        train_ent_info_list = prepare_entity_typing_data(xwlp_train, tokenizer, args)

        print(train_inputs.shape)

        test_inputs, test_masks, test_labels, test_start_idx, \
        test_tokenized_sen_list, test_wlp_ent_type_list, test_doc_name_list, \
        test_ent_info_list = prepare_entity_typing_data(xwlp_test, tokenizer, args)

        print(test_inputs.shape)

        train_data = TensorDataset(train_inputs, train_masks, train_labels, train_start_idx)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

        test_data = TensorDataset(test_inputs, test_masks, test_labels, test_start_idx)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)

        model = BertSimpleEntityMaskTyping.from_pretrained(args.lm, num_labels=len(LABEL2ID))
        model.resize_token_embeddings(len(tokenizer))

        if n_gpu > 1:
            model.to(device)
            model = torch.nn.DataParallel(model)
        else:
            model.cuda()

        optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=False)

        train_loss_set = []
        best_f1 = 0
        saved_model_dir = f"{args.output_dir}/split_{split_idx}"
        print(saved_model_dir)

        # trange is a tqdm wrapper around the normal python range
        for epoch_num in trange(args.epochs, desc="Epoch"):

            # Set our model to training mode (as opposed to evaluation mode)
            model.train()

            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # Train the data for one epoch
            for step, batch in enumerate(train_dataloader):

                if step % 100 == 0 and step > 0:
                    print("The number of steps: {}, {}".format(step, datetime.now()))

                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels, b_start_idx = batch
                #         print(b_input_ids.size(), b_input_mask.size(), b_labels.size(), b_subj_idx.size(), b_obj_idx.size())
                # Clear out the gradients (by default they accumulate)
                optimizer.zero_grad()
                # Forward pass
                loss, _ = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels,
                                ent_start=b_start_idx
                                )
                if n_gpu > 1:
                    loss = loss.mean()

                train_loss_set.append(loss)
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # Update parameters and take a step using the computed gradient
                #         scheduler.step()
                optimizer.step()

                # Update tracking variables
                tr_loss += loss
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            print("Epoch {} - Train loss: {}".format(epoch_num, tr_loss / nb_tr_steps), '\n')

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

            print("\nThe performance on test set: ")
            prec_test, rec_test, f1_test = score([ID2LABEL[gold_id] for gold_id in test_gold],
                                                 [ID2LABEL[pred_id] for pred_id in test_pred],
                                                 verbose=False
                                                 )
            print("Precision-Score: {}".format(prec_test))
            print("Recall-Score: {}".format(rec_test))
            print("F1-Score: {}".format(f1_test))
            print()

            pred_dict = {}
            pred_dict["pred_tags"] = [ID2LABEL[pred_id] for pred_id in test_pred]
            pred_dict["gold_tags"] = [ID2LABEL[gold_id] for gold_id in test_gold]
            pred_dict["ent_info_list"] = test_ent_info_list
            pred_dict["doc_name_list"] = test_doc_name_list

            assert [item[-1] for item in test_ent_info_list] == [ID2LABEL[gold_id] for gold_id in test_gold]

            assert len(test_ent_info_list) == len(test_doc_name_list)

            # Check if the current model is the best model
            if f1_test > best_f1:
                best_f1 = f1_test

                config = vars(args).copy()

                config['saved_model_dir'] = saved_model_dir
                config['best_performed_epoch'] = epoch_num

                config['precision_test'] = prec_test
                config['recall_test'] = rec_test
                config['f1_test'] = f1_test

                print(config, '\n')

                # Save the model
                if args.save_model:

                    if not os.path.exists(saved_model_dir):
                        os.makedirs(saved_model_dir)

                    print("\nSave the best model to {}".format(saved_model_dir))
                    if n_gpu > 1:
                        model.module.save_pretrained(save_directory=saved_model_dir)
                    else:
                        model.save_pretrained(save_directory=saved_model_dir)
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