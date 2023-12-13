# Training script

import os
import sys
import pickle
import json
import time
import copy
from fastprogress import master_bar, progress_bar
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer

from utils.arguments import get_train_args
from utils.data_prep import build_data_loader
from utils.load_name_list import get_alias2id
from utils.bert_features import *
from utils.training_control import *
from model.model import CSN


# training log
LOG_FORMAT = '%(asctime)s %(name)s %(levelname)s %(pathname)s %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%m:%s %a'


def train():
    """
    Training script.

    return
        best_dev_acc: the best development accuracy.
        best_test_acc: the accuracy on test instances of the model that has the best performance on development instances.
    """
    args = get_train_args()
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())

    print("#######################OPTIONS########################")
    print(json.dumps(vars(args), indent=4))

    # checkpoint
    checkpoint_dir = os.path.join(args.checkpoint_dir, 
                                  os.path.join(args.model_name, timestamp))

    # logging
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'tensorboard'))

    logging_name = os.path.join(checkpoint_dir, 'training_log.log')
    logging.basicConfig(level=logging.INFO,
                        format=LOG_FORMAT,
                        datefmt=DATE_FORMAT,
                        filename=logging_name)

    # device
    device = torch.device('cpu')

    # data files
    train_file = args.train_file
    dev_file = args.dev_file
    test_file = args.test_file
    name_list_path = args.name_list_path
    namelist_file = args.namelist_file

    # 작품 별 등장인물 이름 가져오기
    with open(namelist_file, 'r', encoding='utf-8') as file:
        name_list_per_novel = json.load(file)

    alias2id = get_alias2id(name_list_path)

    # build training, development and test data loaders
    train_data = build_data_loader(train_file, alias2id, args, skip_only_one=True)
    print("The number of training instances: " + str(len(train_data)))
    dev_data = build_data_loader(dev_file, alias2id, args)
    print("The number of development instances: " + str(len(dev_data)))
    test_data = build_data_loader(test_file, alias2id, args)
    print("The number of test instances: " + str(len(test_data)))

    ### 여기까지 분리해놓기. 

    # example
    print('##############DEV EXAMPLE#################')
    dev_test_iter = iter(dev_data)
    _, CSSs, sent_char_lens, mention_poses, quote_idxes, cut_css, one_hot_label, true_index, category, Name, Scene, Place, Time, Cut_position, candidates_list, instance_index = dev_test_iter.next()
    print('Candidate-specific segments:')
    print(CSSs)
    print('Nearest mention positions:')
    print(mention_poses)
    test_test_iter = iter(test_data)
    print('##############TEST EXAMPLE#################')
    _, CSSs, sent_char_lens, mention_poses, quote_idxes, cut_css, one_hot_label, true_index, category, Name, Scene, Place, Time, Cut_position, candidates_list, instance_index = test_test_iter.next()
    print('Candidate-specific segments:')
    print(CSSs)
    print('Nearest mention positions:')
    print(mention_poses)

    # initialize model
    tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrained_dir)
    model = CSN(args)
    model = model.to(device)

    # initialize optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Unknown optimizer type...")

    # loss criterion
    loss_fn = nn.MarginRankingLoss(margin=args.margin)

    # training loop
    print("############################Training Begins...################################")

    # logging best
    best_overall_dev_acc = 0
    best_explicit_dev_acc = 0
    best_implicit_dev_acc = 0
    best_latent_dev_acc = 0
    best_dev_loss = 0
    new_best = False

    # control parameters
    patience_counter = 0
    backward_counter = 0

    epoch_bar = master_bar(range(args.num_epochs))
    for epoch in epoch_bar:
        acc_numerator = 0
        acc_denominator = 0
        train_loss = 0

        model.train()
        optimizer.zero_grad()

        print('Epoch: %d' % (epoch + 1))
        for i, (_, CSSs, sent_char_lens, mention_poses, quote_idxes, cut_css, one_hot_label, true_index, category, Name, Scene, Place, Time, Cut_position, candidates_list, instance_index) \
            in enumerate(progress_bar(train_data, total=len(train_data), parent=epoch_bar)):
            
            try:
                features, tokens_list = convert_examples_to_features(CSSs, tokenizer)
                scores, scores_false, scores_true = model(features, sent_char_lens, mention_poses, quote_idxes, true_index, device, tokens_list, cut_css)
                if scores == '1':
                    continue
                # backward propagation and weights update
                for x, y in zip(scores_false, scores_true):
                    # compute loss
                    loss = loss_fn(x.unsqueeze(0), y.unsqueeze(0), torch.tensor(-1.0).unsqueeze(0).to(device))
                    train_loss += loss.item()
                    
                    # backward propagation
                    loss /= args.batch_size
                    loss.backward(retain_graph=True)
                    backward_counter += 1

                    # update parameters
                    if backward_counter % args.batch_size == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                # training accuracy
                acc_numerator += 1 if scores.max(0)[1].item() == true_index else 0
                acc_denominator += 1

            except RuntimeError:
                print('OOM occurs...')

        acc = acc_numerator / acc_denominator
        train_loss /= len(train_data)

        # logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', acc, epoch)
        logging.info('train_acc: %.4f' % (acc))
        print('train_acc: %.4f' % (acc))
        print('train_loss: %.4f' % (train_loss))

        # adjust learning rate after each epoch
        adjust_learning_rate(optimizer, args.lr_decay)

        # Evaluation
        model.eval()

        def eval(eval_data, subset_name):
            """
            Evaluate performance on a given subset.

            params
                eval_data: the set of instances to be evaluate on.
                subset_name: the name of the subset for logging.

            return
                acc_numerator_sub: the number of correct predictions.
                acc_denominator_sub: the total number of instances.
                sum_loss: the sum of evaluation loss on positive-negative pairs.
            """
            overall_eval_acc_numerator = 0
            overall_eval_acc_denominator = len(eval_data)
            explicit_eval_acc_numerator = 0
            explicit_eval_acc_denominator = 0
            implicit_eval_acc_numerator = 0
            implicit_eval_acc_denominator = 0
            latent_eval_acc_numerator = 0
            latent_eval_acc_denominator = 0

            eval_sum_loss = 0

            for _, CSSs, sent_char_lens, mention_poses, quote_idxes,  cut_css, _, true_index, category, Name, Scene, Place, Time, Cut_position, candidates_list, instance_index \
                in progress_bar(eval_data, total=len(eval_data), parent=epoch_bar):
                
                with torch.no_grad():
                    # features = convert_examples_to_features(examples=CSSs, tokenizer=tokenizer)
                    features, tokens_list = convert_examples_to_features(CSSs, tokenizer)
                    scores, scores_false, scores_true = model(features, sent_char_lens, mention_poses, quote_idxes, true_index, device, tokens_list, cut_css)
                    # if quotes have unk, ignore that case
                    # 인용문이 비어 있을 때 대체
                    if scores == '1':
                        continue
                    loss_list = [loss_fn(x.unsqueeze(0), y.unsqueeze(0), torch.tensor(-1.0).unsqueeze(0).to(device)) for x, y in zip(scores_false, scores_true)]

                eval_sum_loss += sum(x.item() for x in loss_list)

                # evaluate accuracy
                correct = 1 if scores.max(0)[1].item() == true_index else 0
                overall_eval_acc_numerator += correct
                # if category == 'explicit':
                #     explicit_eval_acc_numerator += correct
                #     explicit_eval_acc_denominator += 1
                # if category == 'implicit':
                #     implicit_eval_acc_numerator += correct
                #     implicit_eval_acc_denominator += 1
                if category == 'latent':
                    latent_eval_acc_numerator += correct
                    latent_eval_acc_denominator += 1
                if correct == 0:
                    print('아쉽게도 틀렸습니다.')
                    print(f'{Name} 작품의 인스턴스 {instance_index}에서 에러가 났습니다. {candidates_list}에서 {true_index}를 맞췄어야 하는데 {scores.max(0)[1].item()}을 골랐습니다.')
                elif correct == 1:
                    print('대단해요! 맞췄어요.')
                    name_idx = candidates_list[true_index]
                    if len(str(name_idx)) == 1:
                        name_idx = '0'+str(name_idx)
                    chr_name = f'&C{name_idx}&'
                    rst_name = [name_list_per_novel[idx][chr_name][0] for idx, item in enumerate(name_list_per_novel) if item['title'] == Name+'  ']
                    print(f'{Name} 작품의 인스턴스 {instance_index}의 발화자는 {rst_name}입니다.')
                    print(f'장소는 {Place}이고, 시간은 {Time} 입니다.')


            overall_eval_acc = overall_eval_acc_numerator / overall_eval_acc_denominator
            # explicit_eval_acc = explicit_eval_acc_numerator / explicit_eval_acc_denominator
            # implicit_eval_acc = implicit_eval_acc_numerator / implicit_eval_acc_denominator
            latent_eval_acc = latent_eval_acc_numerator / latent_eval_acc_denominator
            eval_avg_loss = eval_sum_loss / overall_eval_acc_denominator

            # logging
            writer.add_scalar('Loss/' + subset_name, eval_avg_loss, epoch)
            writer.add_scalar('Accuracy/' + subset_name, overall_eval_acc, epoch)
            logging.info(subset_name + '_overall_acc: %.4f' % (overall_eval_acc))
            print(subset_name + '_overall_acc: %.4f' % (overall_eval_acc))
            # print(subset_name + '_explicit_acc: %.4f' % (explicit_eval_acc))
            # print(subset_name + '_implicit_acc: %.4f' % (implicit_eval_acc))
            print(subset_name + '_latent_acc: %.4f' % (latent_eval_acc))
            print(subset_name + '_overall_loss: %.4f' % (eval_avg_loss))

            # return overall_eval_acc, explicit_eval_acc, implicit_eval_acc, latent_eval_acc, eval_avg_loss
            return overall_eval_acc, latent_eval_acc, latent_eval_acc, latent_eval_acc, eval_avg_loss

        # development stage
        overall_dev_acc, explicit_dev_acc, implicit_dev_acc, latent_dev_acc, dev_avg_loss = eval(dev_data, 'dev')
        # overall_dev_acc, latent_dev_acc, latent_dev_acc, latent_dev_acc, dev_avg_loss = eval(dev_data, 'dev')

        # save the model with best performance
        if overall_dev_acc > best_overall_dev_acc:
            best_overall_dev_acc = overall_dev_acc
            best_explicit_dev_acc = explicit_dev_acc
            best_implicit_dev_acc = implicit_dev_acc
            # best_explicit_dev_acc = latent_dev_acc
            # best_implicit_dev_acc = latent_dev_acc
            best_latent_dev_acc = latent_dev_acc
            best_dev_loss = dev_avg_loss
            
            patience_counter = 0
            new_best = True
        else:
            patience_counter += 1
            new_best = False

        # only save the model which outperforms the former best on development set
        if new_best:
            # test stage
            overall_test_acc, explicit_test_acc, implicit_test_acc, latent_test_acc, test_avg_loss = eval(test_data, 'test')
            try:
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    {
                    'args': vars(args),
                    'training_loss': train_loss,
                    'best_overall_dev_acc': best_overall_dev_acc,
                    'best_explicit_dev_acc': best_explicit_dev_acc,
                    'best_implicit_dev_acc': best_implicit_dev_acc,
                    'best_latent_dev_acc': best_latent_dev_acc,
                    'best_overall_dev_loss': best_dev_loss,
                    'overall_test_acc': overall_test_acc,
                    'explicit_test_acc': explicit_test_acc,
                    'implicit_test_acc': implicit_test_acc,
                    'latent_test_acc': latent_test_acc,
                    'overall_test_loss': test_avg_loss
                    },
                    checkpoint_dir)
            except Exception as e:
                print(e)

        # early stopping
        if patience_counter > args.patience:
            print("Early stopping...")
            break

        print('------------------------------------------------------')
        if new_best == False:
            overall_test_acc = 0
        # print('overall_test_acc', overall_test_acc)
    return best_overall_dev_acc, overall_test_acc


if __name__ == '__main__':
    # run several times and calculate average accuracy and standard deviation
    dev = []
    test = []
    # for i in range(3):    
    #     dev_acc, test_acc = train()
    #     dev.append(dev_acc)
    #     test.append(test_acc)

    dev_acc, test_acc = train()
    dev.append(dev_acc)
    test.append(test_acc)

    dev = np.array(dev)
    test = np.array(test)

    dev_mean = np.mean(dev)
    dev_std = np.std(dev)
    test_mean = np.mean(test)
    test_std = np.std(test)

    print(str(dev_mean) + '(±' + str(dev_std) + ')')
    print(str(test_mean) + '(±' + str(test_std) + ')')
