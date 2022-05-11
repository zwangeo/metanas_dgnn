import os
import sys
sys.path.append('..')
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import utils
from utils import *
import torch
from torch import nn
import warnings
from collections import defaultdict as ddict
from utils import *
# from train_degnn import eval_model, compute_metric
from recorder import S_Recorder, A_Recorder
from datetime import datetime
warnings.filterwarnings('ignore')
criterion = torch.nn.functional.cross_entropy


def train_metalearner(cI, learner_w_grad, learner_wo_grad, metalearner, meta_optimizer, test_loader, args, logger, device, performance):
    learner_wo_grad.transfer_params(learner_w_grad, cI)
    preds, labels = utils.get_pred(learner_wo_grad, test_loader, device)  # we may try val_loader / test_loader
    loss = criterion(preds, labels)
    # performance['metatrain']['loss'].append(np.round(loss.item(), 4))
    performance['loss'].append(np.round(loss.item(), 4))
    meta_optimizer.zero_grad()
    loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(metalearner.parameters(), max_norm=args.clip)
    meta_optimizer.step()


def train_learner(learner_w_grad, metalearner, dataloaders, args, logger, device, performance, optimizer=None):
    best_learner = utils.get_model(model_name=args.model, layers=args.layers, in_features=args.in_features, out_features=args.out_features, prop_depth=args.prop_depth, args=args, logger=logger).to(device)
    best_learner.load_state_dict(learner_w_grad.state_dict())

    cI = metalearner.metalstm.cI.data
    hs = [None]

    train_loader, val_loader, test_loader = dataloaders
    metric = args.metric
    s_recorder = S_Recorder(metric)
    search_time = []


    for step in range(args.epoch):
        start = datetime.now()
        cI, hs = meta_optimize_model(cI, hs, learner_w_grad, metalearner, train_loader, device, args, optimizer)
        end = datetime.now()
        search_time.append((end-start).seconds)

        # print(learner_w_grad.log_alpha_agg.weight.view(-1))
        train_loss, train_acc, train_auc, train_ap = utils.eval_model(learner_w_grad, train_loader, device)
        val_loss, val_acc, val_auc, val_ap = utils.eval_model(learner_w_grad, val_loader, device)

        s_recorder.update(train_acc, train_auc, train_ap, val_acc, val_auc, val_ap)
        learner_w_grad.update_z_hard(step)
        logger.info('[{}][eps {}][search][epoch {}] ... best val {}: {:.4f} ... train loss: {:.4f}, val loss: {:.4f} ... train {}: {:.4f}, val {}: {:.4f}'.
                    format(args.phase, args.cur_idx, step, metric, s_recorder.get_best_val_metric()[0][metric], train_loss, val_loss,
                           metric, s_recorder.get_latest_metrics()[0],
                           metric, s_recorder.get_latest_metrics()[1]))

        val_results, learner_w_grad.max_step = s_recorder.get_best_val_metric()
        if val_auc >= s_recorder.get_best_val_metric()[0]['auc']:
            # best_learner = copy.deepcopy(learner)
            best_learner.load_state_dict(learner_w_grad.state_dict())
            best_learner.rec_load(learner_w_grad)

        # if (step - learner_w_grad.max_step) > 3*args.kill_cnt:
        if (step - learner_w_grad.max_step) > args.kill_cnt:
            logger.info('Early Stopping for Searching!')
            break

    # val_results, learner_w_grad.max_step = s_recorder.get_best_val_metric()
    logger.info('[{}][eps {}][search] ... max step: {}, acc: {:.4f}, auc: {:.4f}, ap: {:.4f}'.format
                (args.phase, args.cur_idx, learner_w_grad.max_step, val_results['acc'], val_results['auc'], val_results['ap']))
    # return cI
    # print(best_learner.log_alpha_agg.weight)
    # print(best_learner.preprocess.weight)
    # print(learner_w_grad.log_alpha_agg.weight)
    # print(learner_w_grad.preprocess.weight)

    performance['search_max_step'].append(learner_w_grad.max_step)
    performance['search_time'].append(np.sum(search_time[:learner_w_grad.max_step+1]))
    return cI, best_learner


def meta_optimize_model(cI, hs, learner_w_grad, metalearner, train_loader, device, args, optimizer):
    learner_w_grad.train()
    for batch in train_loader:
        batch = batch.to(device)
        label = batch.y
        # get the loss/grad
        # learner_w_grad.copy_flat_params(cI)
        prediction = learner_w_grad(batch)
        loss = criterion(prediction, label, reduction='mean')

        # print(learner_w_grad.log_alpha_agg.weight)
        # print(learner_w_grad.preprocess.weight[0])
        if optimizer:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            # print(learner_w_grad.log_alpha_agg.weight)
            # print(learner_w_grad.preprocess.weight[0])

        else:
            loss.backward(retain_graph=True)
        grad = learner_w_grad.get_flat_grads()
        # preprocess grad & loss and metalearner forward
        grad_prep = utils.preprocess_grad_loss(grad)  # [n_learner_params, 2]
        loss_prep = utils.preprocess_grad_loss(loss.data.unsqueeze(0))  # [1, 2]
        metalearner_input = [loss_prep, grad_prep, grad.unsqueeze(1)]
        cI, h = metalearner(metalearner_input, hs[-1])
        hs.append(h)
        learner_w_grad.copy_flat_params(cI)

        # print(learner_w_grad.log_alpha_agg.weight)
        # print(learner_w_grad.preprocess.weight[0])
        # print('//////////////////')

    return cI, hs
