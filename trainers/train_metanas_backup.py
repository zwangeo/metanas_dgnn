import os
import sys
sys.path.append('..')
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from utils import *
import torch
from torch import nn
import warnings
from collections import defaultdict as ddict
from utils import *
# from train_degnn import eval_model, compute_metric
from recorder import S_Recorder, A_Recorder
warnings.filterwarnings('ignore')
criterion = torch.nn.functional.cross_entropy


def train_metalearner(cI, learner_w_grad, learner_wo_grad, metalearner, meta_optimizer, test_loader, args, logger, device, performance):
    learner_wo_grad.transfer_params(learner_w_grad, cI)
    preds, labels = get_pred(learner_wo_grad, test_loader, device)  # we may try val_loader / test_loader
    loss = criterion(preds, labels)
    # performance['metatrain']['loss'].append(np.round(loss.item(), 4))
    performance['loss'].append(np.round(loss.item(), 4))
    meta_optimizer.zero_grad()
    loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(metalearner.parameters(), max_norm=args.clip)
    meta_optimizer.step()


def train_learner(learner_w_grad, metalearner, dataloaders, args, logger, device, optimizer=None):
    train_loader, val_loader, test_loader = dataloaders
    metric = args.metric
    s_recorder = S_Recorder(metric)

    cI = metalearner.metalstm.cI.data
    # if args.model == 'Meta-NAS':
    # if args.learner_init == 'meta':
    #     learner_w_grad.copy_flat_params(cI)
    hs = [None]

    for step in range(args.epoch):
        cI, hs = meta_optimize_model(cI, hs, learner_w_grad, metalearner, train_loader, device, args, optimizer)
        # print(learner_w_grad.log_alpha_agg.weight.view(-1))
        train_loss, train_acc, train_auc, train_ap = eval_model(learner_w_grad, train_loader, device)
        val_loss, val_acc, val_auc, val_ap = eval_model(learner_w_grad, val_loader, device)

        s_recorder.update(train_acc, train_auc, train_ap, val_acc, val_auc, val_ap)
        learner_w_grad.update_z_hard(step)
        logger.info('[{}][eps {}][search][epoch {}] ... best val {}: {:.4f} ... train loss: {:.4f}, val loss: {:.4f} ... train {}: {:.4f}, val {}: {:.4f}'.
                    format(args.phase, args.cur_idx, step, metric, s_recorder.get_best_val_metric()[0][metric], train_loss, val_loss,
                           metric, s_recorder.get_latest_metrics()[0],
                           metric, s_recorder.get_latest_metrics()[1]))

    val_results, learner_w_grad.max_step = s_recorder.get_best_val_metric()
    logger.info('[{}][eps {}][search] ... max step: {}, acc: {:.4f}, auc: {:.4f}, ap: {:.4f}'.format
                (args.phase, args.cur_idx, learner_w_grad.max_step, val_results['acc'], val_results['auc'], val_results['ap']))
    # return cI
    return cI, learner_w_grad


def meta_optimize_model(cI, hs, learner_w_grad, metalearner, train_loader, device, args, optimizer):
    learner_w_grad.train()
    for batch in train_loader:
        batch = batch.to(device)
        label = batch.y
        # get the loss/grad
        # learner_w_grad.copy_flat_params(cI)
        prediction = learner_w_grad(batch)
        loss = criterion(prediction, label, reduction='mean')
        if optimizer:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        else:
            loss.backward(retain_graph=True)
        grad = learner_w_grad.get_flat_grads()
        # preprocess grad & loss and metalearner forward
        grad_prep = preprocess_grad_loss(grad)  # [n_learner_params, 2]
        loss_prep = preprocess_grad_loss(loss.data.unsqueeze(0))  # [1, 2]
        metalearner_input = [loss_prep, grad_prep, grad.unsqueeze(1)]
        cI, h = metalearner(metalearner_input, hs[-1])
        hs.append(h)
        learner_w_grad.copy_flat_params(cI)
    return cI, hs
