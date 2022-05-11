import os
import sys
sys.path.append('..')
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from utils import *
import torch
from torch import nn
import warnings
from collections import defaultdict as ddict
from utils import *
from recorder import S_Recorder, A_Recorder
import copy
warnings.filterwarnings('ignore')
criterion = torch.nn.functional.cross_entropy


def train_model(learner, optimizer, dataloaders, args, logger, device, performance):
    train_loader, val_loader, test_loader = dataloaders
    best_learner = search(learner, optimizer, train_loader, val_loader, args, logger, device)
    retrain(learner, optimizer, train_loader, test_loader, args, logger, device, performance)
    logger.info('*************************************************************************************************')
    return best_learner

def search(learner, optimizer, train_loader, val_loader, args, logger, device):
    s_recorder = S_Recorder(args.metric)
    best_learner = None

    for step in range(args.epoch):
        optimize_model(learner, optimizer, train_loader, device, args)
        train_loss, train_acc, train_auc, train_ap = eval_model(learner, train_loader, device)
        val_loss, val_acc, val_auc, val_ap = eval_model(learner, val_loader, device)
        s_recorder.update(train_acc, train_auc, train_ap, val_acc, val_auc, val_ap)
        # learner.update_z_hard() only exists in the search phase, where z (arch) is obtained through evolving beta
        # z should be fixed once conduct learner.derive_arch()
        learner.update_z_hard(step)

        logger.info('[eps {}][search][epoch {}] ... best val {}: {:.4f} ... train loss: {:.4f}, val loss: {:.4f} ... train {}: {:.4f}, val {}: {:.4f}'.
                    format(args.cur_idx, step, args.metric, s_recorder.get_best_val_metric()[0][args.metric], train_loss, val_loss,
                           args.metric, s_recorder.get_latest_metrics()[0],
                           args.metric, s_recorder.get_latest_metrics()[1]))

        if val_auc >= s_recorder.get_best_val_metric()[0]:
            best_learner = copy.deepcopy(learner)

    val_results, learner.max_step = s_recorder.get_best_val_metric()
    logger.info('[eps {}][search] ... max step: {}, acc: {:.4f}, auc: {:.4f}, ap: {:.4f}'.format
                (args.cur_idx, learner.max_step, val_results['acc'], val_results['auc'], val_results['ap']))

    # best_learner.rec_reset()
    return best_learner

def retrain(learner, optimizer, train_loader, test_loader, args, logger, device, performance):
    learner_, optimizer_ = setup_retrain(learner, optimizer, args, logger, device)
    a_recorder = A_Recorder(args.metric)

    for step in range(args.retrain_epoch):
        optimize_model(learner_, optimizer_, train_loader, device, args)
        train_loss, train_acc, train_auc, train_ap = eval_model(learner_, train_loader, device)
        test_loss, test_acc, test_auc, test_ap, test_preds, test_labels = eval_model(learner_, test_loader, device, return_preds=True)
        a_recorder.update(train_acc, train_auc, train_ap, test_acc, test_auc, test_ap, test_preds, test_labels)

        logger.info('[eps {}][{}][epoch {}] ... best test {}: {:.4f} ... train loss: {:.4f}, test loss: {:.4f} ... train {}: {:.4f}, test {}: {:.4f}'.
                    format(args.cur_idx, args.second_stage, step, args.metric, a_recorder.get_best_test()[0][args.metric], train_loss, test_loss,
                           args.metric, a_recorder.get_latest_metrics()[0],
                           args.metric, a_recorder.get_latest_metrics()[1]))
        if (step - a_recorder.get_best_test()[1]) > args.kill_cnt:
            logger.info('Early Stopping!')
            break
    test_results, max_step = a_recorder.get_best_test()
    logger.info('[eps {}][{}] ... max step: {}, acc: {:.4f}, auc: {:.4f}, ap: {:.4f}'.format
                (args.cur_idx, args.second_stage, max_step, test_results['acc'], test_results['auc'], test_results['ap']))
    performance['acc'].append(np.round(test_results['acc'], 4))
    performance['auc'].append(np.round(test_results['auc'], 4))
    performance['ap'].append(np.round(test_results['ap'], 4))
    performance['preds'].append(a_recorder.get_best_test()[0]['preds'])
    performance['labels'].append(a_recorder.get_best_test()[0]['labels'])



