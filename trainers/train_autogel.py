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
# from utils import *
from recorder import S_Recorder, A_Recorder
import copy
from datetime import datetime
warnings.filterwarnings('ignore')
criterion = torch.nn.functional.cross_entropy


def train_model(learner, optimizer, dataloaders, args, logger, device, performance):
    # train_loader, val_loader, test_loader = dataloaders
    best_learner = search(learner, optimizer, dataloaders, args, logger, device, performance)
    # retrain(learner, optimizer, dataloaders, args, logger, device, performance)
    retrain(best_learner, optimizer, dataloaders, args, logger, device, performance)
    logger.info('*************************************************************************************************')
    return best_learner


def search(learner, optimizer, dataloaders, args, logger, device, performance):
    from utils import optimize_model, eval_model, get_model
    best_learner = get_model(model_name=args.model, layers=args.layers, in_features=args.in_features, out_features=args.out_features, prop_depth=args.prop_depth, args=args, logger=logger).to(device)
    best_learner.load_state_dict(learner.state_dict())
    train_loader, val_loader, test_loader = dataloaders
    s_recorder = S_Recorder(args.metric)
    # best_learner = None
    search_time = []

    for step in range(args.epoch):
        start = datetime.now()
        optimize_model(learner, optimizer, train_loader, device, args)
        end = datetime.now()
        search_time.append((end-start).seconds)

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

        val_results, learner.max_step = s_recorder.get_best_val_metric()
        if val_auc >= s_recorder.get_best_val_metric()[0]['auc']:
            # best_learner = copy.deepcopy(learner)
            best_learner.load_state_dict(learner.state_dict())
            best_learner.rec_load(learner)
        # print(best_learner.log_alpha_agg.weight)

        # if (step - learner.max_step) > 3*args.kill_cnt:
        if (step - learner.max_step) > args.kill_cnt:
            logger.info('Early Stopping for Searching!')
            break

    # val_results, learner.max_step = s_recorder.get_best_val_metric()
    logger.info('[eps {}][search] ... max step: {}, acc: {:.4f}, auc: {:.4f}, ap: {:.4f}'.format
                (args.cur_idx, learner.max_step, val_results['acc'], val_results['auc'], val_results['ap']))

    # best_learner.rec_reset()
    # print(best_learner.log_alpha_agg.weight)
    # print(best_learner.log_alpha_agg.weight)
    # print(best_learner.preprocess.weight)
    performance['search_max_step'].append(learner.max_step)
    performance['search_time'].append(np.sum(search_time[:learner.max_step+1]))
    return best_learner

def retrain(learner, optimizer, dataloaders, args, logger, device, performance):
    from utils import setup_retrain, optimize_model, eval_model, get_model
    train_loader, val_loader, test_loader = dataloaders
    learner_, optimizer_ = setup_retrain(learner, optimizer, args, logger, device)
    a_recorder = A_Recorder(args.metric)
    retrain_time = []


    for step in range(args.retrain_epoch):
        start = datetime.now()
        optimize_model(learner_, optimizer_, train_loader, device, args)
        end = datetime.now()
        retrain_time.append((end-start).seconds)

        train_loss, train_acc, train_auc, train_ap = eval_model(learner_, train_loader, device)
        test_loss, test_acc, test_auc, test_ap, test_preds, test_labels = eval_model(learner_, test_loader, device, return_preds=True)
        a_recorder.update(train_acc, train_auc, train_ap, test_acc, test_auc, test_ap, test_preds, test_labels)

        logger.info('[eps {}][{}][epoch {}] ... best test {}: {:.4f} ... train loss: {:.4f}, test loss: {:.4f} ... train {}: {:.4f}, test {}: {:.4f}'.
                    format(args.cur_idx, args.second_stage, step, args.metric, a_recorder.get_best_test()[0][args.metric], train_loss, test_loss,
                           args.metric, a_recorder.get_latest_metrics()[0],
                           args.metric, a_recorder.get_latest_metrics()[1]))
        if (step - a_recorder.get_best_test()[1]) > args.kill_cnt:
            logger.info('Early Stopping for Retraining!')
            break
    test_results, max_step = a_recorder.get_best_test()
    logger.info('[eps {}][{}] ... max step: {}, acc: {:.4f}, auc: {:.4f}, ap: {:.4f}'.format
                (args.cur_idx, args.second_stage, max_step, test_results['acc'], test_results['auc'], test_results['ap']))

    performance['acc'].append(np.round(test_results['acc'], 4))
    performance['auc'].append(np.round(test_results['auc'], 4))
    performance['ap'].append(np.round(test_results['ap'], 4))
    performance['preds'].append(a_recorder.get_best_test()[0]['preds'])
    performance['labels'].append(a_recorder.get_best_test()[0]['labels'])

    performance['retrain_max_step'].append(max_step)
    performance['retrain_time'].append(np.sum(retrain_time[:max_step+1]))




