import os
import sys
sys.path.append('..')
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from utils import *
import torch
import warnings
from collections import defaultdict as ddict
from recorder import Recorder
from datetime import datetime
criterion = torch.nn.functional.cross_entropy
warnings.filterwarnings('ignore')


def train_model(model, optimizer, dataloaders, args, logger, device, performance):
    from utils import optimize_model, eval_model
    train_loader, val_loader, test_loader = dataloaders
    metric = args.metric
    recorder = Recorder(metric)
    retrain_time = []


    for step in range(args.retrain_epoch):
        start = datetime.now()
        optimize_model(model, optimizer, train_loader, device, args)
        end = datetime.now()
        retrain_time.append((end-start).seconds)

        train_loss, train_acc, train_auc, train_ap = eval_model(model, train_loader, device)
        val_loss, val_acc, val_auc, val_ap = eval_model(model, val_loader, device)
        test_loss, test_acc, test_auc, test_ap, test_preds, test_labels = eval_model(model, test_loader, device, return_preds=True)

        recorder.update(train_acc, train_auc, train_ap, val_acc, val_auc, val_ap, test_acc, test_auc, test_ap, test_preds, test_labels)
        logger.info('[eps {}][epoch {}] ... (with val) best test {}: {:.4f} ... train loss: {:.4f}, val loss: {:.4f} ... train {}: {:.4f} val {}: {:.4f} test {}: {:.4f}'.format
                    (args.cur_idx, step, metric, recorder.get_best_test(val=True)[0][metric], train_loss, val_loss,
                     metric, recorder.get_latest_metrics()[0],
                     metric, recorder.get_latest_metrics()[1],
                     metric, recorder.get_latest_metrics()[2]))
        if (step - recorder.get_best_test(val=True)[1]) > args.kill_cnt:
            logger.info('Early Stopping!')
            break
    logger.info('(with val) final test %s: %.4f (epoch: %d, val %s: %.4f)' %
                (metric, recorder.get_best_test(val=True)[0][metric],
                 recorder.get_best_test(val=True)[-1], metric, recorder.get_best_val_metric(val=True)[0]))

    _, max_step = recorder.get_best_test(val=True)

    performance['acc'].append(np.round(recorder.get_best_test(val=True)[0]['acc'], 4))
    performance['auc'].append(np.round(recorder.get_best_test(val=True)[0]['auc'], 4))
    performance['ap'].append(np.round(recorder.get_best_test(val=True)[0]['ap'], 4))
    performance['preds'].append(recorder.get_best_test(val=True)[0]['preds'])
    performance['labels'].append(recorder.get_best_test(val=True)[0]['labels'])

    performance['retrain_max_step'].append(max_step)
    performance['retrain_time'].append(np.sum(retrain_time[:max_step+1]))
    logger.info('*************************************************************************************************')




