import numpy as np
import torch
from collections import defaultdict as ddict
criterion = torch.nn.functional.cross_entropy

class Recorder:
    """
    always return test numbers except the last method
    """
    def __init__(self, metric):
        self.metric = metric
        self.train_results = ddict(list)
        self.val_results = ddict(list)
        self.test_results = ddict(list)
        # self.preds_labels = ddict(list)


    def update(self, train_acc, train_auc, train_ap, val_acc, val_auc, val_ap, test_acc, test_auc, test_ap, test_preds, test_labels):
        self.train_results['acc'].append(train_acc)
        self.train_results['auc'].append(train_auc)
        self.train_results['ap'].append(train_ap)

        self.val_results['acc'].append(val_acc)
        self.val_results['auc'].append(val_auc)
        self.val_results['ap'].append(val_ap)

        self.test_results['acc'].append(test_acc)
        self.test_results['auc'].append(test_auc)
        self.test_results['ap'].append(test_ap)
        self.test_results['preds'].append(test_preds.detach())
        self.test_results['labels'].append(test_labels.detach())


    def get_best_val_metric(self, val):
        max_step = self.get_best_test(val=val)[-1]
        best_val_metric = self.val_results[self.metric][max_step]
        return best_val_metric, max_step

    def get_best_test(self, val):
        if val:
            max_step = int(np.argmax(np.array(self.val_results[self.metric])))
        else:
            max_step = int(np.argmax(np.array(self.test_results[self.metric])))
        # best_test_metric = ddict(float)
        best_test = {}
        for _ in ['acc', 'auc', 'ap', 'preds', 'labels']:
            best_test[_] = self.test_results[_][max_step]
        return best_test, max_step

    # return results on train, val, test for each epoch
    def get_latest_metrics(self):
        return self.train_results[self.metric][-1], self.val_results[self.metric][-1], self.test_results[self.metric][-1]


class S_Recorder:
    """
    derive architectures based on val results
    """
    def __init__(self, metric):
        self.metric = metric
        self.train_results = ddict(list)
        self.val_results = ddict(list)

    def update(self, train_acc, train_auc, train_ap, val_acc, val_auc, val_ap):
        self.train_results['acc'].append(train_acc)
        self.train_results['auc'].append(train_auc)
        self.train_results['ap'].append(train_ap)
        self.val_results['acc'].append(val_acc)
        self.val_results['auc'].append(val_auc)
        self.val_results['ap'].append(val_ap)

    def get_best_val_metric(self):
        max_step = int(np.argmax(np.array(self.val_results[self.metric])))
        best_val_metric = ddict(float)
        for metric in ['acc', 'auc', 'ap']:
            best_val_metric[metric] = self.val_results[metric][max_step]
        return best_val_metric, max_step

    # return results on train, val for each epoch
    def get_latest_metrics(self):
        return self.train_results[self.metric][-1], self.val_results[self.metric][-1]


class A_Recorder:
    """
    evaluate the searched GNN architecture on the test set
    """
    def __init__(self, metric):
        self.metric = metric
        self.train_results = ddict(list)
        self.test_results = ddict(list)
        self.preds_labels = ddict(list)


    def update(self, train_acc, train_auc, train_ap, test_acc, test_auc, test_ap, test_preds, test_labels):
        self.train_results['acc'].append(train_acc)
        self.train_results['auc'].append(train_auc)
        self.train_results['ap'].append(train_ap)
        self.test_results['acc'].append(test_acc)
        self.test_results['auc'].append(test_auc)
        self.test_results['ap'].append(test_ap)

        self.test_results['preds'].append(test_preds.detach())
        self.test_results['labels'].append(test_labels.detach())


    def get_best_test(self):
        max_step = int(np.argmax(np.array(self.test_results[self.metric])))
        best_test = {}
        for _ in ['acc', 'auc', 'ap', 'preds', 'labels']:
            best_test[_] = self.test_results[_][max_step]
        return best_test, max_step

    # return results on train, test for each epoch
    def get_latest_metrics(self):
        return self.train_results[self.metric][-1], self.test_results[self.metric][-1]


# import numpy as np
# import torch
# from collections import defaultdict as ddict
# criterion = torch.nn.functional.cross_entropy
#
# class Recorder:
#     """
#     always return test numbers except the last method
#     """
#     def __init__(self, metric):
#         self.metric = metric
#         self.train_results = ddict(list)
#         self.val_results = ddict(list)
#         self.test_results = ddict(list)
#
#     def update(self, train_acc, train_auc, train_ap, val_acc, val_auc, val_ap, test_acc, test_auc, test_ap):
#         self.train_results['acc'].append(train_acc)
#         self.train_results['auc'].append(train_auc)
#         self.train_results['ap'].append(train_ap)
#         self.val_results['acc'].append(val_acc)
#         self.val_results['auc'].append(val_auc)
#         self.val_results['ap'].append(val_ap)
#         self.test_results['acc'].append(test_acc)
#         self.test_results['auc'].append(test_auc)
#         self.test_results['ap'].append(test_ap)
#
#     def get_best_val_metric(self, val):
#         max_step = self.get_best_test_metric(val=val)[-1]
#         best_val_metric = self.val_results[self.metric][max_step]
#         return best_val_metric, max_step
#
#     def get_best_test_metric(self, val):
#         if val:
#             max_step = int(np.argmax(np.array(self.val_results[self.metric])))
#         else:
#             max_step = int(np.argmax(np.array(self.test_results[self.metric])))
#         best_test_metric = ddict(float)
#         for metric in ['acc', 'auc', 'ap']:
#             best_test_metric[metric] = self.test_results[metric][max_step]
#         return best_test_metric, max_step
#
#     # return results on train, val, test for each epoch
#     def get_latest_metrics(self):
#         return self.train_results[self.metric][-1], self.val_results[self.metric][-1], self.test_results[self.metric][-1]
#
#
# class S_Recorder:
#     """
#     derive architectures based on val results
#     """
#     def __init__(self, metric):
#         self.metric = metric
#         self.train_results = ddict(list)
#         self.val_results = ddict(list)
#
#     def update(self, train_acc, train_auc, train_ap, val_acc, val_auc, val_ap):
#         self.train_results['acc'].append(train_acc)
#         self.train_results['auc'].append(train_auc)
#         self.train_results['ap'].append(train_ap)
#         self.val_results['acc'].append(val_acc)
#         self.val_results['auc'].append(val_auc)
#         self.val_results['ap'].append(val_ap)
#
#     def get_best_val_metric(self):
#         max_step = int(np.argmax(np.array(self.val_results[self.metric])))
#         best_val_metric = ddict(float)
#         for metric in ['acc', 'auc', 'ap']:
#             best_val_metric[metric] = self.val_results[metric][max_step]
#         return best_val_metric, max_step
#
#     # return results on train, val for each epoch
#     def get_latest_metrics(self):
#         return self.train_results[self.metric][-1], self.val_results[self.metric][-1]
#
#
# class A_Recorder:
#     """
#     evaluate the searched GNN architecture on the test set
#     """
#     def __init__(self, metric):
#         self.metric = metric
#         self.train_results = ddict(list)
#         self.test_results = ddict(list)
#         self.preds_labels = ddict(list)
#
#     def update(self, train_acc, train_auc, train_ap, test_acc, test_auc, test_ap):
#         # def update(self, train_acc, train_auc, train_ap, test_acc, test_auc, test_ap, test_preds, test_labels):
#         self.train_results['acc'].append(train_acc)
#         self.train_results['auc'].append(train_auc)
#         self.train_results['ap'].append(train_ap)
#         self.test_results['acc'].append(test_acc)
#         self.test_results['auc'].append(test_auc)
#         self.test_results['ap'].append(test_ap)
#         # self.preds_labels['preds'].append(test_preds)
#         # self.preds_labels['labels'].append(test_labels)
#
#     def get_best_test_metric(self):
#         max_step = int(np.argmax(np.array(self.test_results[self.metric])))
#         best_test_metric = ddict(float)
#         for metric in ['acc', 'auc', 'ap']:
#             best_test_metric[metric] = self.test_results[metric][max_step]
#         return best_test_metric, max_step
#
#     # return results on train, test for each epoch
#     def get_latest_metrics(self):
#         return self.train_results[self.metric][-1], self.test_results[self.metric][-1]