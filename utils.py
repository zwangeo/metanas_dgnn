import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import random
import os
import multiprocessing as mp
from tqdm import tqdm
import time
import sys
from copy import deepcopy
import torch
from torch_geometric.data import DataLoader, Data
import torch_geometric.utils as tgu
# from debug import *
from datetime import datetime, timedelta
from itertools import combinations
from metalearner import MetaLearner
from collections import defaultdict as ddict
from collections import Counter
from learners.degnn import *
from learners.auto import *
import copy
from torch.optim.lr_scheduler import StepLR
import warnings
warnings.filterwarnings('ignore')
criterion = torch.nn.functional.cross_entropy
# attributed_nodes_all
from run_scripts import *


def read_file(args, logger):
    dataset = args.dataset
    interval = args.interval
    di_flag = args.directed
    if dataset in ['elliptic']:
        task = 'node_classification'
    elif dataset in ['uci', 'enron', 'sbm', 'as', 'bc_otc', 'bc_alpha', 'hep_th', 'movielens_10m', 'yelp']:
        task = 'link_prediction'
        if dataset in ['movielens_10m', 'yelp']:
            args.bipartite = True
    else:
        raise ValueError('dataset not found')
    directory = './data/' + task + '/' + dataset + '/'
    edges, start, end = read_edges(dataset, directory, args)
    # here the edges are remapped by calling node_remap()
    edges, node_id_mapping, num_nodes, num_edges = node_remap(edges, args)

    aggregated_edges = aggregate_edges(dataset, edges, interval, start, end)
    logger.info('Read in {} for {} || start: {} | end: {} | interval: {} | total # nodes: {} | total # edges (number of tuples): {} | directed: {}'.format(
        dataset, task, start, end, args.interval, num_nodes, num_edges, di_flag))

    # print(len(aggregated_edges))
    # print([len(_) for _ in aggregated_edges])
    dynG_accumulated = []
    dynG_next = []
    edges_accumulated = []
    for idx, edges in enumerate(aggregated_edges):
        if idx < len(aggregated_edges) - 1:
            edges = list(map(lambda x: x[:2], edges))
            edges_next = list(map(lambda x: x[:2], aggregated_edges[idx+1]))
            edges_accumulated.extend(edges)
            # G_accumulated = dynG_helper(num_nodes, edges_accumulated, args)
            # G_next = dynG_helper(num_nodes, edges_next, args)
            # G_accumulated = dynG_helper(attributed_nodes_all, edges_accumulated, args)
            # G_next = dynG_helper(attributed_nodes_all, edges_next, args)
            G_accumulated = dynG_helper(edges_accumulated, args)
            G_next = dynG_helper(edges_next, args)

            dynG_accumulated.append(G_accumulated)
            dynG_next.append(G_next)

            new_edges_next = len(set(G_next.edges) - set(G_accumulated.edges)) if args.target == 'new_lp' else len(set(G_next.edges))
            logger.info('Snapshot index: {} | # nodes with positive degree: {} | # edges_accumulated (so far appeared or not): {} | # new_edges_next : {}'.format(
                idx,
                np.count_nonzero(get_degrees(G_accumulated)),
                G_accumulated.number_of_edges(),
                new_edges_next
            ))
        else:
            edges = list(map(lambda x: x[:2], edges))
            edges_accumulated.extend(edges)

    # args.n_eps = len(dynG_accumulated)
    args.end_idx = len(dynG_accumulated) if args.end_idx < 0 else args.end_idx

    dynG_accumulated = dynG_accumulated[args.start_idx : args.end_idx]
    dynG_next = dynG_next[args.start_idx : args.end_idx]
    logger.info('Snapshots selected: from {} to {}'.format(args.start_idx, args.end_idx))
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    return attributed_nodes_all, dynG_accumulated, dynG_next, task


def optimize_model(model, optimizer, dataloader, device, args):
    model.train()
    for batch in dataloader:
        batch = batch.to(device)
        label = batch.y
        prediction = model(batch)
        loss = criterion(prediction, label, reduction='mean')
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
        optimizer.step()

def eval_model(model, dataloader, device, return_preds=False):
    model.eval()
    with torch.no_grad():
        predictions, labels = get_pred(model, dataloader, device)
    loss, acc, auc, ap = compute_metric(predictions, labels)
    if not return_preds:
        return loss, acc, auc, ap
    else:
        return loss, acc, auc, ap, predictions, labels

def get_pred(model, dataloader, device):
    predictions = []
    labels = []
    for batch in dataloader:
        batch = batch.to(device)
        prediction = model(batch)
        labels.append(batch.y)
        predictions.append(prediction)
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    return predictions, labels

def compute_metric(predictions, labels):
    with torch.no_grad():
        loss = criterion(predictions, labels, reduction='mean').item()
        acc, auc, ap = metric_helper(predictions, labels)
    return loss, acc, auc, ap


def metric_helper(predictions, labels):
    predictions_onehot = torch.argmax(predictions, dim=1)
    correct_predictions = (predictions_onehot == labels)
    acc = correct_predictions.sum().cpu().item() / labels.shape[0]

    predictions = torch.nn.functional.softmax(predictions, dim=-1)
    multi_class = 'ovr'
    if predictions.size(1) == 2:
        predictions = predictions[:, 1]
        multi_class = 'raise'
    auc = roc_auc_score(labels.cpu().numpy(), predictions.cpu().numpy(), multi_class=multi_class)

    target_names = ['negative', 'positive']
    result_dict = classification_report(labels.cpu().numpy(), predictions_onehot.cpu().numpy(), target_names=target_names, output_dict=True, zero_division=1)
    ap = result_dict['macro avg']['precision']
    return acc, auc, ap


def summarize_results(perf_ddict, args, logger):
    if args.dataset == 'enron' and args.eps_usage == 'all':
        for key in perf_ddict:
            perf_ddict[key] = perf_ddict[key][4:]
    n_eps = len(list(perf_ddict.values())[0])
    assert n_eps > 0
    logger.info('Summary >>> task: {}, model: {}, dataset: {}, num_episode: {}, epoch: {}'.format(args.target, args.model, args.dataset, n_eps, args.epoch))

    # macro results
    macro_auc = np.round(sum(perf_ddict['auc'])/n_eps, 4)
    macro_acc = np.round(sum(perf_ddict['acc'])/n_eps, 4)
    macro_ap = np.round(sum(perf_ddict['ap'])/n_eps, 4)
    logger.info('macro auc: {} ### auc for each eps: {}'.format(macro_auc, perf_ddict['auc']))
    logger.info('macro acc: {} ### acc for each eps: {}'.format(macro_acc, perf_ddict['acc']))
    logger.info('macro ap: {} ### ap for each eps: {}'.format(macro_ap, perf_ddict['ap']))

    # micro results
    preds_all = torch.cat(perf_ddict['preds'], dim=0)
    labels_all = torch.cat(perf_ddict['labels'], dim=0)
    micro_acc, micro_auc, micro_ap = metric_helper(preds_all, labels_all)

    micro_auc = np.round(micro_auc, 4)
    micro_acc = np.round(micro_acc, 4)
    micro_ap = np.round(micro_ap, 4)
    logger.info('micro auc: {}'.format(micro_auc))
    logger.info('micro acc: {}'.format(micro_acc))
    logger.info('micro ap: {}'.format(micro_ap))

    # total search and retrain time
    search_time = np.sum(perf_ddict['search_time'])
    retrain_time = np.sum(perf_ddict['retrain_time'])

    logger.info('seartch time: {} ### time for each eps: {}'.format(search_time, perf_ddict['search_time']))
    logger.info('retrain time: {} ### time for each eps: {}'.format(retrain_time, perf_ddict['retrain_time']))
    logger.info('search_max_step: {}'.format(perf_ddict['search_max_step']))
    logger.info('retrain_max_step: {}'.format(perf_ddict['retrain_max_step']))

    # logger.info('seartch time for each eps: {}'.format(perf_ddict['search_time']))
    # logger.info('retrain time for each eps: {}'.format(perf_ddict['retrain_time']))
    # logger.info('total search time: {} seconds'.format(search_time))
    # logger.info('total retrain time: {} seconds'.format(retrain_time))

    return macro_auc, macro_acc, macro_ap, micro_auc, micro_acc, micro_ap, search_time, retrain_time



def check(args):
    # if args.dataset == 'foodweb' and not args.directed:
    #     raise Warning('dataset foodweb is essentially a directed network but currently treated as undirected')
    # if args.dataset == 'simulation':
    #     if args.n is None:
    #         args.n = [10, 20, 40, 80, 160, 320, 640, 1280]
    #     if args.max_sp < args.T:
    #         raise Warning('maximum shortest path distance (max_sp) is less than max number of layers (T), which may deteriorate model capability')

    # get_model_name(args)

    if args.delta_version == 1:
        if args.dataset in ['bc_alpha', 'bc_otc']:
            args.start_idx, args.end_idx = 10, 49
            args.interval = 28
        if args.dataset == 'uci':
            args.start_idx, args.end_idx = 4, 53
            args.interval = 1
            args.lr = 1e-4
        if args.dataset == 'movielens_10m':
            args.start_idx, args.end_idx = 0, -1
            args.interval = 28
            args.num_hops = 1
            args.train_ratio = 0.05
            args.bipartite = True

    if args.delta_version == 2:
        args.save = args.save + '_{}'.format(args.delta_version)

        if args.dataset in ['bc_alpha', 'bc_otc']:
            args.start_idx, args.end_idx = 5, 24
            args.interval = 56
        if args.dataset == 'uci':
            args.start_idx, args.end_idx = 2, 26
            args.interval = 2
            args.lr = 1e-4
        if args.dataset == 'movielens_10m':
            args.start_idx, args.end_idx = 0, -1
            args.interval = 56
            args.num_hops = 1
            args.train_ratio = 0.05
            args.bipartite = True

    if args.delta_version == 3:
        args.save = args.save + '_{}'.format(args.delta_version)

        if args.dataset in ['bc_alpha', 'bc_otc']:
            args.start_idx, args.end_idx = 3, 16
            args.interval = 84
        if args.dataset == 'uci':
            args.start_idx, args.end_idx = 1, 14
            args.interval = 3
            args.lr = 1e-4
        if args.dataset == 'movielens_10m':
            args.start_idx, args.end_idx = 0, -1
            args.interval = 84
            args.num_hops = 1
            args.train_ratio = 0.05
            args.bipartite = True


# def args_helper(args):
    # if args.dataset in ['bc_alpha', 'bc_otc']:
    #     args.start_idx, args.end_idx = 10, 49
    #     args.interval = 28
    # if args.dataset == 'uci':
    #     args.start_idx, args.end_idx = 4, 53
    #     args.interval = 1
    # if args.dataset == 'movielens_10m':
    #     args.start_idx, args.end_idx = 0, -1
    #     args.interval = 28
    #     args.num_hops = 1
    #     args.train_ratio = 0.05
    #     args.bipartite = True

def get_model_name(args):
    if not args.search_based:
        args.model = 'dyn_degnn'
    else:
        args.model = '_'.join([args.learner_init, args.learner_update])


def set_random_seed(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device(args):
    gpu = args.gpu
    return torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')

def get_model(model_name, layers, in_features, out_features, prop_depth, args, logger):
    if model_name == 'dyn_degnn':
        model = DEGNN(layers=layers, in_features=in_features, hidden_features=args.hidden_features, out_features=out_features, prop_depth=prop_depth, args=args, dropout=args.dropout)
    # elif model_name in ['baseline_1', 'baseline_2', 'Meta-NAS', 'Meta-NAS_inc', 'meta_wo_init']:
    # we name the following variants based on how-to-init and how-to-update
    elif model_name in ['rand_sgd', 'inc_sgd', 'meta_sgd', 'rand_meta', 'inc_meta', 'meta_meta']:
        model = AutoGEL(layers=layers, in_features=in_features, hidden_features=args.hidden_features, out_features=out_features, prop_depth=prop_depth, args=args, dropout=args.dropout)
    else:
        raise NotImplementedError
    device = get_device(args)
    logger.info(model.short_summary())
    return model

def get_optimizer(model, args, type):
    # there are totally 4 different types of models in the entire designs
    # 1 type for bl_0, bl_1, and bl_2: manually-crafted GNN model / controller + supernet, sgd each batch
    # 3 types for Meta-NAS:
    #     1. learner_w_grad: controller + supernet
    #           1) how to optimize: controller being optimized with meta and supernet being optimized with sgd
    #           2) optimization frequency: once each batch in the training process of metatrain
    #     2. learner_wo_grad: searched GNN model (childnet)
    #           1) how to optimize: childnet being optimized with sgd
    #           2) optimization frequency: once each batch in the testing process of metatrain
    #     3. metalearner: meta-controller
    #           1) how to optimize: being optimized with sgd
    #           2) optimization frequency: once each eps in metatrain phase

    # if type in ['bls', 'learner_wo_grad', 'metalearner']:
    # if type in ['bls', 'learner_wo_grad', 'metalearner'] or args.meta_target == 'beta_omega':
    #     params = model.parameters()
    # elif type in ['learner_w_grad']:
    #     params = [p for n,p in model.named_parameters() if n not in [_ + '.weight' for _ in model.controller_params]]
    #     # params = [m.weight for n,m in model.named_modules() if n not in model.controller_params]
    # else:
    #     raise NotImplementedError

    lr = args.meta_lr if type == 'metalearner' else args.lr
    # lr = args.lr
    if args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.l2)
    elif args.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=args.l2)
    else:
        raise NotImplementedError


# to save something at the end of metatrain phase (for Meta-NAS model only)
#   1. meta-controller / meta-learner:
#       1) for each eps in the metatest, we always load this trained meta-conteoller at the beginning (different ways with the follwing 2.)
#   2. controller + supernet (we call it the combo as: learner):
#       1) since we follws the incremental settings (during the entire sets of eps, including metatrain and metatest), and
#       2) we always use the well-trained one in the previous to initialize for the current

# def save_ckpt(learner_w_grad, metalearner, optimizer, args):
#     if not os.path.exists(args.save):
#         os.makedirs(args.save)
#     torch.save({'metalearner': metalearner.state_dict(),
#                 'learner_w_grad': learner_w_grad.state_dict(),
#                 'optimizer': optimizer.state_dict()},
#                os.path.join(args.save, 'ckpt.pth.tar'))

def save_ckpt(metalearner, args):
    # args.save = os.path.join(args.save, args.dataset)

    # if args.delta_version != 1:
    #     args.save = args.save + '_{}'.format(args.delta_version)

    args.save = os.path.join(args.save, args.dataset, 'metatrain_'+str(args.metatrain_ratio))
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    torch.save({'metalearner': metalearner.state_dict()},
               os.path.join(args.save, 'epoch_{}.pth.tar'.format(args.epoch)))

def resume_ckpt(args, device):
    # ckpt_path = os.path.join(args.save, args.dataset, 'epoch_{}.pth.tar'.format(args.epoch))
    # if args.delta_version != 1:
    #     args.save = args.save + '_{}'.format(args.delta_version)

    ckpt_path = os.path.join(args.save, args.dataset, 'metatrain_'+str(np.round(args.metatrain_ratio, 1)), 'epoch_{}.pth.tar'.format(args.epoch))

    ckpt = torch.load(ckpt_path, map_location=device)
    metalearner = MetaLearner(args.input_size, args.hidden_size, args.n_learner_params).to(device)
    metalearner.load_state_dict(ckpt['metalearner'])
    return metalearner


# def metatrain_init(args, logger, device):
#     learner_w_grad = get_model(model_name=args.model, layers=args.layers, in_features=args.in_features, out_features=args.out_features, prop_depth=args.prop_depth, args=args, logger=logger).to(device)
#     # learner_wo_grad = get_model(model_name=args.model, layers=args.layers, in_features=args.in_features, out_features=args.out_features, prop_depth=args.prop_depth, args=args, logger=logger).to(device)
#     learner_wo_grad = copy.deepcopy(learner_w_grad)
#     learner_w_grad.update_z_hard()
#     # learner_wo_grad.update_z_hard()
#
#     flat_params = learner_w_grad.get_flat_params()
#     args.n_learner_params = flat_params.size(0)
#     metalearner = MetaLearner(args.input_size, args.hidden_size, args.n_learner_params).to(device)
#     metalearner.metalstm.init_cI(flat_params)
#
#     optimizer = get_optimizer(learner_w_grad, args, type='learner_w_grad')
#     meta_optimizer = get_optimizer(metalearner, args, type='metalearner')
#     return learner_w_grad, learner_wo_grad, metalearner, optimizer, meta_optimizer


def metatrain_init(args, logger, device):
    learner_w_grad = get_model(model_name=args.model, layers=args.layers, in_features=args.in_features, out_features=args.out_features, prop_depth=args.prop_depth, args=args, logger=logger).to(device)
    learner_wo_grad = copy.deepcopy(learner_w_grad)
    learner_w_grad.update_z_hard()

    flat_params = learner_w_grad.get_flat_params()
    args.n_learner_params = flat_params.size(0)
    metalearner = MetaLearner(args.input_size, args.hidden_size, args.n_learner_params).to(device)
    metalearner.metalstm.init_cI(flat_params)

    optimizer = get_optimizer(learner_w_grad, args, type='learner_w_grad')
    meta_optimizer = get_optimizer(metalearner, args, type='metalearner')
    return learner_w_grad, learner_wo_grad, metalearner, optimizer, meta_optimizer
    # return learner_w_grad, metalearner, optimizer, meta_optimizer


def setup_model(args, logger, device):
    learner = get_model(model_name=args.model, layers=args.layers, in_features=args.in_features, out_features=args.out_features, prop_depth=args.prop_depth, args=args, logger=logger).to(device)
    # if args.model != 'baseline_0':
    if args.model != 'dyn_degnn':
        learner.update_z_hard()
    optimizer = get_optimizer(learner, args, type='bls')
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    return learner, optimizer


def setup_retrain(learner, optimizer, args, logger, device):
    # learner.derive_arch()
    # learner_adapt, optimizer_adapt = setup_model(args, logger, device)
    # # learner_adapt = copy.deepcopy(learner)
    # if args.second_stage == 'adapt':
    #     learner_adapt.load_state_dict(learner.state_dict())
    #     optimizer_adapt.load_state_dict(optimizer.state_dict())
    # learner_adapt.rec_load(learner)
    # learner_adapt.derive_arch()
    learner_, optimizer_ = setup_model(args, logger, device)
    learner.derive_arch()
    learner_.rec_load(learner)
    learner_.derive_arch()
    # if args.second_stage == 'adapt':
    #     learner_.load_state_dict(learner.state_dict())
    #     optimizer_.load_state_dict(optimizer.state_dict())

    logger.info('Second stage: {}'.format(args.second_stage))
    logger.info('Searched one-hot vectors:')
    logger.info(learner_.searched_arch_z)
    logger.info('Searched GNN model (child-net):')
    logger.info(learner_.searched_arch_op)
    # return learner_adapt, optimizer_adapt, scheduler_adapt
    return learner_, optimizer_

# def setup_retrain(learner, optimizer, args, logger, device):
#     learner_ = get_model(model_name=args.model, layers=args.layers, in_features=args.in_features, out_features=args.out_features, prop_depth=args.prop_depth, args=args, logger=logger).to(device)
#     learner_.load_state_dict(learner.state_dict())
#     learner_.rec_load(learner)
#     learner_.derive_arch()
#     optimizer_ = get_optimizer(learner_, args, type='bls')
#
#     # if args.second_stage == 'adapt':
#     #     learner_.load_state_dict(learner.state_dict())
#     #     optimizer_.load_state_dict(optimizer.state_dict())
#
#     logger.info('Second stage: {}'.format(args.second_stage))
#     logger.info('Searched one-hot vectors:')
#     logger.info(learner_.searched_arch_z)
#     logger.info('Searched GNN model (child-net):')
#     logger.info(learner_.searched_arch_op)
#     return learner_, optimizer_

# def degnn_eps_init(args, logger, device):
#     learner = get_model(model_name='baseline_0', layers=args.layers, in_features=args.in_features, out_features=args.out_features, prop_depth=args.prop_depth, args=args, logger=logger).to(device)
#     optimizer = get_optimizer(learner, args, type='bls')
#     return learner, optimizer
#
# def nas_eps_init(args, logger, device):
#     learner = get_model(model_name='baseline_1', layers=args.layers, in_features=args.in_features, out_features=args.out_features, prop_depth=args.prop_depth, args=args, logger=logger).to(device)
#     optimizer = get_optimizer(learner, args, type='bls')
#     return learner, optimizer
    # learner_adapt = get_model(model_name='baseline_1', layers=args.layers, in_features=args.in_features, out_features=args.out_features, prop_depth=args.prop_depth, args=args, logger=logger).to(device)
    # optimizer_adapt = get_optimizer(learner, args, type='bls')
    # learner_adapt.load_state_dict(learner.state_dict())
    # optimizer_adapt.load_state_dict(optimizer.state_dict())
    # return learner, learner_adapt, optimizer, optimizer_adapt


# changed: if inherit from the previous, we directly use one learner instead (i.e., inherit the entire supernet);
# so don't need this func learner_init() anymore
# 'inherit' shall happen on the following scenarios:
    # 1) baseline_2: entire process (use only eps metatest split)
    # 2) meta-nas: metatraining phase
    # 3) meta-nas: metatesting phase <previously this step is missing!!!!!!!!!!!!>


def preprocess_grad_loss(x):
    p = 10
    indicator = (x.abs() >= np.exp(-p)).to(torch.float32)
    # preproc1
    x_proc1 = indicator * torch.log(x.abs() + 1e-8) / p + (1 - indicator) * -1
    # preproc2
    x_proc2 = indicator * torch.sign(x) + (1 - indicator) * np.exp(p) * x
    return torch.stack((x_proc1, x_proc2), 1)

def estimate_storage(dataloaders, names, logger):
    total_gb = 0
    for dataloader, name in zip(dataloaders, names):
        dataset = dataloader.dataset
        storage = 0
        total_length = len(dataset)
        sample_size = 100
        for i in np.random.choice(total_length, sample_size):
            storage += (sys.getsizeof(dataset[i].x.storage()) + sys.getsizeof(dataset[i].edge_index.storage()) +
                        sys.getsizeof(dataset[i].y.storage())) + sys.getsizeof(dataset[i].set_indices.storage())
        gb = storage*total_length/sample_size/1e9
        total_gb += gb
    logger.info('Data roughly takes {:.4f} GB in total'.format(total_gb))
    return total_gb


def node_remap(edges, args):
    nodes = ddict(list)
    global remapped_nodes
    remapped_nodes = ddict(list)
    edge_cnt = 0
    remapped_edges = []
    for idx,_ in enumerate(edges):
        u, v = _[:2]
        # if args.dataset in ['movielens_10m', 'yelp']:
        #     args.bipartite = True
        if args.bipartite:
            u, v = 'b0_' + u, 'b1_' + v
            nodes['b0'].append(u)
            nodes['b1'].append(v)
            nodes['all'].extend([u, v])
        else:
            u, v = 'b0_' + u, 'b0_' + v
            nodes['b0'].append(u)
            nodes['all'].extend([u, v])
        edges[idx][: 2] = [u, v]
        edge_cnt += 1
    for key in nodes:
        nodes[key] = sorted(list(set(nodes[key])))
    if args.bipartite:
        assert len(nodes['b0']) > 0 and len(nodes['b1']) > 0

    nodes_all = nodes['all']
    # print(nodes_all)
    num_nodes = len(nodes_all)
    num_edges = edge_cnt
    node_id_mapping = {old_id: new_id for new_id, old_id in enumerate(nodes_all)}
    for key in nodes:
        remapped_nodes[key] = [node_id_mapping[old_id] for old_id in nodes[key]]

    for e in edges:
        u = node_id_mapping[e[0]]
        v = node_id_mapping[e[1]]
        remapped_edges.append([u, v] + e[2:])

    node_bipartite_mapping = ddict(int)
    global attributed_nodes_all
    attributed_nodes_all = []
    for new_id, old_id in enumerate(nodes_all):
        b = 0 if 'b0' in old_id else 1
        n = (new_id, {'bipartite': b})
        node_bipartite_mapping[new_id] = b
        attributed_nodes_all.append(n)
    if args.bipartite:
        bipartite_flag = bipartite_check(remapped_edges, node_bipartite_mapping)
        assert bipartite_flag
    return remapped_edges, node_id_mapping, num_nodes, num_edges


def bipartite_check(remapped_edges, node_bipartite_mapping):
    l = []
    for e in remapped_edges:
        u, v = e[: 2]
        l.append(node_bipartite_mapping[u] + node_bipartite_mapping[v])
    c = Counter(l)
    if len(c) == 1 and list(c.keys()) == [1]:
        return True
    else:
        return False


def read_edges(dataset, dir, args):
    edges = []
    # edges_pruned = []
    ts = []
    fin_edges = open(dir + 'edges.txt')
    for line in fin_edges.readlines():
        line = line.strip()
        if len(line.split()) != 0:
            node1, node2, weight, time = line.split(',')
            time = int(float(time))
            # node1, node2, weight, time = map(int, map(float, line.split(',')))
            if dataset == 'sbm':
                edges.append([node1, node2, time])
                ts.append(time)
            else:
                edges.append([node1, node2, datetime.fromtimestamp(time)])
                ts.append(datetime.fromtimestamp(time))
    fin_edges.close()

    if dataset == 'sbm':
        start_date = 0
        end_date = int(edges[-1][-1])
    else:
        start_date = min(ts) + timedelta(args.del_days_head)
        end_date = max(ts) - timedelta(args.del_days_tail)

    # for _ in edges:
    #     # if start_date <= _[-1] <= end_date:
    #     # edges_pruned.append(_)
    edges.sort(key=lambda x: x[-1])
    return edges, start_date, end_date


def aggregate_edges(dataset, edges, interval, start, end):
    # aggregated_edges = []
    aggregated_edges = ddict(list)
    current = []
    #### to modify later########################
    if dataset == 'sbm':
        for i, edge in enumerate(edges):
            idx = edge[-1]
            aggregated_edges[idx].append(edge)
        # idx = edges[0][-1]
        # for i, edge in enumerate(edges):
        #     if edge[-1] == idx:
        #         current.append(edge)
        #     else:
        #         aggregated_edges.append(current)
        #         idx = edge[-1]
        #         current = []
        #         current.append(edge)
        # aggregated_edges.append(current)
    #### to modify later########################
    else:
        for i, edge in enumerate(edges):
            if edge[-1] < start:
                edge[-1] = start
            if edge[-1] > end:
                edge[-1] = end
            idx = (edge[-1] - start).days//interval
            aggregated_edges[idx].append(edge)
    aggregated_edges = list(aggregated_edges.values())
    return aggregated_edges


# def dynG_helper(num_nodes, edges, args):
#     G = nx.Graph()
#     if isinstance(num_nodes, int):
#         # G.add_nodes_from([i for i in range(num_nodes)])
#         G.add_nodes_from([i for i in range(num_nodes)], bipartite=0)
#     elif isinstance(num_nodes, tuple):
#         G.add_nodes_from([i for i in range(num_nodes[0])], bipartite=0)
#         G.add_nodes_from([i for i in range(num_nodes[0], num_nodes[0]+num_nodes[1])], bipartite=1)
#     else:
#         raise NotImplementedError
#     # print(num_nodes)
#     # print(G.number_of_nodes())
#     G.add_edges_from(edges)
#     # print(G.number_of_nodes())
#     attributes = np.zeros((G.number_of_nodes(), 1), dtype=np.float32)
#     if args.use_degree:
#         attributes += np.expand_dims(np.log(get_degrees(G) + 1), 1).astype(np.float32)
#     if args.use_attributes:
#         # TODO: read in attribute file to concat to axis -1 of attributes, raise error if not found
#         raise NotImplementedError
#     G.graph['attributes'] = attributes
#     return G

# def dynG_helper(attributed_nodes_all, edges, args):
def dynG_helper(edges, args):
    G = nx.Graph()
    G.add_nodes_from(attributed_nodes_all)
    # print(len(attributed_nodes_all))
    # print(G.number_of_nodes())
    G.add_edges_from(edges)
    # print(G.number_of_nodes())
    attributes = np.zeros((G.number_of_nodes(), 1), dtype=np.float32)
    if args.use_degree:
        attributes += np.expand_dims(np.log(get_degrees(G) + 1), 1).astype(np.float32)
    if args.use_attributes:
        # TODO: read in attribute file to concat to axis -1 of attributes, raise error if not found
        raise NotImplementedError
    G.graph['attributes'] = attributes
    return G


# def read_file(args, logger):
#     dataset = args.dataset
#     interval = args.interval
#     di_flag = args.directed
#     if dataset in ['elliptic']:
#         task = 'node_classification'
#     elif dataset in ['uci', 'enron', 'sbm', 'as', 'bc_otc', 'bc_alpha', 'hep_th', 'movielens_10m', 'yelp']:
#         task = 'link_prediction'
#         if dataset in ['movielens_10m', 'yelp']:
#             args.bipartite = True
#     else:
#         raise ValueError('dataset not found')
#     directory = './data/' + task + '/' + dataset + '/'
#     edges, start, end = read_edges(dataset, directory, args)
#     # here the edges are remapped by calling node_remap()
#     edges, node_id_mapping, num_nodes, num_edges = node_remap(edges, args)
#
#     aggregated_edges = aggregate_edges(dataset, edges, interval, start, end)
#     logger.info('Read in {} for {} || start: {} | end: {} | interval: {} | total # nodes: {} | total # edges (number of tuples): {} | directed: {}'.format(
#         dataset, task, start, end, args.interval, num_nodes, num_edges, di_flag))
#
#     # print(len(aggregated_edges))
#     # print([len(_) for _ in aggregated_edges])
#     dynG_accumulated = []
#     dynG_next = []
#     edges_accumulated = []
#     for idx, edges in enumerate(aggregated_edges):
#         if idx < len(aggregated_edges) - 1:
#             edges = list(map(lambda x: x[:2], edges))
#             edges_next = list(map(lambda x: x[:2], aggregated_edges[idx+1]))
#             edges_accumulated.extend(edges)
#             # G_accumulated = dynG_helper(num_nodes, edges_accumulated, args)
#             # G_next = dynG_helper(num_nodes, edges_next, args)
#             # G_accumulated = dynG_helper(attributed_nodes_all, edges_accumulated, args)
#             # G_next = dynG_helper(attributed_nodes_all, edges_next, args)
#             G_accumulated = dynG_helper(edges_accumulated, args)
#             G_next = dynG_helper(edges_next, args)
#
#             dynG_accumulated.append(G_accumulated)
#             dynG_next.append(G_next)
#
#             new_edges_next = len(set(G_next.edges) - set(G_accumulated.edges)) if args.target == 'new_lp' else len(set(G_next.edges))
#             logger.info('Snapshot index: {} | # nodes with positive degree: {} | # edges_accumulated (so far appeared or not): {} | # new_edges_next : {}'.format(
#                 idx,
#                 np.count_nonzero(get_degrees(G_accumulated)),
#                 G_accumulated.number_of_edges(),
#                 new_edges_next
#             ))
#         else:
#             edges = list(map(lambda x: x[:2], edges))
#             edges_accumulated.extend(edges)
#
#     # args.n_eps = len(dynG_accumulated)
#     args.end_idx = len(dynG_accumulated) if args.end_idx < 0 else args.end_idx
#
#     dynG_accumulated = dynG_accumulated[args.start_idx : args.end_idx]
#     dynG_next = dynG_next[args.start_idx : args.end_idx]
#     logger.info('Snapshots selected: from {} to {}'.format(args.start_idx, args.end_idx))
#     logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
#     logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
#     return attributed_nodes_all, dynG_accumulated, dynG_next, task


def meta_split(dynG_accumulated, dynG_next, args):
    l1 = [_.number_of_nodes() for _ in dynG_accumulated]
    l2 = [_.number_of_nodes() for _ in dynG_next]

    num_eps_all = len(dynG_accumulated)
    num_eps_metatrain = int((1-args.metatest_ratio)*num_eps_all)
    # if args.model in ['baseline_0', 'baseline_1', 'baseline_2'] and args.eps_usage == 'partial':
    #     assert args.cur_idx == num_eps_metatrain
    # else:
    #     assert args.cur_idx == 0
    num_eps_metatest = num_eps_all - num_eps_metatrain

    dynG_accumulated_meta = ddict(list)
    dynG_next_meta = ddict(list)
    dynG_accumulated_meta['metatrain'] = dynG_accumulated[:num_eps_metatrain]
    dynG_accumulated_meta['metatest'] = dynG_accumulated[num_eps_metatrain:]
    dynG_next_meta['metatrain'] = dynG_next[:num_eps_metatrain]
    dynG_next_meta['metatest'] = dynG_next[num_eps_metatrain:]
    dynG_pair_metatrain = [[dynG_accumulated_meta['metatrain'][idx], dynG_next_meta['metatrain'][idx]] for idx in range(num_eps_metatrain)]
    dynG_pair_metatest = [[dynG_accumulated_meta['metatest'][idx], dynG_next_meta['metatest'][idx]] for idx in range(num_eps_metatest)]

    return dynG_pair_metatrain, dynG_pair_metatest


def get_dimension(dynG_accumulated, dynG_next, task, args, logger):
    # G_pair = [dynG_accumulated[0], dynG_next[0]]
    try:
        G_pair = [dynG_accumulated[args.cur_idx - args.start_idx], dynG_next[args.cur_idx - args.start_idx]]
        dataloaders = get_data(G_pair, task=task, args=args, logger=logger, info=False)
    except:
        # print(len(dynG_accumulated))
        # print(args.start_idx)
        # print(args.end_idx)
        G_pair = [dynG_accumulated[int((args.end_idx - args.start_idx)/2)], dynG_next[int((args.end_idx - args.start_idx)/2)]]
        dataloaders = get_data(G_pair, task=task, args=args, logger=logger, info=False)
    # print(dataloaders[0])
    # print(dataloaders[0].dataset[0])
    args.in_features = dataloaders[0].dataset[0].x.shape[-1]


def get_data(G_pair, task, args, logger, info=True):
    G_pair = list(map(lambda x: deepcopy(x), G_pair))
    sp_flag = 'sp' in args.feature
    rw_flag = 'rw' in args.feature
    # norm_flag = args.adj_norm
    feature_flags = (sp_flag, rw_flag)

    train_set, val_set, test_set = get_datalist(G_pair, task, feature_flags, args, logger)
    train_loader, val_loader, test_loader = load_datasets(train_set, val_set, test_set, bs=args.bs)
    if info:
        logger.info('Train size :{}, val size: {}, test size: {}'.format(len(train_set), len(val_set), len(test_set)))
    return (train_loader, val_loader, test_loader)


def get_datalist(G_pair, task, feature_flags, args, logger, info=True):
    # --> the num_hops for movielens_10m dataset: we try {1, 2} (the setting in IGMC is 1);
    # default num_hops = 2 for the rest datasets
    # if args.dataset == 'movielen_10m':

    # if args.dataset in ['movielens_10m', 'sbm']:
    #     dir = 'interval_{}_ratio_{}_hop_{}'.format(args.interval, args.train_ratio, args.num_hops)
    # else:
    #     dir = 'interval_{}_ratio_{}'.format(args.interval, args.train_ratio)
    # dir = 'interval_{}_ratio_{}_hop_{}'.format(args.interval, args.train_ratio, args.layers)
    dir = 'interval_{}_ratio_{}_hop_{}'.format(args.interval, args.train_ratio, args.num_hops)
    # load_dir = os.path.join(args.preprocessed_data, args.dataset, dir)
    load_dir = os.path.join(args.preprocessed_data, args.target, args.dataset, dir)

    if not os.path.exists(load_dir):
        os.makedirs(load_dir)
    f = 'eps_{}.npz'.format(args.cur_idx)
    file_path = os.path.join(load_dir, f)
    logger.info('Snapshot idx :{}, train_ratio: {}, val ratio: {}, test ratio: {}'.format(args.cur_idx, args.train_ratio, args.val_ratio, args.test_ratio))
    # if args.dataset == 'enron':
    #     f = np.load(file_path, allow_pickle=True)
    #     data_list = f['data_list']
    #     train_mask = f['train_mask']
    #     val_test_mask = f['val_test_mask']
    #     n_instances = len(data_list)
    #     # logger.info('Successfully load {} train+val+test instances in total.'.format(n_instances))
    #     logger.info('Successfully load {} train+val+test instances in total from {}.'.format(n_instances, load_dir))
    #
    #     # G_pair, labels, set_indices, (train_mask, val_test_mask) = generate_set_indices_labels(G_pair, task, logger, args)
    #     # data_list = extract_subgaphs(G_pair[0], labels, set_indices, prop_depth=args.prop_depth, layers=args.layers, feature_flags=feature_flags, task=task, max_sprw=(args.max_sp, args.rw_depth), parallel=args.parallel, logger=logger, args=args, debug=args.debug)
    #     # # np.savez(file_path, data_list=data_list, train_mask=train_mask, val_test_mask=val_test_mask)
    #     # np.savez(os.path.join(args.preprocessed_data, args.dataset, dir, f), data_list=data_list, train_mask=train_mask, val_test_mask=val_test_mask)
    # else:
    try:
        f = np.load(file_path, allow_pickle=True)
        data_list = f['data_list']
        train_mask = f['train_mask']
        val_test_mask = f['val_test_mask']
        n_instances = len(data_list)
        # logger.info('Successfully load {} train+val+test instances in total.'.format(n_instances))
        logger.info('Successfully load {} train+val+test instances in total from {}.'.format(n_instances, load_dir))

    except:
        G_pair, labels, set_indices, (train_mask, val_test_mask) = generate_set_indices_labels(G_pair, task, logger, args)
        data_list = extract_subgaphs(G_pair[0], labels, set_indices, prop_depth=args.prop_depth, layers=args.layers, feature_flags=feature_flags, task=task, max_sprw=(args.max_sp, args.rw_depth), parallel=args.parallel, logger=logger, args=args, debug=args.debug)
        np.savez(file_path, data_list=data_list, train_mask=train_mask, val_test_mask=val_test_mask)
        # np.savez(os.path.join(args.preprocessed_data, args.dataset, dir, f), data_list=data_list, train_mask=train_mask, val_test_mask=val_test_mask)

    data_list = list(map(lambda d: Data(x=d['x'], edge_index=d['edge_index'], y=d['y'], set_indices=d['set_indices']), data_list))
    # print(data_list[0])
    train_set, val_set, test_set = split_datalist(data_list, (train_mask, val_test_mask), args)
    return train_set, val_set, test_set


def extract_subgaphs(G_accumulated, labels, set_indices, prop_depth, layers, feature_flags, task, max_sprw, parallel, logger, args, debug=False):
    # deal with adj and features
    logger.info('Encode positions ... (Parallel: {})'.format(parallel))
    data_list = []
    hop_num = get_hop_num(prop_depth, layers, max_sprw, feature_flags)
    n_samples = set_indices.shape[0]
    if not parallel:
        for sample_i in tqdm(range(n_samples)):
            data = get_data_sample(G_accumulated, set_indices[sample_i], hop_num, feature_flags, max_sprw,
                                   label=labels[sample_i] if labels is not None else None, args=args, debug=debug)
            data_list.append(data)
            # return
    else:
        pass
    return data_list


def get_data_sample(G_accumulated, set_index, hop_num, feature_flags, max_sprw, label, args, debug=False):
    # first, extract subgraph
    # G_accumulated, G_next = G_pair
    set_index = list(set_index)
    sp_flag, rw_flag = feature_flags
    max_sp, rw_depth = max_sprw
    if len(set_index) > 1:
        G_accumulated = G_accumulated.copy()
        G_accumulated.remove_edges_from(combinations(set_index, 2))

    edge_index = torch.tensor(list(G_accumulated.edges)).long().t().contiguous()
    edge_index = torch.cat([edge_index, edge_index[[1, 0], ]], dim=-1) # !!!!! previously we forgot this line!!!!!

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # hop_num = 1 if args.bipartite else hop_num
    # hop_num = 3 when # layer =2 and prop_depth = 1
    # subgraph_node_old_index, new_edge_index, new_set_index, edge_mask = tgu.k_hop_subgraph(torch.tensor(set_index).long(), hop_num, edge_index, num_nodes=G_accumulated.number_of_nodes(), relabel_nodes=True)
    # subgraph_node_old_index, new_edge_index, new_set_index, edge_mask = tgu.k_hop_subgraph(torch.tensor(set_index).long(), num_hops=args.layers, edge_index=edge_index, num_nodes=G_accumulated.number_of_nodes(), relabel_nodes=True)
    subgraph_node_old_index, new_edge_index, new_set_index, edge_mask = tgu.k_hop_subgraph(torch.tensor(set_index).long(), num_hops=args.num_hops, edge_index=edge_index, num_nodes=G_accumulated.number_of_nodes(), relabel_nodes=True)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # print(edge_index)
    # print(set_index)
    # print(subgraph_node_old_index)
    # print(new_edge_index)
    # print(new_set_index)

    # reconstruct networkx graph object for the extracted subgraph
    num_nodes = subgraph_node_old_index.size(0)
    attributed_nodes_sub = [(new_id, {'bipartite': attributed_nodes_all[old_id][-1]['bipartite']}) for new_id, old_id in enumerate(subgraph_node_old_index.tolist())]
    new_G = nx.from_edgelist(new_edge_index.t().numpy().astype(dtype=np.int32), create_using=type(G_accumulated))
    new_G.add_nodes_from(attributed_nodes_sub)
    # print(new_G.nodes(data=True))
    assert new_G.number_of_nodes() == num_nodes
    # if new_edge_index.size()[1] > 0:
    #     print(new_edge_index.view(-1).numpy().max() + 1)
    #     print(num_nodes)
    #     assert new_edge_index.view(-1).numpy().max() + 1 ==  num_nodes

    # Construct x from x_list
    x_list = []
    attributes = G_accumulated.graph['attributes'] # attributes is None, only 1 column, corresponds to the degree defined in line 471 at dynG_helper()
    if attributes is not None:
        new_attributes = torch.tensor(attributes, dtype=torch.float32)[subgraph_node_old_index]
        if new_attributes.dim() < 2:
            new_attributes.unsqueeze_(1)
        x_list.append(new_attributes)
    # if deg_flag:
    #     x_list.append(torch.log(tgu.degree(new_edge_index[0], num_nodes=num_nodes, dtype=torch.float32).unsqueeze(1)+1))
    if sp_flag:
        features_sp_sample = get_features_sp_sample(new_G, new_set_index.numpy(), max_sp, args)
        features_sp_sample = torch.from_numpy(features_sp_sample).float()
        x_list.append(features_sp_sample)
    if rw_flag:
        adj = np.asarray(nx.adjacency_matrix(new_G, nodelist=np.arange(new_G.number_of_nodes(), dtype=np.int32)).todense().astype(np.float32))  # [n_nodes, n_nodes]
        features_rw_sample = get_features_rw_sample(adj, new_set_index.numpy(), rw_depth=rw_depth)
        features_rw_sample = torch.from_numpy(features_rw_sample).float()
        x_list.append(features_rw_sample)

    x = torch.cat(x_list, dim=-1)
    y = torch.tensor([label], dtype=torch.long) if label is not None else torch.tensor([0], dtype=torch.long)
    new_set_index = new_set_index.long().unsqueeze(0)

    d = {'x': x, 'edge_index': new_edge_index, 'y': y, 'set_indices': new_set_index}
    return d


def split_datalist(data_list, masks, args):
    # generate train_set
    train_mask, val_test_mask = masks
    num_graphs = len(data_list)
    assert((train_mask.sum()+val_test_mask.sum()).astype(np.int32) == num_graphs)
    assert(train_mask.shape[0] == num_graphs)
    train_indices = np.arange(num_graphs)[train_mask.astype(bool)]
    train_set = [data_list[i] for i in train_indices]
    # generate val_set and test_set
    val_test_indices = np.arange(num_graphs)[val_test_mask.astype(bool)]
    val_test_labels = np.array([data.y for data in data_list], dtype=np.int32)[val_test_indices]
    val_indices, test_indices = train_test_split(val_test_indices, test_size=int(args.test_ratio/(args.val_ratio+args.test_ratio)*len(val_test_indices)), stratify=val_test_labels)
    val_set = [data_list[i] for i in val_indices]
    test_set = [data_list[i] for i in test_indices]
    return train_set, val_set, test_set


def load_datasets(train_set, val_set, test_set, bs):
    num_workers = 0
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def generate_set_indices_labels(G_pair, task, logger, args):
    logger.info('Labels unavailable. Generating training/test instances from dataset ...')

    G_pair = list(map(lambda x: x.to_undirected(), G_pair))  # the prediction task completely ignores directions
    n_pos_edges, pos_edges_train, pos_edges_test, neg_edges = sample_pos_neg_sets(G_pair, args)  # each shape [n_pos_samples, set_size], note hereafter each "edge" may contain more than 2 nodes
    assert(n_pos_edges == neg_edges.shape[0])
    n_pos_edges_train = pos_edges_train.shape[0]
    n_pos_edges_test = pos_edges_test.shape[0]
    set_indices = np.concatenate([pos_edges_train, pos_edges_test, neg_edges], axis=0)
    # test_pos_indices = random.sample(range(n_pos_edges), pos_test_size)  # randomly pick pos edges for test
    test_pos_indices = list(range(n_pos_edges_train, n_pos_edges))
    test_neg_indices = list(range(n_pos_edges, n_pos_edges + n_pos_edges_test))  # pick first pos_test_size neg edges for test
    test_mask = get_mask(test_pos_indices + test_neg_indices, length=2*n_pos_edges)
    train_mask = np.ones_like(test_mask) - test_mask
    labels = np.concatenate([np.ones((n_pos_edges, )), np.zeros((n_pos_edges, ))]).astype(np.int32)
    # do not need the remove action at following step any more:
    #     1) since our G_accumulated doesn't have test_pos edges (unlike in DE-GNN and SEAL)!!!!!!!!!!!!!!!!!!!
    #     2) however, there is another remove in get_data_sample() of utils.py --> we need this remove, since it's for avoilding label leakage!!
    # G.remove_edges_from([node_pair for set_index in list(set_indices[test_pos_indices]) for node_pair in combinations(set_index, 2)])

    # permute everything for stable training
    # permutation do not change the content being permutated (won't mess up between training and test)
    permutation = np.random.permutation(2*n_pos_edges)
    set_indices = set_indices[permutation]
    labels = labels[permutation]
    train_mask, test_mask = train_mask[permutation], test_mask[permutation]
    logger.info('Generate {} train+val+test instances in total.'.format(set_indices.shape[0]))
    return G_pair, labels, set_indices, (train_mask, test_mask)


def get_mask(idx, length):
    mask = np.zeros(length)
    mask[idx] = 1
    return np.array(mask, dtype=np.int8)


# def sample_pos_neg_sets(G_pair, task, data_usage=1.0):
def sample_pos_neg_sets(G_pair, args):
    G_accumulated, G_next = G_pair
    # here we define the target links in (t+1) as those new ones that have never appeared between [1,t]
    # instead of all links that appeared between [t,t+1]
    # i.e., the above line implement the data split in the slides P53
    # e.g., G_accumulated.edges=[e1,e2,e3], G_next.edges=[e3,e4,e5], then new_edges_(t+1)=[e4,e5]
    ############################################# get positive edges ###################################################
    pos_edges_train = np.array(list(G_accumulated.edges), dtype=np.int32)
    filtered_new_edges = list(G_next.edges) if not args.constraint else [_ for _ in list(G_next.edges) if G_accumulated.degree(_[0])>0 and G_accumulated.degree(_[1])>0]

    if args.target == 'lp':
        # pos_edges_test = np.array(list(G_next.edges), dtype=np.int32)
        # pos_edges_test = np.array(filtered_new_edges, dtype=np.int32)
        pos_edges_test = np.array(list(set(filtered_new_edges)), dtype=np.int32)
    elif args.target == 'new_lp':
        # pos_edges_test = np.array(list(set(list(G_next.edges)) - set(list(G_accumulated.edges))), dtype=np.int32)
        pos_edges_test = np.array(list(set(filtered_new_edges) - set(list(G_accumulated.edges))), dtype=np.int32)
    else:
        raise NotImplementedError

    if args.train_ratio < 1-1e-6:
        pos_edges_train, _ = retain_partial(pos_edges_train, ratio=args.train_ratio)
    pos_edges_test, _ = retain_partial(pos_edges_test, ratio=args.val_ratio+args.test_ratio)
    # pos_edges_train = np.random.choice(pos_edges_train, args.train_ratio, replace=False)
    # pos_edges_test = np.random.choice(pos_edges_test, (args.val_ratio + args.test_ratio), replace=False)
    n_pos_edges = pos_edges_train.shape[0] + pos_edges_test.shape[0]
    ############################################# get negative edges ###################################################
    set_size = 2
    neg_edges = np.array(sample_neg_sets(G_pair, n_pos_edges, set_size, args), dtype=np.int32)
    return n_pos_edges, pos_edges_train, pos_edges_test, neg_edges


def retain_partial(indices, ratio):
    sample_i = np.random.choice(indices.shape[0], int(ratio * indices.shape[0]), replace=False)
    return indices[sample_i], sample_i


# def sample_neg_sets(G_pair, n_samples, set_size):
def sample_neg_sets(G_pair, n_samples, set_size, args):
    G_accumulated, G_next = G_pair
    neg_sets = []
    n_nodes = G_next.number_of_nodes()
    max_iter = 1e9
    count = 0
    while len(neg_sets) < n_samples:
        count += 1
        if count > max_iter:
            raise Exception('Reach max sampling number of {}, input graph density too high'.format(max_iter))
        if not args.bipartite:
            candid_set = [int(random.random() * n_nodes) for _ in range(set_size)]
        else:
            node_types = ['b0', 'b1']
            random.shuffle(node_types)
            candid_set = [random.choice(remapped_nodes[_]) for _ in node_types]
            # print(candid_set)
        for node1, node2 in combinations(candid_set, 2):
            # if not G.has_edge(node1, node2):
            # if (not G_accumulated.has_edge(node1, node2)) and (not G_next.has_edge(node1, node2)):
            complement_flag = (not G_accumulated.has_edge(node1, node2)) and (not G_next.has_edge(node1, node2))
            existing_nodes_flag = (G_accumulated.degree(node1)>0) and (G_accumulated.degree(node2)>0)
            flag = complement_flag if not args.constraint else (complement_flag and existing_nodes_flag)
            if flag:
                neg_sets.append(candid_set)
                break
    return neg_sets


def split_dataset(n_samples, test_ratio, stratify=None):
    train_indices, test_indices = train_test_split(list(range(n_samples)), test_size=test_ratio, stratify=stratify)
    train_mask = get_mask(train_indices, n_samples)
    test_mask = get_mask(test_indices, n_samples)
    return train_mask, test_mask


# def split_indices(num_graphs, test_ratio, stratify=None):
#     test_size = int(num_graphs*test_ratio)
#     val_size = test_size
#     train_val_set, test_set = train_test_split(np.arange(num_graphs), test_size=test_size, shuffle=True, stratify=stratify)
#     train_set, val_set = train_test_split(train_val_set, test_size=val_size, shuffle=True, stratify=stratify[train_val_set])
#     return train_set, val_set, test_set


def parallel_worker(x):
    return get_data_sample(*x)


def get_features_sp_sample(G, node_set, max_sp, args):
    if not args.bipartite:
        dim = max_sp + 2
        set_size = len(node_set)
        sp_length = np.ones((G.number_of_nodes(), set_size), dtype=np.int32) * -1
        for i, node in enumerate(node_set):
            for node_ngh, length in nx.shortest_path_length(G, source=node).items():
                sp_length[node_ngh, i] = length
        sp_length = np.minimum(sp_length, max_sp)
        onehot_encoding = np.eye(dim, dtype=np.float64)  # [n_features, n_features]
        features_sp = onehot_encoding[sp_length].sum(axis=1)
        return features_sp
    else:
        dim = 2 * max_sp + 2
        set_size = len(node_set)
        # print(node_set)
        assert G.nodes[node_set[0]]['bipartite'] + G.nodes[node_set[1]]['bipartite'] == 1

        sp_length = np.ones((G.number_of_nodes(), set_size), dtype=np.int32) * -1
        for i, node in enumerate(node_set):
            for node_ngh, length in nx.shortest_path_length(G, source=node).items():
                sp_length[node_ngh, i] = length
        sp_length = np.minimum(sp_length, max_sp)

        features_sp = np.ones(G.number_of_nodes(), dtype=np.int32) * -1
        for idx, _ in enumerate(sp_length):
            if sum(_) == -2:
                raise NotImplementedError
            elif 0 in _:
                features_sp[idx] = 0
            elif -1 in _:
                assert max(_) > 0
                features_sp[idx] = max(_)
            else:
                assert min(_) > 0
                features_sp[idx] = min(_)

        for idx, hop in enumerate(features_sp):
            features_sp[idx] = (2 * hop) if G.nodes[idx]['bipartite'] == 0 else (2 * hop + 1)
        onehot_encoding = np.eye(dim, dtype=np.float64)  # [n_features, n_features]
        features_sp = onehot_encoding[features_sp]
        # print(features_sp)
        return features_sp


def get_features_rw_sample(adj, node_set, rw_depth):
    epsilon = 1e-6
    adj = adj / (adj.sum(1, keepdims=True) + epsilon)
    rw_list = [np.identity(adj.shape[0])[node_set]]
    for _ in range(rw_depth):
        rw = np.matmul(rw_list[-1], adj)
        rw_list.append(rw)
    features_rw_tmp = np.stack(rw_list, axis=2)  # shape [set_size, N, F]
    # pooling
    features_rw = features_rw_tmp.sum(axis=0)
    return features_rw


def get_hop_num(prop_depth, layers, max_sprw, feature_flags):
    # TODO: may later use more rw_depth to control as well?
    return int(prop_depth * layers) + 1   # in order to get the correct degree normalization for the subgraph


def shortest_path_length(graph):
    sp_length = np.ones([graph.number_of_nodes(), graph.number_of_nodes()], dtype=np.int32) * -1
    for node1, value in nx.shortest_path_length(graph):
        for node2, length in value.items():
            sp_length[node1][node2] = length

    return sp_length


def collect_tri_sets(G):
    tri_sets = set(frozenset([node1, node2, node3]) for node1 in G for node2, node3 in combinations(G.neighbors(node1), 2) if G.has_edge(node2, node3))
    return [list(tri_set) for tri_set in tri_sets]


def retain_partial(indices, ratio):
    sample_i = np.random.choice(indices.shape[0], int(ratio * indices.shape[0]), replace=False)
    return indices[sample_i], sample_i


def pagerank_inverse(adj, alpha=0.90):
    adj /= (adj.sum(axis=-1, keepdims=True) + 1e-12)
    return np.linalg.inv(np.eye(adj.shape[0]) - alpha * np.transpose(adj, axes=(0,1)))


def get_degrees(G):
    num_nodes = G.number_of_nodes()
    return np.array([G.degree[i] for i in range(num_nodes)])


# TODO: 1. check if storage allows, send all data to gpu 5. (optional) add directed graph
# TODO: 6. (optional) enable using original node attributes as initial feature (only need to modify file readin)
# TODO: 7. (optional) rw using sparse matrix for multiplication
