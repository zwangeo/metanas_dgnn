import os
import argparse
import copy
from log import *
# from train import *
# from simulate import *
from utils import *
from metalearner import MetaLearner
import torch
from run_scripts import *
# from run_scripts import run_rand_sgd, run_inc_sgd, run_meta_sgd, run_rand_meta, run_inc_meta, run_meta_meta


def run(args, logger, device):
    if args.phase == 'metatrain':
        # for _ in [0.1, 0.3, 0.7, 0.9]:
        #     args.metatrain_ratio = _
        #     args.metatest_ratio = 1-args.metatrain_ratio
        #     train_R(args, logger, device)
        # args.log_dir = os.path.join(args.log_dir, 'metatrain')
        args.metatest_ratio = 1-args.metatrain_ratio
        train_R(args, logger, device)

    else:
        args.metatrain_ratio = 1-args.metatest_ratio
        if args.model == 'dyn_degnn':
            run_dyn_degnn(args, logger, device)
        elif args.model == 'rand_sgd':
            run_rand_sgd(args, logger, device)
        elif args.model == 'inc_sgd':
            run_inc_sgd(args, logger, device)
        elif args.model == 'meta_sgd':
            run_meta_sgd(args, logger, device)

        elif args.model == 'rand_meta':
            run_rand_meta(args, logger, device)
        elif args.model == 'inc_meta':
            run_inc_meta(args, logger, device)
        elif args.model == 'meta_meta':
            run_meta_meta(args, logger, device)
        else:
            raise NotImplementedError


def main():
    parser = argparse.ArgumentParser('Interface for Meta-NAS framework')
    parser.add_argument('--phase', type=str, default='metatrain', help='metatrain or metatest phase')
    parser.add_argument('--save_R', type=bool, default=True, help='whether to save the trained meta-controller')

    parser.add_argument('--search_based', type=bool, default=True, help='whether the model is search_based or not')
    parser.add_argument('--model', type=str, default='meta_meta', help='model to use')  # in ['dyn_degnn', 'rand_sgd', 'inc_sgd', 'meta_sgd', 'rand_meta', 'inc_meta', 'meta_meta']
    # parser.add_argument('--learner_init', type=str, default='inc', help='how to initiate the learner') # in ['rand', 'inc', 'meta'], denotes several variants of meta-based models
    # parser.add_argument('--learner_update', type=str, default='meta', help='how to update the learner') # in ['sgd', 'meta'], denotes several variants of meta-based models

    parser.add_argument('--target', type=str, default='new_lp', help='whether to predict all links between [t,t+1] or only new links') # in ['new_lp', 'lp']
    # ref for 'constraint': DySAT appendix C; line 148 of https://github.com/aravindsankar28/DySAT/blob/777f290dcef38c28c390ba4642ce45621e7cedb4/utils/preprocess.py
    parser.add_argument('--constraint', type=bool, default=False, help='whether to constraint evaluation links to those formed with existing nodes (in G_accumulated)')

    parser.add_argument('--delta_version', type=int, default=1, help='version for data split based on different interval')
    parser.add_argument('--dataset', type=str, default='bc_alpha', help='dataset name') # currently relying on dataset to determine task
    parser.add_argument('--bipartite', type=bool, default=False, help='whether the dataset is bipartite or not')
    parser.add_argument('--del_days_head', type=int, default=0, help='number of days to remove from the beginning')
    parser.add_argument('--del_days_tail', type=int, default=0, help='number of days to remove from the end')
    parser.add_argument('--interval', type=int, default=28, help='time interval for each graph snapshots') # days
    # parser.add_argument('--n_eps', type=int, default=0, help='total number of episodes')
    parser.add_argument('--start_idx', type=int, default=10, help='the first snapshot selected')
    parser.add_argument('--end_idx', type=int, default=49, help='the last snapshot selected')
    parser.add_argument('--cur_idx', type=int, default=0, help='current index of the snapshot')
    parser.add_argument('--eps_usage', type=str, default='partial', help='for bls: whether to evaluate on all eps in dynG, or only eps in meta-testset', choices=['all', 'partial'])
    parser.add_argument('--second_stage', type=str, default='retrain', help='whether to adapt or retrain after search')

    # set up for the meta-learner
    parser.add_argument('--input_size', type=int, default=4, help='input size for the first LSTM')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size for the first LSTM')
    parser.add_argument('--n_learner_params', type=int, default=-1, help='number of parameters in learners that are controlled by meta')  # to be decided later, in [# beta, # (beta+omega)]
    parser.add_argument('--meta_target', type=str, default='beta', help='the set(s) of params whose optimization to be controlled by meta')  # in ['beta', 'beta_omega']
    parser.add_argument('--metatrain_ratio', type=float, default=0.5, help='ratio of the snapshots used for meta_test')
    parser.add_argument('--metatest_ratio', type=float, default=0.5, help='ratio of the snapshots used for meta_test')

    # general model and training setting
    parser.add_argument('--num_hops', type=int, default=2, help='for subgraph extraction')
    parser.add_argument('--layers', type=int, default=2, help='largest number of layers')
    parser.add_argument('--in_features', type=int, default=-1, help='input dimension for the learner, to be updated')
    parser.add_argument('--hidden_features', type=int, default=64, help='hidden dimension for the learner')
    # parser.add_argument('--hidden_features', type=int, default=32, help='hidden dimension for the learner')
    parser.add_argument('--out_features', type=int, default=2, help='output dimension for the learner, default value 2 for the LP task')

    parser.add_argument('--metric', type=str, default='auc', help='metric for evaluating performance', choices=['acc', 'auc', 'ap'])
    parser.add_argument('--seed', type=int, default=3, help='seed to initialize all the random modules')
    parser.add_argument('--gpu', type=int, default=3, help='gpu id')

    # parser.add_argument('--train_ratio', type=float, default=0.1, help='use partial positive edges in G_accumulated for training')
    parser.add_argument('--train_ratio', type=float, default=0.1, help='use partial positive edges in G_accumulated for training')
    parser.add_argument('--val_ratio', type=float, default=0.25, help='ratio of the val against whole')
    parser.add_argument('--test_ratio', type=float, default=0.75, help='ratio of the test against whole')

    parser.add_argument('--max_num_pos_train', type=int, default=5e3, help='the maximum of positive edges sampled for training')
    parser.add_argument('--directed', type=bool, default=False, help='(Currently unavailable) whether to treat the graph as directed')
    parser.add_argument('--parallel', default=False, action='store_true', help='(Currently unavailable) whether to use multi cpu cores to prepare data')

    # features and positional encoding
    parser.add_argument('--prop_depth', type=int, default=1, help='propagation depth (number of hops) for one layer')
    parser.add_argument('--use_degree', type=bool, default=True, help='whether to use node degree as the initial feature')
    parser.add_argument('--use_attributes', type=bool, default=False, help='whether to use node attributes as the initial feature')
    parser.add_argument('--feature', type=str, default='sp', help='distance encoding category: shortest path or random walk (landing probabilities)')  # sp (shortest path) or rw (random walk)
    parser.add_argument('--rw_depth', type=int, default=3, help='random walk steps')  # for random walk feature
    parser.add_argument('--max_sp', type=int, default=3, help='maximum distance to be encoded for shortest path feature')

    # model training
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs to search')
    parser.add_argument('--retrain_epoch', type=int, default=200, help='number of epochs to retrain')
    parser.add_argument('--kill_cnt', type=int, default=15, help='early stop if test perf does not improve after kill_cnt epochs (retrain stage)')

    parser.add_argument('--bs', type=int, default=32, help='minibatch size')
    # previously we use 1e-4 as lr, but maybe it's a bit large? Now we try smaller ones
    # parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--meta_lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    # previously we use clip=0.2
    parser.add_argument('--clip', type=float, default=0.1, help='gradient clipping')
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate')

    # simulation (valid only when dataset == 'simulation')
    parser.add_argument('--k', type=int, default=3, help='node degree (k) or synthetic k-regular graph')
    parser.add_argument('--n', nargs='*', help='a list of number of nodes in each connected k-regular subgraph')
    parser.add_argument('--N', type=int, default=1000, help='total number of nodes in simultation')
    parser.add_argument('--T', type=int, default=6, help='largest number of layers to be tested')

    # save & logging & debug
    # parser.add_argument('--preprocessed_data', type=str, default='../complete_design_small_interval_debug/preprocessed_data', help='save preprocessed trainset, valset and testset')
    parser.add_argument('--preprocessed_data', type=str, default='./preprocessed_data', help='save preprocessed trainset, valset and testset')
    parser.add_argument('--save', type=str, default='./meta_checkpoints', help='save metalearner at the end of metatrain phase')
    parser.add_argument('--log_dir', type=str, default='./log', help='log directory')  # sp (shortest path) or rw (random walk)
    parser.add_argument('--summary_file', type=str, default='result_summary.log', help='brief summary of training result')  # sp (shortest path) or rw (random walk)
    parser.add_argument('--debug', default=False, action='store_true',
                        help='whether to use debug mode')
    sys_argv = sys.argv
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    check(args)
    # get_model_name(args)
    logger = set_up_log(args, sys_argv)
    set_random_seed(args)
    device = get_device(args)

    run(args, logger, device)

    # if args.model == 'dyn_degnn':
    #     run_dyn_degnn(args, logger, device)
    #
    # elif args.model == 'rand_sgd':
    #     run_rand_sgd(args, logger, device)
    # elif args.model == 'inc_sgd':
    #     run_inc_sgd(args, logger, device)
    # elif args.model == 'meta_sgd':
    #     run_meta_sgd(args, logger, device)
    #
    # elif args.model == 'rand_meta':
    #     run_rand_meta(args, logger, device)
    # elif args.model == 'inc_meta':
    #     run_inc_meta(args, logger, device)
    # elif args.model == 'meta_meta':
    #     run_meta_meta(args, logger, device)
    # else:
    #     raise NotImplementedError


if __name__ == '__main__':
    main()
