import logging
import time
import os
import socket
from multiprocessing import Process, Lock


def set_up_log(args, sys_argv):
    log_dir = args.log_dir
    if args.phase == 'metatest':
        dataset_log_dir = os.path.join(log_dir, args.dataset, args.model)
    else:
        dataset_log_dir = os.path.join(log_dir, args.dataset, 'metatrain', args.model)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(dataset_log_dir):
        os.makedirs(dataset_log_dir)
    file_path = os.path.join(dataset_log_dir, '{}.log'.format(str(time.time())))

    # if args.phase == 'metatrain':
    #     file_path = os.path.join(dataset_log_dir, '{}_metatrain.log'.format(str(time.time())))
    # else:
    #     file_path = os.path.join(dataset_log_dir, '{}.log'.format(str(time.time())))

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('Create log file at {}'.format(file_path))
    logger.info('Command line executed: python ' + ' '.join(sys_argv))
    logger.info('Full args parsed:')
    logger.info(args)
    return logger


def save_performance_result(args, logger, results):
    # summary_file = args.summary_file
    summary_file = 'result_summary_{}.log'.format(args.dataset)
    if summary_file != 'test':
        summary_file = os.path.join(args.log_dir, args.dataset, summary_file)
    else:
        return
    dataset = args.dataset
    interval = args.interval
    metatest_ratio = args.metatest_ratio
    seed = args.seed
    # model_name = '-'.join([args.model, str(args.num_hops), str(args.layers), str(args.bs), str(args.lr)])
    # model_name = '{}_{}(hop)_{}(layer)_{}(bs)_{}(lr)_{}(search_epoch)'.format(args.model, args.num_hops, args.layers, args.bs, args.lr, args.epoch)
    model_name = '{}_{}[hop]_{}[layer]_{}[bs]_{}[lr]'.format(args.model, args.num_hops, args.layers, args.bs, args.lr)
    log_name = os.path.split(logger.handlers[1].baseFilename)[-1]
    server = socket.gethostname()

    macro_auc, macro_acc, macro_ap, micro_auc, micro_acc, micro_ap, search_time, retrain_time = list(map(str, results))

    macro_results = 'macro (auc, acc, ap): {}, {}, {}'.format(macro_auc, macro_acc, macro_ap)
    micro_results = 'micro (auc, acc, ap): {}, {}, {}'.format(micro_auc, micro_acc, micro_ap)
    time_cost = 'time (search, retrian): {}, {}'.format(search_time, retrain_time)

    # line = '\t'.join([dataset, model_name, str(seed), '>> macro (auc, acc, ap)', macro_auc, macro_acc, macro_ap, '>> micro (auc, acc, ap)', micro_auc, micro_acc, micro_ap, log_name, server]) + '\n'
    # line = '\t'.join([dataset, args.target, model_name, str(seed), str(args.epoch), macro_results, micro_results, time_cost, log_name, server]) + '\n'
    line = '\t'.join([dataset, str(interval), str(metatest_ratio), args.target, model_name, str(seed), str(args.epoch), macro_results, micro_results, time_cost, log_name, server]) + '\n'
    with open(summary_file, 'a') as f:
        f.write(line)  # WARNING: process unsafe!






