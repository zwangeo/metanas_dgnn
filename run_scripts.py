import utils
from utils import *
import os
import sys
# sys.path.append('..')
from trainers.train_degnn import train_model as train_model_degnn
from trainers.train_autogel import train_model as train_model_autogel
from trainers.train_autogel import retrain, S_Recorder, A_Recorder
from trainers.train_metanas import *
import warnings
from log import *
warnings.filterwarnings('ignore')
criterion = torch.nn.functional.cross_entropy


def setup_run(args, logger):
    from utils import read_file, meta_split, get_dimension
    global attributed_nodes_all
    attributed_nodes_all, dynG_accumulated, dynG_next, task = read_file(args, logger)
    dynG_pair_metatrain, dynG_pair_metatest = meta_split(dynG_accumulated, dynG_next, args)
    if args.phase == 'metatest':
        args.cur_idx = args.start_idx + len(dynG_pair_metatrain)
        get_dimension(dynG_accumulated, dynG_next, task, args, logger)
        return dynG_pair_metatest, task
    else:
        args.cur_idx = args.start_idx
        get_dimension(dynG_accumulated, dynG_next, task, args, logger)
        return dynG_pair_metatrain, dynG_pair_metatest, task


def train_R(args, logger, device):
    dynG_pair_metatrain, _, task = setup_run(args, logger)
    learner_w_grad, learner_wo_grad, metalearner, optimizer, meta_optimizer = utils.metatrain_init(args, logger, device)
    # learner_w_grad, metalearner, optimizer, meta_optimizer = metatrain_init(args, logger, device)
    performance = ddict(list)
    # performance = {'metatrain': ddict(list), 'metatest': ddict(list)}
    args.phase = 'metatrain'

    for idx, G_pair in enumerate(dynG_pair_metatrain):
        # if idx > 0:
        #     print(learner_w_grad.log_alpha_agg.weight)
        #     print(learner_w_grad.preprocess.weight)
        learner_w_grad.train()
        learner_wo_grad.train()

        dataloaders = utils.get_data(G_pair, task=task, args=args, logger=logger)
        train_loader, val_loader, test_loader = dataloaders
        storage = utils.estimate_storage(dataloaders, ['train_loader', 'val_loader', 'test_loader'], logger)

        # trian learner
        cI, _ = train_learner(learner_w_grad, metalearner, dataloaders, args, logger, device, performance, optimizer=optimizer)
        # cI = train_learner(learner_w_grad, metalearner, train_loader, val_loader, args, logger, device, optimizer=optimizer)
        logger.info('*************************************************************************************************')
        # train meta-learner
        train_metalearner(cI, learner_w_grad, learner_wo_grad, metalearner, meta_optimizer, test_loader, args, logger, device, performance)
        torch.nn.utils.clip_grad_norm_(metalearner.parameters(), max_norm=args.clip)
        meta_optimizer.step()
        learner_w_grad.rec_reset()
        args.cur_idx += 1
        #
        # for n,p in metalearner.named_parameters():
        #     print(n)
        #     print(p.view(-1))
        # print(metalearner.metalstm.WF.grad.view(-1))
        # print(metalearner.metalstm.WI.grad.view(-1))
        # print(metalearner.metalstm.WF.view(-1))
        # print(metalearner.metalstm.WI.view(-1))

    if args.save_R:
        utils.save_ckpt(metalearner, args)

    # logger.info('Summary >>> meta loss for each meta-train eps: {}'.format(performance['metatrain']['loss']))
    logger.info('Summary >>> meta loss for each meta-train eps: {}'.format(performance['loss']))
    logger.info('Finish meta-training!')
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')


    search_time = np.sum(performance['search_time'])
    retrain_time = np.sum(performance['retrain_time'])
    logger.info('seartch time: {} ### time for each eps: {}'.format(search_time, performance['search_time']))
    logger.info('retrain time: {} ### time for each eps: {}'.format(retrain_time, performance['retrain_time']))
    logger.info('search_max_step: {}'.format(performance['search_max_step']))
    logger.info('retrain_max_step: {}'.format(performance['retrain_max_step']))
    ################################################## finish metatrain ################################################


# dyn_degnn
def run_dyn_degnn(args, logger, device):
    dynG_pair, task = setup_run(args, logger)
    performance = ddict(list)
    for idx, G_pair in enumerate(dynG_pair):
        learner, optimizer = utils.setup_model(args, logger, device)
        dataloaders = utils.get_data(G_pair, task=task, args=args, logger=logger)
        storage = utils.estimate_storage(dataloaders, ['train_loader', 'val_loader', 'test_loader'], logger)
        train_model_degnn(learner, optimizer, dataloaders, args, logger, device, performance)
        args.cur_idx += 1
    results = utils.summarize_results(performance, args, logger)
    save_performance_result(args, logger, results)


# rand_sgd
def run_rand_sgd(args, logger, device):
    dynG_pair, task = setup_run(args, logger)
    performance = ddict(list)
    for idx, G_pair in enumerate(dynG_pair):
        learner, optimizer = utils.setup_model(args, logger, device)
        # print(learner.temperature)
        # print(learner.max_step)

        dataloaders = utils.get_data(G_pair, task=task, args=args, logger=logger)
        storage = utils.estimate_storage(dataloaders, ['train_loader', 'val_loader', 'test_loader'], logger)
        _ = train_model_autogel(learner, optimizer, dataloaders, args, logger, device, performance)
        args.cur_idx += 1
        # print(_.log_alpha_agg.weight)
    results = utils.summarize_results(performance, args, logger)
    save_performance_result(args, logger, results)


# inc_sgd
def run_inc_sgd(args, logger, device):
    dynG_pair, task = setup_run(args, logger)
    # learner, optimizer = utils.setup_model(args, logger, device)
    performance = ddict(list)
    for idx, G_pair in enumerate(dynG_pair):
        learner, optimizer = utils.setup_model(args, logger, device)
        if idx > 0:
            learner.load_state_dict(_.state_dict())
            # print(learner.log_alpha_agg.weight)
            # print(learner.preprocess.weight)
            # print(_.log_alpha_agg.weight)
            # print(learner.log_alpha_agg.weight)
        # print(learner.temperature)
        # print(learner.max_step)


        dataloaders = utils.get_data(G_pair, task=task, args=args, logger=logger)
        storage = utils.estimate_storage(dataloaders, ['train_loader', 'val_loader', 'test_loader'], logger)
        _ = train_model_autogel(learner, optimizer, dataloaders, args, logger, device, performance)

        # learner.rec_reset()
        # learner.update_z_hard()
        # print(learner.log_alpha_agg.weight)

        args.cur_idx += 1
        # print(_.log_alpha_agg.weight)

    results = utils.summarize_results(performance, args, logger)
    save_performance_result(args, logger, results)


# meta_sgd
def run_meta_sgd(args, logger, device):
    dynG_pair_metatest, task = setup_run(args, logger)
    _, _, _, _, _ = utils.metatrain_init(args, logger, device)

    # performance = {'metatrain': ddict(list), 'metatest': ddict(list)}
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    logger.info('Start meta-test!')
    # args.phase = 'metatest'
    # args.cur_idx += len(dynG_pair_metatrain)


    # dynG_pair, task = setup_run(args, logger)
    performance = ddict(list)
    for idx, G_pair in enumerate(dynG_pair_metatest):
        # learner, optimizer = setup_model(args, logger, device)
        metalearner = utils.resume_ckpt(args, device)
        learner = utils.get_model(model_name=args.model, layers=args.layers, in_features=args.in_features, out_features=args.out_features, prop_depth=args.prop_depth, args=args, logger=logger).to(device)
        learner.copy_flat_params(metalearner.metalstm.cI.data)
        learner.update_z_hard()
        optimizer = utils.get_optimizer(learner, args, type='bls')
        #
        # print(learner.temperature)
        # print(learner.max_step)

        dataloaders = utils.get_data(G_pair, task=task, args=args, logger=logger)
        storage = utils.estimate_storage(dataloaders, ['train_loader', 'val_loader', 'test_loader'], logger)
        _ = train_model_autogel(learner, optimizer, dataloaders, args, logger, device, performance)
        args.cur_idx += 1
        # print(_.log_alpha_agg.weight)

    results = utils.summarize_results(performance, args, logger)
    save_performance_result(args, logger, results)


# rand_meta
def run_rand_meta(args, logger, device):
    ################################################## start metatest ##################################################
    # assert args.meta_init == False
    # args.learner_init = 'rand'
    dynG_pair_metatest, task = setup_run(args, logger)
    _, _, _, _, _ = utils.metatrain_init(args, logger, device)

    performance = ddict(list)
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    logger.info('Start meta-test!')
    # args.phase = 'metatest'
    # args.cur_idx += len(dynG_pair_metatrain)

    for idx, G_pair in enumerate(dynG_pair_metatest):
        metalearner = utils.resume_ckpt(args, device)
        learner_w_grad = utils.get_model(model_name=args.model, layers=args.layers, in_features=args.in_features, out_features=args.out_features, prop_depth=args.prop_depth, args=args, logger=logger).to(device)
        learner_w_grad.update_z_hard()
        optimizer = utils.get_optimizer(learner_w_grad, args, type='learner_w_grad')

        # print(learner_w_grad.temperature)
        # print(learner_w_grad.max_step)

        dataloaders = utils.get_data(G_pair, task=task, args=args, logger=logger)
        storage = utils.estimate_storage(dataloaders, ['train_loader', 'val_loader', 'test_loader'], logger)
        # cI, _ = train_learner(learner_w_grad, metalearner, dataloaders, args, logger, device, optimizer=optimizer)
        cI, _ = train_learner(learner_w_grad, metalearner, dataloaders, args, logger, device, performance, optimizer=optimizer)
        # retrain(learner_w_grad, optimizer, dataloaders, args, logger, device, performance['metatest'])
        retrain(_, optimizer, dataloaders, args, logger, device, performance)
        logger.info('*************************************************************************************************')
        args.cur_idx += 1
        # print(_.log_alpha_agg.weight)

    # logger.info('Summary >>> meta loss for each meta-train eps: {}'.format(performance['metatrain']['loss']))
    # results = utils.summarize_results(performance['metatest'], args, logger)
    results = utils.summarize_results(performance, args, logger)
    save_performance_result(args, logger, results)


# inc_meta
def run_inc_meta(args, logger, device):
    ################################################## start metatest ##################################################
    # args.learner_init = 'inc'
    dynG_pair_metatest, task = setup_run(args, logger)
    _, _, _, _, _ = utils.metatrain_init(args, logger, device)

    performance = ddict(list)
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    logger.info('Start meta-test!')
    # args.phase = 'metatest'
    # args.cur_idx += len(dynG_pair_metatrain)

    for idx, G_pair in enumerate(dynG_pair_metatest):
        metalearner = utils.resume_ckpt(args, device)
        learner_w_grad = utils.get_model(model_name=args.model, layers=args.layers, in_features=args.in_features, out_features=args.out_features, prop_depth=args.prop_depth, args=args, logger=logger).to(device)
        learner_w_grad.update_z_hard()
        optimizer = utils.get_optimizer(learner_w_grad, args, type='learner_w_grad')
        if idx > 0:
            learner_w_grad.load_state_dict(_.state_dict())
            #     print(learner_w_grad.log_alpha_agg.weight)
            #     print(learner_w_grad.preprocess.weight)
        # print(learner_w_grad.temperature)
        # print(learner_w_grad.max_step)

        dataloaders = utils.get_data(G_pair, task=task, args=args, logger=logger)
        storage = utils.estimate_storage(dataloaders, ['train_loader', 'val_loader', 'test_loader'], logger)
        # cI, _ = train_learner(learner_w_grad, metalearner, dataloaders, args, logger, device, optimizer=optimizer)
        cI, _ = train_learner(learner_w_grad, metalearner, dataloaders, args, logger, device, performance, optimizer=optimizer)
        retrain(learner_w_grad, optimizer, dataloaders, args, logger, device, performance)

        # 1) since the z_hard now is one-hot (discrete) due to calling of retrain(),
        # hence we need to call update_z_hard() to make it continuous for searching in the next eps
        # 2) we also need to call rec_reset() to rest the learner recorder
        learner_w_grad.update_z_hard()
        learner_w_grad.rec_reset()
        args.cur_idx += 1
        # print(learner_w_grad.log_alpha_agg.weight)

        logger.info('*************************************************************************************************')
    # logger.info('Summary >>> meta loss for each meta-train eps: {}'.format(performance['metatrain']['loss']))
    # results = utils.summarize_results(performance['metatest'], args, logger)
    results = utils.summarize_results(performance, args, logger)
    save_performance_result(args, logger, results)


# meta_meta
def run_meta_meta(args, logger, device):
    ################################################## start metatest ##################################################
    # assert args.meta_init == False
    # args.learner_init = 'rand'
    dynG_pair_metatest, task = setup_run(args, logger)
    _, _, _, _, _ = utils.metatrain_init(args, logger, device)

    performance = ddict(list)
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    logger.info('Start meta-test!')
    # args.phase = 'metatest'
    # args.cur_idx += len(dynG_pair_metatrain)

    for idx, G_pair in enumerate(dynG_pair_metatest):
        metalearner = utils.resume_ckpt(args, device)
        learner_w_grad = utils.get_model(model_name=args.model, layers=args.layers, in_features=args.in_features, out_features=args.out_features, prop_depth=args.prop_depth, args=args, logger=logger).to(device)
        ####################################################################################
        # the following line is the only difference between rand_meta and meta_meta variants
        learner_w_grad.copy_flat_params(metalearner.metalstm.cI.data)
        ####################################################################################
        learner_w_grad.update_z_hard()
        optimizer = utils.get_optimizer(learner_w_grad, args, type='learner_w_grad')
        # print(learner_w_grad.temperature)
        # print(learner_w_grad.max_step)

        dataloaders = utils.get_data(G_pair, task=task, args=args, logger=logger)
        storage = utils.estimate_storage(dataloaders, ['train_loader', 'val_loader', 'test_loader'], logger)
        # cI, _ = train_learner(learner_w_grad, metalearner, dataloaders, args, logger, device, optimizer=optimizer)
        cI, _ = train_learner(learner_w_grad, metalearner, dataloaders, args, logger, device, performance, optimizer=optimizer)
        # retrain(learner_w_grad, optimizer, dataloaders, args, logger, device, performance['metatest'])
        retrain(_, optimizer, dataloaders, args, logger, device, performance)
        logger.info('*************************************************************************************************')
        args.cur_idx += 1
        # print(_.log_alpha_agg.weight)

    # logger.info('Summary >>> meta loss for each meta-train eps: {}'.format(performance['metatrain']['loss']))
    # results = utils.summarize_results(performance['metatest'], args, logger)
    results = utils.summarize_results(performance, args, logger)
    save_performance_result(args, logger, results)