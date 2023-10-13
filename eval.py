import os
import pandas as pd
import random
from shutil import rmtree

from recbole.config.configurator import Config
from recbole.model.knowledge_aware_recommender import *

from logging import getLogger
from recbole.utils import init_logger, init_seed
from utils import get_model
from recbole.trainer import KGTrainer
from recbole.config import Config
from recbole.sampler import Sampler
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.data import create_dataset, data_preparation
from recbole.data.dataset import Dataset
from recbole.data.interaction import Interaction
import numpy as np

def evaluate_kg(model_type_str, device, topk, save_path=None, fake_kg_path='movielens-100k-fake', metrics=None, gpu_num=1, experiment_type=None):
    config_dict = {'seed':random.randint(0, 10000), 'gpu_id':tuple(range(gpu_num)), 'checkpoint_dir':'saved{}/'.format(str(device).split(':')[-1])}
    if metrics != None:
        config_dict['metrics'] = metrics
    if topk != None:
        config_dict['topk'] = topk
    config_file_list = None
    model_type = get_model(model_type_str)
    if model_type_str == 'MCRec':
        config_file_list = ['MCRec_config.yaml']
    elif model_type_str == 'RippleNet':
        config_file_list = ['RippleNet_config.yaml']
    elif model_type_str == 'KGIN':
        config_file_list = ['KGIN_config.yaml']

    # evaluation for fake kg
    config = Config(model=model_type, dataset=fake_kg_path, config_file_list=config_file_list, config_dict=config_dict)
    init_logger(config)
    logger = getLogger()
    # logger.info(config)
    config['device'] = device

    # dataset filtering
    dataset = create_dataset(config)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # KG preparation for no knowledge experiment
    if experiment_type == 'noknowledge':
        dataset = _renew_kg(dataset, train_data)

    # model loading and initialization
    if model_type_str == 'MCRec':
        model = model_type(config, dataset, train_data, valid_data).to(config['device'])
    elif model_type_str == 'RippleNet' or model_type_str == 'KGIN':
        model = model_type(config, dataset, train_data).to(config['device'])
    else:
        model = model_type(config, dataset).to(config['device'])

    # trainer loading and initialization
    trainer = KGTrainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=False)

    # model evaluation
    test_result = trainer.evaluate(test_data, show_progress=False)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))

    if save_path is not None:
        print("Path saved in: {}".format(save_path))
        if not os.path.exists(save_path):
            file = open(save_path, 'a')
            header = '\t'.join(['{}@{}'.format(metric, k) for k in config['topk'] for metric in config['metrics']])
            file.write('{}\n'.format(header))
        else:
            file = open(save_path, 'a')
        file.write('{}\n'.format('\t'.join([str(test_result['{}@{}'.format(metric.lower(), k)]) for k in config['topk'] for metric in config['metrics']])))
        file.flush()
        while not file.closed:
            file.close()

    rmtree('./saved{}/'.format(str(device).split(':')[-1]))

def cold_start_evaluate(model_type_str, device, topk, save_path=None, fake_kg_path='movielens-100k-fake', metrics=None, gpu_num=1, experiment_type=None):
    config_dict = {'seed':random.randint(0, 10000), 'gpu_id':tuple(range(gpu_num)), 'checkpoint_dir':'saved{}/'.format(str(device).split(':')[-1])}
    if metrics != None:
        config_dict['metrics'] = metrics
    if topk != None:
        config_dict['topk'] = topk
    config_file_list = None
    model_type = get_model(model_type_str)
    if model_type_str == 'MCRec':
        config_file_list = ['MCRec_config.yaml']
    elif model_type_str == 'RippleNet':
        config_file_list = ['RippleNet_config.yaml']
    elif model_type_str == 'KGIN':
        config_file_list = ['KGIN_config.yaml']

    config = Config(model=model_type, dataset=fake_kg_path, config_file_list=config_file_list, config_dict=config_dict)
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    config['device'] = device
    

    # dataset filtering
    dataset = create_dataset(config)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # KG preparation for no knowledge experiment
    if experiment_type == 'noknowledge':
        dataset = _renew_kg(dataset, train_data)

    # model loading and initialization
    if model_type_str == 'MCRec':
        model = model_type(config, dataset, train_data, valid_data).to(config['device'])
    elif model_type_str == 'RippleNet' or model_type_str == 'KGIN':
        model = model_type(config, dataset, train_data).to(config['device'])
    else:
        model = model_type(config, dataset).to(config['device'])

    # trainer loading and initialization
    trainer = KGTrainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=False)

    # model evaluation
    test_result = trainer.evaluate(test_data, show_progress=False)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))

    cs_test_dataset = pd.read_table(os.path.join('./dataset', fake_kg_path, '{}.test'.format(fake_kg_path)))

    test_user_id = []
    test_item_id = []
    cs_user_id_list = cs_test_dataset.loc[:,'user_id'].tolist()
    cs_item_id_list = cs_test_dataset.loc[:,'item_id'].tolist()
    drop_list = []
    for i in range(len(cs_user_id_list)):
        try:
            dataset.token2id('item_id', str(cs_item_id_list[i]))
            dataset.token2id('user_id', str(cs_user_id_list[i]))
        except(ValueError):
            drop_list.append(i)
            continue
        test_item_id.append(dataset.token2id('item_id', str(cs_item_id_list[i])))
        test_user_id.append(dataset.token2id('user_id', str(cs_user_id_list[i])))
    
    cs_test_dataset.drop(index=drop_list, inplace=True)
    cs_test_dataset.loc[:, 'user_id'] = test_user_id
    cs_test_dataset.loc[:, 'item_id'] = test_item_id
    cs_test_dataset['user_id'] = pd.to_numeric(cs_test_dataset['user_id'])
    cs_test_dataset['item_id'] = pd.to_numeric(cs_test_dataset['item_id'])

    cs_test_dataset = dataset.copy(dataset._dataframe_to_interaction(cs_test_dataset))
    cs_test_sampler = Sampler('test', cs_test_dataset)
    cs_test_sampler = cs_test_sampler.set_phase('test')
    cs_test_data = FullSortEvalDataLoader(config, cs_test_dataset, cs_test_sampler, shuffle=False)
    cs_test_result = trainer.evaluate(cs_test_data, show_progress=False)

    logger.info('cold start test result: {}'.format(cs_test_result))

    if save_path is not None:
        print("Path saved in: {}".format(save_path))
        if not os.path.exists(save_path):
            file = open(save_path, 'a')
            original_header = '\t'.join(['{}@{}'.format(metric, k) for k in config['topk'] for metric in config['metrics']])
            file.write('{}\n'.format(original_header))
        else:
            file = open(save_path, 'a')
        file.write('{}\n'.format('\t'.join([str(cs_test_result['{}@{}'.format(metric.lower(), k)]) for k in config['topk'] for metric in config['metrics']])))
        file.flush()
        while not file.closed:
            file.close()

    rmtree('./saved{}/'.format(str(device).split(':')[-1]))

def _renew_kg(dataset, train_data):
    head = dataset.token2id(dataset.entity_field, dataset.id2token(dataset.uid_field, train_data._dataset.inter_feat.user_id))
    tail = dataset.token2id(dataset.entity_field, dataset.id2token(dataset.iid_field, train_data._dataset.inter_feat.item_id))
    relation = [1]*len(head)+[2]*len(head)
    interaction = {'head_id':np.concatenate([head,tail]), 'relation_id':relation, 'tail_id':np.concatenate([tail,head])}

    interaction = Interaction(interaction)

    dataset.kg_feat = interaction
    return dataset