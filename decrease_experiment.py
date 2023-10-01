import os
import torch
import random
import numpy as np
import pandas as pd

from utils import copy_dataset, add_self_relation

def run_once(args_dict:dict, device: torch.device):
    # Environment settings
    dataset_str     : str   = args_dict['dataset']
    test_type       : str   = args_dict['test_type']
    rate            : float = args_dict['rate']

    # Generate fake kg
    suffix = str(device).split(':')[-1]
    fake_dataset = '{}-fake{}'.format(dataset_str, suffix)
    src_path = './dataset/{}/'.format(dataset_str)
    temp_path = os.path.join('./dataset/', fake_dataset)
    copy_dataset(suffix, src_path, temp_path, dataset_str)

    src_pd = pd.read_table(os.path.join(src_path, dataset_str+'.kg'))
    if test_type == 'fact':
        n_fact = src_pd.shape[0]
        dst_pd = src_pd.sample(round(n_fact*rate))
        # dst_pd.to_csv(os.path.join(temp_path, '{}-fake{}.kg'.format(dataset_str, suffix)), '\t', index=False)
    elif test_type == 'relation':
        relation_set = set(src_pd.loc[:, 'relation_id:token'].tolist())
        n_relation = len(relation_set)
        drop_relation_set = set(random.sample(list(relation_set), k=round(n_relation*(1-rate))))

        dst_pd = src_pd.loc[~src_pd.loc[:, 'relation_id:token'].isin(drop_relation_set), :]
        # dst_pd.to_csv(os.path.join(temp_path, '{}-fake{}.kg'.format(dataset_str, suffix)), '\t', index=False)
    elif test_type == 'entity':
        entity_set = set(src_pd.loc[:, 'head_id:token'].tolist())
        entity_set.update(set(src_pd.loc[:, 'tail_id:token'].tolist()))
        n_entity = len(entity_set)
        drop_entity_set = set(random.sample(list(entity_set), k=round(n_entity*(1-rate))))

        dst_pd = src_pd.loc[~(src_pd.loc[:,'head_id:token'].isin(drop_entity_set) | src_pd.loc[:, 'tail_id:token'].isin(drop_entity_set)), :]
        # dst_pd.to_csv(os.path.join(temp_path, '{}-fake{}.kg'.format(dataset_str, suffix)), '\t', index=False)
    else:
        raise ValueError('Unsupported test type.')
    
    dst_pd.to_csv(os.path.join(temp_path, fake_dataset + '.kg'), '\t', index=False)
    add_self_relation(fake_dataset)