import os
import torch
import random
import numpy as np
import pandas as pd

from utils import copy_dataset_and_filter, add_self_relation

def run_once(args_dict:dict, device: torch.device):
    # Environment settings
    dataset_str     : str   = args_dict['dataset']
    test_type       : str   = args_dict['test_type']
    rate            : float = args_dict['rate']

    # Convert interaction file to kg file
    suffix = str(device).split(':')[-1]
    fake_dataset = '{}-fake{}'.format(dataset_str, suffix)
    src_path = './dataset/{}/'.format(dataset_str)
    temp_path = os.path.join('./dataset/', fake_dataset)
    copy_dataset_and_filter(suffix, src_path, temp_path, dataset_str)

    inter_pd = pd.read_table(os.path.join(temp_path, fake_dataset+'.inter'))
    # use 'fact' as the default test type
    if test_type == 'fact':
        kg_pd = inter_pd.loc[:, ['user_id:token', 'item_id:token']]
        kg_pd.insert(1, 'relation_id:token', ['interaction']*kg_pd.shape[0])
        kg_pd.columns = ['head_id:token', 'relation_id:token', 'tail_id:token']


        kg_pd_t = inter_pd.loc[:, ['item_id:token', 'user_id:token']]
        kg_pd_t.insert(1, 'relation_id:token', ['interaction_t']*kg_pd_t.shape[0])
        kg_pd_t.columns = ['head_id:token', 'relation_id:token', 'tail_id:token']

        kg_pd = pd.concat([kg_pd, kg_pd_t], ignore_index=True)

        item_list = list(set(inter_pd.loc[:,'item_id:token']))
        link_pd = pd.DataFrame({'item_id:token':item_list, 'entity_id:token':item_list})
    else:
        raise ValueError('Unsupported test type.')
    
    kg_pd.to_csv(os.path.join(temp_path, fake_dataset + '.kg'), '\t', index=False)
    link_pd.to_csv(os.path.join(temp_path, fake_dataset + '.link'), '\t', index=False)
