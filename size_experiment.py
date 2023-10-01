import os
import torch
import random
import numpy as np
import pandas as pd

from utils import get_dataset, copy_dataset, add_self_relation

def run_once(args_dict:dict, device: torch.device):
    # Environment settings
    dataset_str     : str   = args_dict['dataset']
    test_type       : str   = args_dict['test_type']
    rate            : float = args_dict['rate']
    model_type_str  : str   = args_dict['model_type_str']
    topk            : int   = args_dict['topk']
    worker_num      : int   = args_dict['worker_num']

    config_dict = {'seed':random.randint(0, 10000), 'gpu_id':tuple(range(worker_num)), 
                   'topk':topk, 'checkpoint_dir':'saved{}/'.format(str(device).split(':')[-1])}

    dataset = get_dataset(config_dict, dataset_str, model_type_str)

    # Generate fake kg
    suffix = str(device).split(':')[-1]
    fake_dataset = '{}-fake{}'.format(dataset_str, suffix)
    src_path = './dataset/{}/'.format(dataset_str)
    temp_path = os.path.join('./dataset/', fake_dataset)
    copy_dataset(suffix, src_path, temp_path, dataset_str)

    src_file = open(os.path.join(src_path, '{}.kg'.format(dataset_str)), 'r')
    dst_file = open(os.path.join(temp_path, fake_dataset + '.kg'), 'w')

    fact_num = len(src_file.readlines()) - 1
    entity_token_array = dataset.field2id_token['entity_id']
    relation_token_array = dataset.field2id_token['relation_id']
    if test_type == 'entity':
        # RecBole library adds 1 additional entity that don't actually exist.
        new_entity_num = round(rate * dataset.entity_num)
        if rate <= 1:
            entity_token_array = entity_token_array[:new_entity_num+1]
        else:
            new_entity_num = new_entity_num-dataset.entity_num
            new_entity_array = np.array(['new_entity_{}'.format(i) for i in range(new_entity_num)])
            entity_token_array = np.append(entity_token_array, new_entity_array)
    elif test_type == 'relation':
        # RecBole library adds 2 additional relations that don't actually exist in kg file.
        relation_num = max(round(rate * (dataset.relation_num-2)),1)
        if rate <= 1:
            relation_token_array = relation_token_array[:relation_num+1]
        else:
            new_relation_num = relation_num - dataset.relation_num
            new_relation_array = np.array(['new_relation_{}'.format(i) for i in range(new_relation_num)])
            relation_token_array = np.append(relation_token_array, new_relation_array)
    elif test_type == 'fact':
        fact_num = round(fact_num * rate)
    else:
        raise NameError('Invalid test type.')
    
    # labels
    dst_file.write('head_id:token\trelation_id:token\ttail_id:token\n')
    for _ in range(fact_num):
        head = entity_token_array[random.randint(1,len(entity_token_array)-1)]
        relation = relation_token_array[random.randint(1,len(relation_token_array)-1)]
        tail = entity_token_array[random.randint(1,len(entity_token_array)-1)]
        dst_file.write('{}\t{}\t{}\n'.format(head, relation, tail))

    while not src_file.closed:
        src_file.close()
    while not dst_file.closed:
        dst_file.close()

    add_self_relation(fake_dataset)