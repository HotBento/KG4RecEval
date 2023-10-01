import os
import torch
import random
import pandas as pd

from utils import get_dataset, copy_dataset


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

    # Generate random kg
    suffix = str(device).split(':')[-1]
    src_path = './dataset/{}/'.format(dataset_str)
    temp_path = os.path.join('./dataset/', '{}-fake{}'.format(dataset_str, suffix))
    copy_dataset(suffix, src_path, temp_path, dataset_str)

    if test_type == 'kg':
        src_file = open(os.path.join(src_path, '{}.kg'.format(dataset_str)), 'r')
        dst_file = open(os.path.join(temp_path, '{}-fake{}.kg'.format(dataset_str, suffix)), 'w')

        is_first = True
        for line in src_file:
            if is_first:
                dst_file.write(line)
                is_first = False
                continue
            if random.random() < rate:
                head = dataset.id2token('entity_id', random.randint(1,dataset.entity_num-1))
                relation = dataset.id2token('relation_id', random.randint(1,dataset.relation_num-1))
                tail = dataset.id2token('entity_id', random.randint(1,dataset.entity_num-1))
                dst_file.write('{}\t{}\t{}\n'.format(head, relation, tail))
            else:
                dst_file.write(line)
        while not src_file.closed:
            src_file.close()
        while not dst_file.closed:
            dst_file.close()
    else:
        raise NameError('Invalid test type.')
