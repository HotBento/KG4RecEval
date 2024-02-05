import os
import torch
import random
import pandas as pd

from utils import get_dataset, copy_dataset


def run_once(args_dict:dict, dst_dataset:str, seed=0):
    # Environment settings
    dataset_str     : str   = args_dict['dataset']
    test_type       : str   = args_dict['test_type']
    rate            : float = args_dict['rate']
    model_type_str  : str   = args_dict['model_type_str']
    topk            : int   = args_dict['topk']

    random.seed(seed)
    torch.random.manual_seed(seed)

    config_dict = {'seed':seed, 'gpu_id':tuple(range(torch.cuda.device_count())), 
                   'topk':topk, 'checkpoint_dir':'temp/'}

    dataset = get_dataset(config_dict, dataset_str, model_type_str)

    # Generate random kg
    src_path = os.path.join('./dataset/', dataset_str)
    temp_path = os.path.join('./dataset/', dst_dataset)
    copy_dataset(dataset_str, dst_dataset)

    if test_type == 'kg':
        src_file = open(os.path.join(src_path, f'{dataset_str}.kg'), 'r')
        dst_file = open(os.path.join(temp_path, f'{dst_dataset}.kg'), 'w')

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
