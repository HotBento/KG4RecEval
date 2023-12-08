import os
import torch
import random
import pandas as pd
import numpy as np

from utils import get_dataset, copy_dataset, add_self_relation

def run_once(args_dict:dict, device: torch.device):
    # Environment settings
    dataset_str     : str   = args_dict['dataset']
    test_type       : str   = args_dict['test_type']
    rate            : float = args_dict['rate']
    model_type_str  : str   = args_dict['model_type_str']
    topk            : int   = args_dict['topk']
    worker_num      : int   = args_dict['worker_num']
    ptr             : int   = args_dict['ptr']
    all_test_users  : list  = args_dict['all_test_users']
    cs_threshold    : int   = args_dict['cs_threshold']

    config_dict = {'seed':random.randint(0, 10000), 'gpu_id':tuple(range(worker_num)), 
                   'topk':topk, 'checkpoint_dir':'saved{}/'.format(str(device).split(':')[-1])}

    dataset = get_dataset(config_dict, dataset_str, model_type_str)
    type_dict = {'user_id':'token', 'item_id':'token', 'rating':'float', 'timestamp':'float'}

    # Split the inter file into two files
    suffix = str(device).split(':')[-1]
    fake_dataset = '{}-fake{}'.format(dataset_str, suffix)
    src_path = './dataset/{}/'.format(dataset_str)
    temp_path = os.path.join('./dataset/', fake_dataset)
    copy_dataset(suffix, src_path, temp_path, dataset_str)

    test_df = pd.DataFrame([], columns=dataset.inter_feat.columns)
    train_df = dataset.inter_feat.copy()
    test_user_list = all_test_users[ptr]
    for user in test_user_list:
        user_inter = dataset.inter_feat.loc[dataset.inter_feat.loc[:, 'user_id']==user,['user_id', 'item_id']]
        index = list(user_inter.index)
        train_index = random.choices(index, k=cs_threshold)
        test_index = list(set(index)-set(train_index))
        test_df = pd.concat([test_df, dataset.inter_feat.loc[test_index,:]])
        train_df = train_df.drop(index=test_index, axis=0)
    
    test_df = test_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    train_user_token = [dataset.id2token('user_id', i) for i in train_df.loc[:,'user_id'].tolist()]
    train_item_token = [dataset.id2token('item_id', i) for i in train_df.loc[:,'item_id'].tolist()]
    train_df.loc[:,'user_id'] = train_user_token
    train_df.loc[:,'item_id'] = train_item_token
    train_df.columns = [f'{column}:{type_dict.setdefault(column, "token")}' for column in train_df.columns]
    # if len(train_df.columns)==3:
    #     train_df.columns = ['user_id:token','item_id:token','timestamp:float']
    # else:
    #     train_df.columns = ['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float']

    test_user_token = [dataset.id2token('user_id', i) for i in test_df.loc[:,'user_id'].tolist()]
    test_item_token = [dataset.id2token('item_id', i) for i in test_df.loc[:,'item_id'].tolist()]
    test_df.loc[:,'user_id'] = test_user_token
    test_df.loc[:,'item_id'] = test_item_token

    test_df.to_csv(os.path.join(temp_path, fake_dataset + '.test'), sep='\t', index=False)
    train_df.to_csv(os.path.join(temp_path, fake_dataset + '.inter'), sep='\t', index=False)
    if test_type == 'random':
        src_file = open(os.path.join(src_path, dataset_str + '.kg'), 'r')
        dst_file = open(os.path.join(temp_path, fake_dataset + '.kg'), 'w')

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
    elif test_type[0] == 'd':
        src_pd = pd.read_table(os.path.join(src_path, dataset_str+'.kg'))
        if test_type == 'd_fact':
            n_fact = src_pd.shape[0]
            dst_pd = src_pd.sample(round(n_fact*rate))
            dst_pd.to_csv(os.path.join(temp_path, fake_dataset + '.kg'), '\t', index=False)
        elif test_type == 'd_relation':
            relation_set = set(src_pd.loc[:, 'relation_id:token'].tolist())
            n_relation = len(relation_set)
            drop_relation_set = set(random.sample(list(relation_set), k=round(n_relation*(1-rate))))

            dst_pd = src_pd.loc[~src_pd.loc[:, 'relation_id:token'].isin(drop_relation_set), :]
            dst_pd.to_csv(os.path.join(temp_path, fake_dataset + '.kg'), '\t', index=False)
        elif test_type == 'd_entity':
            entity_set = set(src_pd.loc[:, 'head_id:token'].tolist())
            entity_set.update(set(src_pd.loc[:, 'tail_id:token'].tolist()))
            n_entity = len(entity_set)
            drop_entity_set = set(random.sample(list(entity_set), k=round(n_entity*(1-rate))))

            dst_pd = src_pd.loc[~(src_pd.loc[:,'head_id:token'].isin(drop_entity_set) | src_pd.loc[:, 'tail_id:token'].isin(drop_entity_set)), :]
            dst_pd.to_csv(os.path.join(temp_path, fake_dataset + '.kg'), '\t', index=False)
    elif test_type[0] == 's':
        src_file = open(os.path.join(src_path, '{}.kg'.format(dataset_str)), 'r')
        dst_file = open(os.path.join(temp_path, fake_dataset + '.kg'), 'w')

        fact_num = len(src_file.readlines()) - 1
        entity_token_array = dataset.field2id_token['entity_id']
        relation_token_array = dataset.field2id_token['relation_id']
        if test_type == 's_entity':
            # RecBole library adds 1 additional entity that don't actually exist.
            new_entity_num = round(rate * dataset.entity_num)
            if rate <= 1:
                entity_token_array = entity_token_array[:new_entity_num+1]
            else:
                new_entity_num = new_entity_num-dataset.entity_num
                new_entity_array = np.array(['new_entity_{}'.format(i) for i in range(new_entity_num)])
                entity_token_array = np.append(entity_token_array, new_entity_array)
        elif test_type == 's_relation':
            # RecBole library adds 2 additional relations that don't actually exist in kg file.
            relation_num = max(round(rate * (dataset.relation_num-2)),1)
            if rate <= 1:
                relation_token_array = relation_token_array[:relation_num+1]
            else:
                new_relation_num = relation_num - dataset.relation_num
                new_relation_array = np.array(['new_relation_{}'.format(i) for i in range(new_relation_num)])
                relation_token_array = np.append(relation_token_array, new_relation_array)
        elif test_type == 's_fact':
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
        

def generate_test_user_list(con, dataset_str:str, test_user_ratio:float, device=torch.device('cuda:0')):
    config_dict = {'seed':random.randint(0, 10000),'checkpoint_dir':'saved{}/'.format(str(device).split(':')[-1])}

    dataset = get_dataset(config_dict, dataset_str)
    test_user_num = round(test_user_ratio * dataset.user_num)

    user_list = list(range(1, dataset.user_num))
    test_user_list = []
    while len(test_user_list) < test_user_num:
        if len(user_list) == 0:
            break
        user = user_list.pop(random.randint(0, len(user_list)-1))
        user_inter = dataset.inter_feat.loc[dataset.inter_feat.loc[:, 'user_id']==user,['user_id', 'item_id']]
        if user_inter.shape[0] < 25:
            continue
        test_user_list.append(user)
    con.send(test_user_list)