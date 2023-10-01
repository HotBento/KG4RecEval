import numpy as np
import pandas as pd
import torch
import os

from shutil import copyfile

from recbole.data.interaction import Interaction
from recbole.model.knowledge_aware_recommender import *
from CorrectRippleNet import CorrectRippleNet
from MCRec import MCRec

from recbole.data.dataset.kg_dataset import KnowledgeBasedDataset

from recbole.utils import init_logger, init_seed
from recbole.config import Config
from recbole.data import create_dataset


import argparse

@torch.no_grad()
def full_sort_scores(uid_series, model, test_data, device=None):
    """Calculate the scores of all items for each user in uid_series.

    Note:
        The score of [pad] and history items will be set into -inf.

    Args:
        uid_series (numpy.ndarray or list): User id series.
        model (AbstractRecommender): Model to predict.
        test_data (FullSortEvalDataLoader): The test_data of model.
        device (torch.device, optional): The device which model will run on. Defaults to ``None``.
            Note: ``device=None`` is equivalent to ``device=torch.device('cpu')``.

    Returns:
        torch.Tensor: the scores of all items for each user in uid_series.
    """
    device = device or torch.device("cpu")
    uid_series = torch.tensor(uid_series)
    uid_field = test_data.dataset.uid_field
    dataset = test_data.dataset
    model.eval()

    if not test_data.is_sequential:
        input_interaction = dataset.join(Interaction({uid_field: uid_series}))
        history_item = test_data.uid2history_item[uid_series.cpu()]
        # for i , hist_iid in enumerate(history_item):
        #     if hist_iid == None:
        #         print((hist_iid,i))
        history_row = torch.cat(
            [torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_item)]
        )
        history_col = torch.cat(list(history_item))
        history_index = history_row, history_col
    else:
        _, index = (dataset.inter_feat[uid_field] == uid_series[:, None]).nonzero(
            as_tuple=True
        )
        input_interaction = dataset[index]
        history_index = None

    # Get scores of all items
    input_interaction = input_interaction.to(device)
    try:
        scores = model.full_sort_predict(input_interaction)
    except NotImplementedError:
        input_interaction = input_interaction.repeat_interleave(dataset.item_num)
        input_interaction.update(
            test_data.dataset.get_item_feature().to(device).repeat(len(uid_series))
        )
        scores = model.predict(input_interaction)

    scores = scores.view(-1, dataset.item_num)
    scores[:, 0] = -np.inf  # set scores of [pad] to -inf
    if history_index is not None:
        scores[history_index] = -np.inf  # set scores of history items to -inf

    return scores


def full_sort_topk(uid_series, model, test_data, k, device=None):
    """Calculate the top-k items' scores and ids for each user in uid_series.

    Note:
        The score of [pad] and history items will be set into -inf.

    Args:
        uid_series (numpy.ndarray): User id series.
        model (AbstractRecommender): Model to predict.
        test_data (FullSortEvalDataLoader): The test_data of model.
        k (int): The top-k items.
        device (torch.device, optional): The device which model will run on. Defaults to ``None``.
            Note: ``device=None`` is equivalent to ``device=torch.device('cpu')``.

    Returns:
        tuple:
            - topk_scores (torch.Tensor): The scores of topk items.
            - topk_index (torch.Tensor): The index of topk items, which is also the internal ids of items.
    """
    scores = full_sort_scores(uid_series, model, test_data, device)
    return torch.topk(scores, k)

def parse_args():
    parser = argparse.ArgumentParser()
    # Experiment type
    parser.add_argument('--experiment', type=str, default='false',
                        choices=['false', 'decrease', 'size', 'coldstart'],
                        help='Choose the type of experiment.'
                        )
    # General settings
    parser.add_argument('--dataset', type=str, default='movielens-100k',
                        choices=['movielens-100k', 'Amazon_Books-part', 'lfm1b-tracks-part'],
                        help='Choose the dataset.')
    parser.add_argument('--model', type=str, nargs='+',
                        default=['KGCN', 'RippleNet', 'CFKG', 'CKE', 'KGIN', 'KGNNLS', 'KTUP'],
                        help='Choose the type of models.')
    parser.add_argument('--worker_num', type=int, default=1,
                        help='Number of workers. No more than the gpu number.')
    parser.add_argument('--rate', type=float, nargs='+',
                        default=[0, 0.25, 0.5, 0.75, 1.0],
                        help='Rate list of the experiments.')
    parser.add_argument('--topk', type=float, nargs='+', default=[10],
                        help='Length of recommendation list used in evaluation.')
    parser.add_argument('--eval_times', type=int, default=50,
                        help='Evaluation times.')
    parser.add_argument('--metrics', type=str, nargs='+',
                        default=['Recall', 'MRR', 'NDCG', 'Hit', 'Precision'],
                        help='Evaluation metrics.')
    # Experiment settings
    parser.add_argument('--test_type_list', type=str, nargs='+',
                        default=['kg'],
                        help='Detailed experiment type.')
    # Only for cold start experiment
    parser.add_argument('--test_user_ratio', type=float, default=0.1,
                        help='Cold-start test user number. Only for cold-start experiment.')
    parser.add_argument('--cs_threshold', type=int, default=1,
                        help='Threshold for cold-start users. Only for cold-start experiment.')
    parser.add_argument('--offset', type=int, default=0, help='Offset of cuda core.')
    return parser.parse_args()

def get_model(model_type_str: str):
    if model_type_str == 'KGCN':
        model_type = KGCN
    elif model_type_str == 'RippleNet':
        model_type = CorrectRippleNet
    elif model_type_str == 'CFKG':
        model_type = CFKG
    elif model_type_str == 'CKE':
        model_type = CKE
    elif model_type_str == 'KGIN':
        model_type = KGIN
    elif model_type_str == 'KGNNLS':
        model_type = KGNNLS
    elif model_type_str == 'KTUP':
        model_type = KTUP
    elif model_type_str == 'MCCLK':
        model_type = MCCLK
    elif model_type_str == 'MKR':
        model_type = MKR
    elif model_type_str == 'MCRec':
        model_type = MCRec
    else:
        raise NameError('Invalid recommender system.')
    
    return model_type

def get_dataset(config_dict, dataset_str, model_type_str='KGCN'):
    model_type = get_model(model_type_str)
    
    # Get original dataset
    config = Config(model=model_type, dataset=dataset_str, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)

    dataset : KnowledgeBasedDataset = create_dataset(config)

    return dataset

def copy_dataset(suffix:str, src_path:str, temp_path:str, dataset_str:str):
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    for src_file in os.listdir(src_path):
        temp_file = '{}-fake{}.{}'.format(dataset_str, suffix, src_file.split('.')[-1])
        copyfile(os.path.join(src_path, src_file), os.path.join(temp_path, temp_file))

def filter_data(test_type_list, dataset_list, model_list, rate_list, result_path, experiment_type):
    for test_type in test_type_list:
        for dataset_str in dataset_list:
            for model in model_list:
                for rate in rate_list:
                    path = os.path.join(result_path, dataset_str, experiment_type, test_type, model+'_'+rate+'.txt')
                    if not os.path.exists(path):
                        continue
                    file_data = ''
                    with open(path, 'r') as file:
                        line = file.readline()
                        file_data += line
                        metrics_num = len(line.split('\t'))
                        for line in file:
                            if len(line.split('\t')) == metrics_num:
                                file_data += line
                    with open(path, 'w') as file:
                        file.write(file_data)

def add_self_relation(dataset: str):
    link_pd = pd.read_table(os.path.join('./dataset', dataset, dataset + '.link'))
    kg_pd = pd.read_table(os.path.join('./dataset', dataset, dataset + '.kg'))
    entity_set = set(kg_pd.loc[:, 'head_id:token']) | set(kg_pd.loc[:, 'tail_id:token']) | set(link_pd.loc[:,'entity_id:token'])
    
    with open(os.path.join('./dataset/', dataset, dataset + '.kg'), 'a') as dst_file:
        for i in entity_set:
            dst_file.write('{}\t{}\t{}\n'.format(i, 'self_to_self', i))