import numpy as np
import torch
import torch.nn as nn

from recbole.data.dataset.kg_dataset import KnowledgeBasedDataset
from recbole.data.dataloader.knowledge_dataloader import KnowledgeBasedDataLoader
from recbole.config.configurator import Config
from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType

from recbole.trainer import KGTrainer
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.model.knowledge_aware_recommender import CFKG

import torch.profiler as profiler

"""
MCRec
##################################################
Reference:
    Hu, Binbin, et al. "Leveraging meta-path based context for top-n recommendation with a neural co-attention model." Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 2018.
"""
class MCRec(KnowledgeRecommender):
    input_type = InputType.PAIRWISE
    def __init__(self, config:Config, dataset:KnowledgeBasedDataset, train_data:KnowledgeBasedDataLoader, valid_data):
        super().__init__(config, dataset)
        # load parameters info
        self.embedding_size :   int         = config["embedding_size"]
        self.metapath_type  :   list[str]   = config["metapath_type"] # length of metapath should be shorter than 4
        self.feature_size   :   int         = config["feature_size"]
        self.path_num       :   int         = config["path_num"]
        self.sample_size    :   int         = config["sample_size"]
        self.MLP_layers     :   list[int]   = config["MLP_layers"]
        self.mp_embed_size  :   int         = config["mp_embed_size"] # default 128
        self.att_size       :   int         = config["att_size"] # default 128

        # define embedding
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)

        # generate knowledge graph
        self.metapath_dict = self._sample_metapath(config, train_data, valid_data)

        # generate layers
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        self.conv_dict:nn.ModuleDict[str, nn.Conv1d] = nn.ModuleDict()
        for mp_type in self.metapath_type:
            self.conv_dict[mp_type] = nn.Conv1d(self.feature_size, self.mp_embed_size, len(mp_type), 1, padding='valid', device=self.device)

        self.metapath_attention_layers = nn.ModuleList([
            nn.Linear(2*self.embedding_size+self.mp_embed_size, self.att_size),
            self.activation,
            nn.Linear(self.att_size, 1),
            self.activation
        ])

        self.user_attention_layer = nn.Linear(self.embedding_size+self.mp_embed_size, self.embedding_size)
        self.item_attention_layer = nn.Linear(self.embedding_size+self.mp_embed_size, self.embedding_size)

        MLP_input_list = [self.embedding_size*2 + self.mp_embed_size] + self.MLP_layers[:-1]
        self.MLP = nn.ModuleList()
        for i in range(len(self.MLP_layers)):
            self.MLP.append(nn.Linear(MLP_input_list[i], self.MLP_layers[i]))
            self.MLP.append(self.activation)
        
        self.predict_layer = nn.Linear(self.MLP_layers[-1], 1)

        # generate loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()

        # parameters initiallization
        self.apply(xavier_normal_initialization)

    def _generate_pretrain_embedding(self, ex_config, train_data, valid_data) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        The original paper didn't provide the code for pre-training.
        Here we use CFKG instead as it is more suitable for KG-based RSs.
        '''
        config_dict = {'embedding_size':self.feature_size, 'gpu_id':tuple(range(int(str(self.device).split(':')[-1]))), 'checkpoint_dir':'saved{}/'.format(str(self.device).split(':')[-1])}
        config = Config(model=CFKG, dataset=ex_config["dataset"], config_dict=config_dict)
        config['device'] = self.device
        dataset = create_dataset(config)
        data_preparation(config, dataset)
        model = CFKG(config, dataset).to(config['device'])
        trainer = KGTrainer(config, model)
        trainer.fit(train_data, valid_data, show_progress=False)
        self.entity_feature = model.entity_embedding.weight
        self.user_feature = model.user_embedding.weight

    def _sample_metapath(self, config:Config, train_data:KnowledgeBasedDataLoader, valid_data) -> dict[str,dict[int,dict[int,list]]]:
        '''
        The method used in the official code are somewhat different from that described in the paper.
        Here we refer to the method in the paper for processing.
        output:
            metapath_dict[metapath_type][start_user][end_item] :
                sampled metapath list of start_user->metapath_type->end_item
        '''
        metapath_dict:dict[str,dict[int,dict[int,list]]] = dict()
        self._generate_pretrain_embedding(config, train_data, valid_data)
        self.interaction_matrix = train_data._dataset.inter_matrix(form="coo").astype(np.float32)
        head_entities = train_data._dataset.head_entities.tolist()
        tail_entities = train_data._dataset.tail_entities.tolist()
        kg = {}
        for i in range(len(head_entities)):
            head_ent = head_entities[i]
            tail_ent = tail_entities[i]
            kg.setdefault(head_ent, [])
            kg[head_ent].append(tail_ent)
        users = self.interaction_matrix.row.tolist()
        items = self.interaction_matrix.col.tolist()
        user_dict = {}
        item_dict = {}
        for i in range(len(users)):
            user = users[i]
            item = items[i]
            user_dict.setdefault(user, [])
            item_dict.setdefault(item, [])
            user_dict[user].append(item)
            item_dict[item].append(user)
        for mp in self.metapath_type:
            metapath_dict.setdefault(mp, dict())
            for start in range(self.n_users):
                path_list = self._sample_step(start, mp, kg, user_dict, item_dict)
                if path_list == None:
                    continue
                for path in path_list:
                    metapath_dict[mp].setdefault(path[0], dict())
                    metapath_dict[mp][path[0]].setdefault(path[-1], [])
                    metapath_dict[mp][path[0]][path[-1]].append(path)
            for i in range(len(metapath_dict[mp])):
                for j in range(len(metapath_dict[mp][i])):
                    if len(range(len(metapath_dict[mp][i][j]))) > self.path_num:
                        metapath_dict[mp][i][j] = metapath_dict[mp][i][j][:self.path_num]
            
        return metapath_dict

    def _sample_step(self, start:int, metapath_type:str, kg:dict[int,list], user_dict:dict[int,list], 
                     item_dict:dict[int,list]):
        if metapath_type[:2] == 'ui':
            if start not in user_dict:
                return None
            next_list = user_dict[start]
        elif metapath_type[:2] == 'iu':
            if start not in item_dict:
                return None
            next_list = item_dict[start]
        elif metapath_type[:2] == 'ii' or metapath_type[:2] == 'ei':
            if start not in kg:
                return None
            next_list = []
            temp = kg[start]
            for i in temp:
                if i < self.n_items:
                    next_list.append(i)
        elif metapath_type[:2] == 'ie' or metapath_type[:2] == 'ee':
            if start not in kg:
                return None
            next_list = kg[start]
        else:
            raise ValueError('Unknown meta-path type.')
        if len(next_list) == 0:
            return None
        
        # Sample
        if metapath_type[0] == 'u':
            start_e = self.user_feature[start]
        elif metapath_type[0] == 'i' or metapath_type[0] == 'e':
            start_e = self.entity_feature[start]
        score_list = []
        for next_node in next_list:
            if metapath_type[1] == 'u':
                next_e = self.user_feature[next_node]
            elif metapath_type[1] == 'i' or metapath_type[1] == 'e':
                next_e = self.entity_feature[next_node]
            score_list.append((next_node, float(next_e@start_e/next_e.norm()*start_e.norm())))
        score_list.sort(key=lambda x:x[1], reverse=True)
        score_list = score_list[:min(self.sample_size, len(score_list))]
        next_list = [i[0] for i in score_list]
        
        if len(metapath_type) == 2:
            return [[i] for i in next_list]
        
        path = []
        for i in next_list:
            res = self._sample_step(metapath_type[0], metapath_type[1:], kg, user_dict, item_dict)
            if res == None:
                continue
            path = path + [[i]+res[j] for j in res]
        if path == []:
            return None
        return path

    def _user_attention(self, user_latent:torch.Tensor, metapath_latent:torch.Tensor) -> torch.Tensor:
        '''
        input:
            user_latent: batch_size * embedding_size
            metapath_latent: batch_size * metapath_type_num * mp_embed_size
        output:
            output: batch_size * embedding_size
        '''  
        input = torch.concatenate([user_latent, metapath_latent], -1)
        output = self.user_attention_layer(input) # batch_size * embedding_size
        output = self.activation(output)
        atten = self.softmax(output, dim=-1)
        output = torch.mul(user_latent, atten)
        return output
    
    def _item_attention(self, item_latent:torch.Tensor, metapath_latent:torch.Tensor) -> torch.Tensor:
        '''
        input:
            item_latent: batch_size * embedding_size
            metapath_latent: batch_size * metapath_type_num * mp_embed_size
        output:
            output: batch_size * embedding_size
        '''  
        input = torch.concatenate([item_latent, metapath_latent], -1)
        output = self.user_attention_layer(input) # batch_size * embedding_size
        output = self.activation(output)
        atten = self.softmax(output, dim=-1)
        output = torch.mul(item_latent, atten)
        return output
    
    def _get_metapath_embedding(self, metapath_latent_dict:dict[str, torch.Tensor]) -> torch.Tensor:
        '''
        metapath_latent in metapath_latent_dict: batch_size * path_num * timestamps * feature_size
        self.feature_size is equal to length in the source code
        '''
        path_output = []
        for mp_type in self.metapath_type:
            x = metapath_latent_dict[mp_type].reshape((-1, len(mp_type), self.mp_embed_size))
            x = self.conv_dict[mp_type](x.transpose(-1,-2)).transpose(-1,-2)
            x = x.reshape((-1, self.path_num, x.shape[-2], x.shape[-1]))
            x = self.activation(x)
            x, _ = torch.max(x,-2) # batch_size * path_num * mp_embed_size
            x = self.dropout(x)
            x, _ = torch.max(x,-2) # batch_size * mp_embed_size
            path_output.append(x)
        return torch.stack(path_output,-2) # batch_size * metapath_type_num * mp_embed_size
    
    def _get_metapath_latent_dict(self, user:torch.Tensor, item:torch.Tensor) -> dict[str, torch.Tensor]:
        metapath_latent_dict:dict[str, torch.Tensor] = dict()
        user = user.tolist()
        item = item.tolist()
        for mp in self.metapath_type:
            # batch_size * path_num * timestamps * feature_size
            metapath_latent_dict[mp] = torch.zeros((len(user), self.path_num, len(mp), self.feature_size), device=self.device)
            for index in range(len(user)):
                u = user[index]
                i = item[index]
                if u not in self.metapath_dict[mp]:
                    continue
                if i not in self.metapath_dict[mp][u]:
                    continue
                for j,path in enumerate(self.metapath_dict[mp][u][i]):
                    for k,node in enumerate(path):
                        if mp[k] == 'u':
                            metapath_latent_dict[mp][index][j][k] = self.user_feature[node]
                        elif mp[k] == 'i' or mp[k] == 'e':
                            metapath_latent_dict[mp][index][j][k] = self.entity_feature[node]
                        else:
                            raise ValueError('Unknown meta-path.')
                        
        return metapath_latent_dict
    
    def _metapath_attention(self, user_latent:torch.Tensor, item_latent:torch.Tensor, metapath_latent:torch.Tensor) -> torch.Tensor:
        '''
        input:
            user_latent: batch_size * embedding_size
            item_latent: batch_size * embedding_size
            metapath_latent: batch_size * metapath_type_num * mp_embed_size
        output:
            output: batch_size * mp_embed_size
        '''        
        user_latent = user_latent.unsqueeze(-2)
        item_latent = item_latent.unsqueeze(-2)
        repeats = [1] * user_latent.dim()
        repeats[-2] = metapath_latent.size(-2)
        user_latent = user_latent.repeat(repeats)
        item_latent = item_latent.repeat(repeats)

        input = torch.concatenate([user_latent, item_latent, metapath_latent], -1) # batch_size * metapath_type_num * (mp_embed_size + 2*embedding_size)
        for mp_att_layer in self.metapath_attention_layers:
            input = mp_att_layer(input)
        output = input.squeeze(-1) # batch_size * metapath_type_num
        atten = self.softmax(output, dim=-1) # batch_size * metapath_type_num
        output = torch.matmul(atten.unsqueeze(-2), metapath_latent).squeeze(-2)
        return output

    '''
    run time analysis
    '''
    # def forward(self, user:torch.Tensor, item:torch.Tensor) -> torch.Tensor:
    #     with profiler.record_function("embedding"):
    #         user_latent = self.user_embedding(user)
    #         item_latent = self.entity_embedding(item)

    #     with profiler.record_function("get_metapath_latent_dict"):
    #         metapath_latent_dict = self._get_metapath_latent_dict(user, item)
    #     with profiler.record_function("get_metapath_embedding"):
    #         path_output = self._get_metapath_embedding(metapath_latent_dict) # batch_size * metapath_type_num * mp_embed_size
    #     with profiler.record_function("metapath_attention"):
    #         path_output = self._metapath_attention(user_latent, item_latent, path_output)
    #     with profiler.record_function("ui_attention"):
    #         user_attention = self._user_attention(user_latent, path_output)
    #         item_attention = self._item_attention(item_latent, path_output)

    #     with profiler.record_function("output"):
    #         output = torch.concat([user_attention, path_output, item_attention], -1)
    #         for layer in self.MLP:
    #             output = layer(output)

    #         output = self.predict_layer(output)
    #         output = self.sigmoid(output)

    #     return output

    def forward(self, user:torch.Tensor, item:torch.Tensor) -> torch.Tensor:
        user_latent = self.user_embedding(user)
        item_latent = self.entity_embedding(item)

        metapath_latent_dict = self._get_metapath_latent_dict(user, item)
        path_output = self._get_metapath_embedding(metapath_latent_dict) # batch_size * metapath_type_num * mp_embed_size
        path_output = self._metapath_attention(user_latent, item_latent, path_output)
        user_attention = self._user_attention(user_latent, path_output)
        item_attention = self._item_attention(item_latent, path_output)

        output = torch.concat([user_attention, path_output, item_attention], -1)
        for layer in self.MLP:
            output = layer(output)

        output = self.predict_layer(output)
        output = self.sigmoid(output)

        return output
    
    def calculate_loss(self, interaction:Interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        pos_output = self.forward(user, pos_item)
        neg_output = self.forward(user, neg_item)

        predict = torch.cat((pos_output, neg_output)).view(-1)
        target = torch.zeros(len(user) * 2, dtype=torch.float32, device=self.device)
        target[: len(user)] = 1
        rec_loss = self.bce_loss(predict, target)

        return rec_loss

    def predict(self, interaction:Interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item).view(-1)

    def full_sort_predict(self, interaction:Interaction):
        user_index = interaction[self.USER_ID]
        item_index = torch.tensor(range(self.n_items), device=self.device)

        user = torch.unsqueeze(user_index, dim=1).repeat(1, item_index.shape[0])
        user = torch.flatten(user)
        item = torch.unsqueeze(item_index, dim=0).repeat(user_index.shape[0], 1)
        item = torch.flatten(item)
        result = self.forward(user, item).view(-1)

        return result
