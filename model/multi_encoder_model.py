'''
1.Multiencoder, and encourage starting from different cities
2.Layer decoder that glimpse unvisited then self attn cur/des
'''

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import math
import numpy as np
import os
import copy
from tqdm import tqdm

from model.encoder import GraphAttentionEncoder, MultiHeadAttentionLayer

from utils import TSP, Utils


class MEAM(nn.Module):
    def __init__(self, hidden_size=128, encoder_layers=3, decoder_layers=2, \
        u_clip=10, n_heads=8, n_encoders=5, topk=0, need_dt=True):
        super().__init__()
        # get model setting args
        self.model_setting_dict = locals()
        self.model_setting_dict.pop('self')
        self.model_setting_dict.pop('__class__')

        # model setting from init parameters
        self.hidden_size = hidden_size
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.u_clip = u_clip
        self.n_heads = n_heads
        self.n_encoders = n_encoders
        self.topk = topk  # topk self action mask, >0 valid
        self.need_dt = need_dt

        # model parameters
        # multi-encoders
        self.encoders = nn.ModuleList(
            [GraphAttentionEncoder(self.n_heads, self.hidden_size, self.encoder_layers, 2, 'batch', 4 * self.hidden_size) for _ in range(self.n_encoders)]
        )
        # glimpse + self attn decoders 
        self.cur_holder = nn.Parameter(torch.Tensor(self.hidden_size))
        self.des_holder = nn.Parameter(torch.Tensor(self.hidden_size))
        self.glimpseQ = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size, False) for _ in range(self.decoder_layers)]
        )
        self.glimpseK = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size, False) for _ in range(self.decoder_layers)]
        )
        self.glimpseV = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size, False) for _ in range(self.decoder_layers)]
        )
        self.glimpseVOut = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size, False) for _ in range(self.decoder_layers)]
        )
        self.self_attn_decoder_layers = nn.ModuleList(
            [MultiHeadAttentionLayer(self.n_heads, self.hidden_size, 4 * self.hidden_size, 'instance') for _ in range(self.decoder_layers)]
        )
        # probs
        self.probQ = nn.Linear(self.hidden_size, self.hidden_size, False)
        self.probK = nn.Linear(self.hidden_size, self.hidden_size, False)
        
        # optimizer, create one at first train or load from state dict
        self.optimizer = None
        # pos
        self.pos_embedding = MEAM.position_encoding_init(2, self.hidden_size)
        
    def forward(self, x_all: torch.Tensor, dt_graph: torch.Tensor=None, decode_type='sample'):
        '''
        Forward call for this model

        :param x_all: float tensor of size B * n * 2
        :param dt_graph: float tensor of size B * n * n, maybe none
        :param decode_type: str sample or greedy
        :returns: a dict {routes, probs, first_step_prob}, all tensor of shape ways * B * n
        '''
        B, n, _ = x_all.shape
        EB = B * self.n_encoders
        # precompute
        node_embeddings_list = []
        glimpse_keys_list = []
        glimpse_values_list = []
        prob_key = None

        for i in range(self.n_encoders):
            node_embeddings_list.append(self.encoders[i](x_all, dt_graph))
        node_embeddings = torch.cat(node_embeddings_list, 0)  # EB * n * hidden, multi-path batch parallel

        for i in range(self.decoder_layers):
            glimpse_keys_list.append(self.glimpseK[i](node_embeddings))
            glimpse_values_list.append(self.glimpseV[i](node_embeddings))
        prob_key = self.probK(node_embeddings)
        # then forward all the way! batch expand to be EB
        mask = torch.ones(EB, n).to(node_embeddings.device)
        pos_embedding = self.pos_embedding.to(node_embeddings.device).expand(EB, -1, -1)  # EB * 2 * hidden
        des_cur_embedding = torch.stack([self.des_holder.expand(EB, -1), self.cur_holder.expand(EB, -1)], 1)  # EB * 2 * hidden
        route_record = []
        probs_record = []
        first_prob = None
        for step in range(n):
            # add pos embedding at first
            des_cur_embedding = des_cur_embedding + pos_embedding
            for i in range(self.decoder_layers):
                # self attn then glimpse
                des_cur_embedding = self.self_attn_decoder_layers[i](des_cur_embedding, None)
                des_cur_embedding = MEAM.mha(des_cur_embedding, glimpse_keys_list[i], glimpse_values_list[i], mask, self.n_heads)
                des_cur_embedding = self.glimpseVOut[i](des_cur_embedding)
            # compute the probs
            prob_query = self.probQ(des_cur_embedding.sum(1, keepdim=True))  # EB * 1 * hidden, then prob_key is EB * n * hidden
            prob_u = torch.matmul(prob_query, prob_key.transpose(1, 2)).squeeze(1) / math.sqrt(self.hidden_size)  # EB * n
            prob_u = self.u_clip * prob_u.tanh()  # clip first, then mask
            masked_u = prob_u + mask.log()  # apply visited action mask first
            if step>0 and self.topk > 0:
                # topk after mask the visited, the first step never topk as kl_loss 
                topkIdx = torch.topk(masked_u, self.topk, 1)[1]  # EB * k
                topk_mask = torch.zeros(EB, n).to(node_embeddings.device)
                topk_mask[torch.arange(EB).unsqueeze(1).expand(EB, self.topk), topkIdx] = 1
                masked_u = prob_u + (mask * topk_mask).log()
            probs = masked_u.softmax(-1)
            # then decode, idx of shape EB
            if decode_type == 'sample':
                idx = torch.multinomial(probs, 1).view(-1)
            elif decode_type == 'greedy':
                idx = torch.argmax(probs, 1).view(-1)
            else:
                raise NotImplementedError('decode type {} not supported'.format(decode_type))
            route_record.append(idx)
            probs_record.append(probs[list(range(EB)), idx])
            if step == 0:
                # EB * n
                first_prob = probs
            # then update the des_cur_embedding
            des_idx, cur_idx = route_record[0], route_record[-1]
            des_embedding, cur_embedding = node_embeddings[list(range(EB)), des_idx], node_embeddings[list(range(EB)), cur_idx]
            des_cur_embedding = torch.stack([des_embedding, cur_embedding], 1)
            mask[list(range(EB)), idx] = 0
        # decode done, return dict of route, probs, first probs dist
        route_record = torch.stack(route_record, 1).reshape(self.n_encoders, B, n)
        probs_record = torch.stack(probs_record, 1).reshape(self.n_encoders, B, n)
        first_step_probs = first_prob.reshape(self.n_encoders, B, n)
        return {'routes': route_record, 'probs': probs_record, 'first_step_prob': first_step_probs}


    # training funcs
    def train_self(self, problem_size=50, batch_size=512, epochs=100, steps_per_epoch=2500, lr=1e-4, \
        kl_ratio=1e-2, max_grad_norm=1.0, eval_set_path='', train_name='default'):
        # get local args at the every start, the func args
        args_dict = locals()  
        args_dict.pop('self')    
        # get the eval loader by path
        eval_data_set = TSP.get_eval_data_set(eval_set_path, self.need_dt)
        eval_data_loader = DataLoader(eval_data_set, batch_size=batch_size, shuffle=False)
        # train setting
        train_name = Utils.get_train_name_with_time(train_name)
        save_dir = os.path.join('save', 'tsp_{}'.format(problem_size), train_name)  # save/tsp_50/default
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, 'setting.txt'), 'w') as f:
            # train setting
            f.write('train settings:\n')
            for k, v in args_dict.items():
                f.write('{}:{}\n'.format(k, v))
            # model setting
            f.write('\nmodel settings:\n')
            for k, v in self.model_setting_dict.items():
                f.write('{}:{}\n'.format(k, v))
        
        # start training
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            print('training epoch {}/{}'.format(epoch+1, epochs))
            self.cuda()
            self.train()
            train_iterator = TSP.get_train_data_iterator(steps_per_epoch, batch_size, problem_size, self.need_dt, None)
            train_iterator.reset()
            for item in tqdm(train_iterator, unit='step', desc='epoch {}'.format(epoch)):
                self.train_batch(item, kl_ratio, max_grad_norm)
            # now end of an epoch, do eval and save
            eval_cost_ways = self.greedy_eval(eval_data_loader)  # ways * N
            eval_cost = eval_cost_ways.min(0)[0]

            model_path = os.path.join(save_dir, 'actor-{}.pt'.format(epoch))
            record_path = os.path.join(save_dir, 'record.txt')
            with open(record_path, 'a') as f:
                f.write('epoch {}, validating result on {} samples: {}\n'.format(epoch, len(eval_cost), eval_cost.mean().item()))
            self.save_or_load(True, model_path)
        

    
    def train_batch(self, item: dict, kl_ratio: float, max_grad_norm: float):
        '''
        Train a batch, a forward and backward
        '''
        data = torch.from_numpy(item['data']).cuda()
        dt = torch.from_numpy(item['dt']).cuda() if self.need_dt else None
        return_dict = self(data, dt_graph=dt, decode_type='sample')
        # all of size ways * B * n
        routes, probs, first_step_prob = return_dict['routes'], return_dict['probs'], return_dict['first_step_prob']
        ways, B, n = routes.shape
        cost = TSP.compute_cost(data.expand(ways, B, n, 2).reshape(-1, n, 2), routes.reshape(-1, n)).reshape(ways, B)
        # greedy baseline
        with torch.no_grad():
            return_dict_ = self(data, dt_graph=dt, decode_type='greedy')
            routes_ = return_dict_['routes']
            cost_ = TSP.compute_cost(data.expand(ways, B, n, 2).reshape(-1, n, 2), routes_.reshape(-1, n)).reshape(ways, B)
        # loss(RL and KL loss), backward, clip grad norm, step
        rl_loss = torch.mean((cost - cost_).detach() * probs.log().sum(-1))

        kl_loss = 0
        count = 0
        for i in range(self.n_encoders):
            for j in range(self.n_encoders):
                # kl loss loop, ways
                if i == j:
                    continue
                count += 1
                first_probs_i, first_probs_j = first_step_prob[i], first_step_prob[j]  # B * n
                kl_loss = kl_loss + torch.mean(torch.sum(first_probs_i * (first_probs_i.log() - first_probs_j.log()), -1))
        if count != 0:
            kl_loss = kl_loss / count

        loss = rl_loss + kl_ratio * kl_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm, norm_type=2)
        self.optimizer.step()



    # utils function of this model
    def save_or_load(self, save: bool, path: str):
        '''
        save or load model and optimizer

        :param save: True for save, False for load
        :param path: save or load model path
        '''
        if save:
            save_dict = {
                'model': self.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            torch.save(save_dict, path)
        else:
            load_dict = torch.load(path)
            self.load_state_dict(load_dict['model'])
            self.optimizer.load_state_dict(load_dict['optimizer'])

    def greedy_eval(self, eval_data_loader: DataLoader) -> torch.Tensor:
        '''
        Eval greedy decode of this model

        :param eval_data_loader: get it by call TSP.get_eval_data_set() then warped with dataloader, see code for details

        :returns: a cost tensor of size ways * N, N is the number of instances
        '''
        self.eval()
        cost_record = []
        with torch.no_grad():
            for item in tqdm(eval_data_loader, unit='batch', desc='eval'):
                # note: this has been converted to tensor by dataloader...
                data = item['data'].cuda()
                dt = item['dt'].cuda() if self.need_dt else None
                return_dict = self(data, dt_graph=dt, decode_type='greedy')
                routes = return_dict['routes']
                ways, B, n = routes.shape

                cost = TSP.compute_cost(data.expand(ways, B, n, 2).reshape(-1, n, 2), routes.reshape(-1, n)).reshape(ways, B)
                cost_record.append(cost)
        return torch.cat(cost_record, 1)







    
    # computation funcs
    @staticmethod
    def mha(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor=None, n_heads=8):
        '''
        Simple mha computation

        :param query: tensor of size batch * nq * hidden
        :param key: tensor of size batch * nk * hidden
        :param value: tensor of size batch * nk * hidden
        :param mask: tensor of size batch * nk, 0 means masked
        :param n_heads: num of multiattn

        :returns: nq querys to nk targets, attn weight sumed value, size batch * nq * hidden
        '''
        B, nq, hidden = query.shape
        nk = key.shape[1]
        subhidden = hidden // n_heads
        assert subhidden * n_heads == hidden, 'hidden {} can not be divided by heads {}!'.format(hidden, n_heads)
        query = query.reshape(B, nq, n_heads, -1).transpose(1, 2)  # batch * heads * nq * subhidden
        key = key.reshape(B, nk, n_heads, -1).permute(0, 2, 3, 1)  # batch * heads * subhidden * nk 
        u = torch.matmul(query, key) / math.sqrt(subhidden)  # batch * heads * nq * nk
        if mask is not None:
            u = u + mask.unsqueeze(1).unsqueeze(1).log()
        attn = u.softmax(-1)  # batch * heads * nq * nk
        # then multi head attn sum
        value = value.reshape(B, nk, n_heads, -1).permute(0, 2, 1, 3)  # batch * heads * nk * hidden
        value_return = torch.matmul(attn, value)  # batch * heads * nq * hidden
        return value_return.transpose(1, 2).reshape(B, nq, hidden)

    @staticmethod
    def position_encoding_init(n_position: int, emb_dim: int) -> torch.Tensor:
        # return pos embedding of shape [n, hidden]
        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
            for pos in range(1,n_position+1)])
        

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
        return torch.from_numpy(position_enc).type(torch.FloatTensor)
        
