'''
1.Multiencoder, and encourage starting from different cities
2.Layer decoder that glimpse unvisited then self attn cur/des
'''

import torch
import torch.nn as nn
import torch.optim as optim

import math

from model.encoder import GraphAttentionEncoder, MultiHeadAttentionLayer



class MEAM(nn.Module):
    def __init__(self, hidden_size=128, encoder_layers=3, decoder_layers=2, \
        u_clip=10, n_heads=8, n_encoders=5, topk=0):
        super().__init__()
        # model setting from init parameters
        self.hidden_size = hidden_size
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.u_clip = u_clip
        self.n_heads = n_heads
        self.n_encoders = n_encoders
        self.topk = topk  # topk self action mask, >0 valid

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
        :returns: a dict {route, probs, first_probs}
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
                des_cur_embedding = self.glimpseVOut(des_cur_embedding)
            # compute the probs
            prob_query = self.probQ(des_cur_embedding.sum(1, keepdim=True))  # EB * 1 * hidden, then prob_key is EB * n * hidden
            prob_u = torch.matmul(prob_query, prob_key.transpose(1, 2)).squeeze(1) / math.sqrt(self.hidden_size)  # EB * n
            prob_u = self.u_clip * prob_u.tanh()  # clip first, then mask
            masked_u = prob_u + mask.log()  # apply visited action mask first
            if step>0 and self.topk > 0:
                # topk after mask the visited, the first step never topk as kl_loss 
                topkIdx = torch.topk(masked_u, self.topk, 1)[1]  # EB * k
                topk_mask = torch.zeros(EB, n).to(node_embeddings.device)
                topk_mask[torch.arange(EB).expand(EB, k), topkIdx] = 1
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
        # decode done, return dict of route, probs, first probs dist
        # route: ways * B * n
        # probs: ways * B * n
        # first probs: list of len ways, each tensor of size B*n
        # to do
                

                






        


        pass

    def train(self):
        pass

    def train_epoch(self):
        '''
        Train an epoch, eval and save at last
        '''
        pass
    def train_batch(self):
        '''
        Train a batch
        '''
        pass
    
    # utils function of this model
    def save_self(self):
        '''
        save model and optimizer
        '''
        pass

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
        
