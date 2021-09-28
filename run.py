import argparse
import time

from model import MEAM, MEAM_AMDECODER
from utils import TSP

import torch.optim as optim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model settings
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size of model')
    parser.add_argument('--encoder_layers', type=int, default=3, help='encoder layers of model')
    parser.add_argument('--decoder_layers', type=int, default=1, help='decoder layers of model')
    parser.add_argument('--u0_clip', type=float, default=10.0, help='u clip for first step')
    parser.add_argument('--u_clip', type=float, default=10.0, help='u clip')
    parser.add_argument('--n_heads', type=int, default=8, help='heads for multi-head')
    parser.add_argument('--n_encoders', type=int, default=5, help='number of encoders')
    parser.add_argument('--topk', type=int, default=5, help='topk action mask')
    parser.add_argument('--need_dt', type=bool, default=True, help='use dt or not')
    parser.add_argument('--am_decoder', type=bool, default=False, help='use am deocder')
    # training settings
    parser.add_argument('--problem_size', type=int, default=20, help='problem size of tsp')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='how many epochs to run')
    parser.add_argument('--steps_per_epoch', type=int, default=2500, help='num of steps per epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='lr for training')
    parser.add_argument('--kl_ratio', type=float, default=1e-2, help='kl loss ratio')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='max grad norm')
    parser.add_argument('--eval_set_path', type=str, default=None, help='eval set path')
    parser.add_argument('--train_name', type=str, default='default', help='name for this run')
    parser.add_argument('--load_path', type=str, default=None, help='load path')

    # eval only
    parser.add_argument('--eval_only', action='store_true', help='Set this value to only evaluate model')
    parser.add_argument('--eval_batch_size', type=int, default=1024, help='eval batch size')

    # get ready for training
    opts = parser.parse_args()
    opts = vars(opts)

    # model settings
    hidden_size = opts['hidden_size']
    encoder_layers = opts['encoder_layers']
    decoder_layers = opts['decoder_layers']
    u0_clip = opts['u0_clip']
    u_clip = opts['u_clip']
    n_heads = opts['n_heads']
    n_encoders = opts['n_encoders']
    topk = opts['topk']
    need_dt = opts['need_dt']
    am_decoder = opts['am_decoder']

    # training settings
    problem_size = opts['problem_size']
    batch_size = opts['batch_size']
    epochs = opts['epochs']
    steps_per_epoch = opts['steps_per_epoch']
    lr = opts['lr']
    kl_ratio = opts['kl_ratio']
    max_grad_norm = opts['max_grad_norm']
    eval_set_path = opts['eval_set_path']
    train_name = opts['train_name']
    load_path = opts['load_path']

    # eval only
    eval_only = opts['eval_only']
    eval_batch_size = opts['eval_batch_size']

        
    # init model, 2 types(self attn decoder & attention model decoder)
    if am_decoder:
        model = MEAM_AMDECODER(hidden_size, encoder_layers, decoder_layers, u0_clip, u_clip, n_heads, n_encoders, topk, need_dt)
    else:
        model = MEAM(hidden_size, encoder_layers, decoder_layers, u0_clip, u_clip, n_heads, n_encoders, topk, need_dt)
    
    # default eval set path, seed is equal to problem size
    if eval_set_path is None:
        eval_set_path = 'data/tsp{}_size10000_seed{}.npz'.format(problem_size, problem_size)

    # warm start
    if load_path is not None:
        model.optimizer = optim.Adam(model.parameters(), lr=lr)
        model.save_or_load(False, load_path)


    # training or eval
    if eval_only:
        # eval set ready
        eval_data_set = TSP.get_eval_data_set(eval_set_path, model.need_dt)
        eval_data_loader = DataLoader(eval_data_set, batch_size=eval_batch_size, shuffle=False)
        model.cuda()
        model.eval()

        # greedy eval
        start = time.perf_counter()
        costs = model.greedy_eval(eval_data_loader)
        end = time.perf_counter()

        # print result
        eval_cost = costs.min(0)[0].mean().item()
        print('take {} s'.format(end - start))
        print('eval result {}'.format(eval_cost))
    else:
        model.train_self(problem_size, batch_size, epochs, steps_per_epoch, lr, kl_ratio, \
            max_grad_norm, eval_set_path, train_name)
    
    
    
    