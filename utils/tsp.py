import torch
from torch.utils.data import Dataset

import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

import os
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Lock, Pipe

class TSP:
    @staticmethod
    def compute_cost(data: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        '''
        Compute the cost of a batch of tsp routes

        :param data: a batch of tsp instances, shape batch*n*2
        :param pi: a batch of routes, shape batch*n
        :returns: a batch of costs, shape batch
        '''
        assert (
            torch.arange(pi.size(1), out=pi.new()).type_as(pi).view(1, -1).expand_as(pi) ==
            pi.sort(dim=1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        d = data.gather(1, pi.unsqueeze(-1).expand_as(data))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1)

    @staticmethod
    def get_train_data_iterator(steps: int, batch_size: int, problem_size: int, need_dt: bool, data_path: str):
        '''
        Return a TspDataIterator for training, check the param is valid or not
        '''
        assert type(steps) == int and steps > 0
        assert type(batch_size) == int and batch_size > 0
        assert type(problem_size) == int and problem_size > 0
        assert type(need_dt) == bool
        if data_path is not None:
            assert type(data_path) == str, 'data_path should be str'
            assert data_path.endswith('.npz'), 'data file should be .npz'
            assert os.path.exists(data_path), 'data path {} not exists!'.format(data_path)
        return TspDataIterator(steps, batch_size, problem_size, need_dt, data_path)
    @staticmethod
    def get_eval_data_set(data_path: str, need_dt: bool):
        '''
        Return a pytorch dataset from file 
        '''
        data = np.load(data_path)['x'].astype(np.float32)  # open file at first
        item = {'data': data}
        if need_dt:
            item['dt'] = dt_graph(data)
        return TSPDataset(item)
        

class TspDataIterator:
    '''
    Tsp data iterator for training, data is generated on the fly or sample from file with replacement
    '''
    def __init__(self, steps: int, batch_size: int, problem_size: int, need_dt: bool, data_path: str):
        self.steps = steps
        self.batch_size = batch_size
        self.problem_size = problem_size
        self.need_dt = need_dt
        self.data_path = data_path

        # params about iteration
        self.i = steps
        self.main_pipe = None
        self.sub_pipe = None
        self.subp = None
        self.data = None
    
    def reset(self):
        '''
        call before the iteration start
        '''
        self.i = 0
        if self.need_dt:
            # item is from subprocess
            self.main_pipe, self.sub_pipe = Pipe(duplex=False)
            self.subp = mp.Process(target=data_generator, args=(self.sub_pipe, self.data_path, self.steps, \
                self.batch_size, self.problem_size))
            self.subp.daemon = True
            self.subp.start()
        else:
            # item is from self, maybe from file
            if self.data_path is not None:
                self.data = np.load(filename)['x'].astype(np.float32)
        


    def __next__(self) -> dict:
        '''
        Return a dict, keys may contains ['data', 'dt'], or to be extended afterwards
        '''
        if self.i < self.steps:
            self.i += 1
            item = None
            if self.need_dt:
                # from subprocess
                item = self.main_pipe.recv()
            else:
                # no dt need, in this process
                if self.data_path is not None:
                    # from file
                    item = {'data': self.data[np.random.randint(0, len(self.data), (self.batch_size,))]}
                else:
                    # on the fly
                    item = {'data': np.random.rand(self.batch_size, self.problem_size, 2).astype(np.float32)}
            return item

        else:
            if self.need_dt:
                self.main_pipe.close()
                self.sub_pipe.close()
                self.subp.join()
            raise StopIteration()

    def __len__(self):
        '''
        Return len for tqdm
        '''
        return steps

class TSPDataset(Dataset):
    def __init__(self, item: dict):
        '''
        Init a dataset

        :param data: a dict contains 'data'->np.array, may contains dt graphs
        '''
        super().__init__()
        self.item = item

        self.len = len(self.item['data'])
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        ret = {}
        for k in self.item.keys():
            ret[k] = self.item[k][idx]
        return ret





'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~cross line~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
def data_generator(sub_pipe: Pipe, data_path: str, steps: int, batch_size: int, problem_size: int):
    '''
    For subprocess to dt computation, data can be from file or on the fly
    '''
    if data_path is not None:
        # open from file, then sample with replacement
        # data of shape: N * n * 2
        data = np.load(filename)['x'].astype(np.float32)
        for i in range(steps):
            idx = np.random.randint(0, len(data), (batch_size,))
            batch_data = data[idx]
            batch_dt_graph = dt_graph(batch_data)
            item = {'data': batch_data, 'dt': batch_dt_graph}
            sub_pipe.send(item)
    else:
        # uniform on the fly
        for i in range(steps):
            batch_data = np.random.rand(batch_size, problem_size, 2).astype(np.float32)
            batch_dt_graph = dt_graph(batch_data)
            item = {'data': batch_data, 'dt': batch_dt_graph}
            sub_pipe.send(item)


def dt_graph(data: np.array) -> np.array:
    '''
    Compute DT for a batch of tsp instances

    :param data: a batch of tsp instances, numpy array of float32, batch * n * 2
    :returns: DT for input, numpy array of float32, batch * n * n
    '''
    B, n, _ = data.shape
    graph_list = []
    for i in range(B):
        points = data[i]
        dt = Delaunay(points)
        tri = dt.simplices  # np array of size m * 3
        # construct the DT graph
        edges = np.zeros((n, n), dtype=np.int8)
        edges[tri[:, 0], tri[:, 1]] = 1
        edges[tri[:, 1], tri[:, 0]] = 1
        edges[tri[:, 0], tri[:, 2]] = 1
        edges[tri[:, 2], tri[:, 0]] = 1
        edges[tri[:, 1], tri[:, 2]] = 1
        edges[tri[:, 2], tri[:, 1]] = 1
        graph_list.append(edges)
    graphs = np.stack(graph_list, axis=0) 
    return graphs.astype(np.float32)


