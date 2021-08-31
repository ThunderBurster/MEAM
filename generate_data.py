import numpy as np
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem_size', type=int, default=None, help='problem size of tsp')
    parser.add_argument('--val_size', type=int, default=None, help='number of instances')
    parser.add_argument('--seed', type=int, default=None, help='the seed of np rng')
    opts = parser.parse_args()
    opts = vars(opts)
    for k, v in opts.items():
        assert v is not None, 'please give {}'.format(k)

    problem_size = opts['problem_size']
    val_size = opts['val_size']
    seed = opts['seed']

    rng = np.random.RandomState(seed)
    x = rng.rand(val_size, problem_size, 2)

    if not os.path.exists('data'):
        os.makedirs('data')
    save_path = 'tsp{}_size{}_seed{}.npz'.format(problem_size, val_size, seed)
    save_path = os.path.join('data', save_path)
    np.savez(save_path, x=x)




