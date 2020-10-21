import os
import subprocess
import itertools
import time
import math


T = 100  # horizon

# folder to save
base_path = 'width_exp_T{:d}'.format(T)

if not os.path.exists(base_path):
    os.makedirs(base_path)


# experimental setup
width = [100, 500, 1000, 5000, 10000, 50000]
seeds = [0]  # list(range(3)) #random number generator seeds
dataset = ['mnist']  # 'cifar10' 'mnist'
loss = ['NLL']  # cross entropy
betas = [0.5, 0.75, 1.0]
batch_sizes = [100]
gamma0s = [1.0]  # base step-size
alphas = [0.0]


grid = itertools.product(width, seeds, dataset, loss, betas, alphas, batch_sizes, gamma0s)
# print(list(grid))


processes = []

jobs_file = open('jobs', 'w')

for w, s, d, l, beta, alpha, batch_size, gamma0 in grid:

    save_dir = base_path + '/{:04d}_{:02d}_{}_{}_{}_{}_{}_{}'.format(w, s, d, l, beta, 1.0 - alpha, batch_size, gamma0)

    # step-size
    lr = gamma0 * (w ** beta)

    # number of iterations
    num_iter = T * (w / lr) ** (1 /(1 - alpha))

    if os.path.exists(save_dir):
        # folder created only at the end when all is done!
        print('folder already exists, quitting')
        continue

    cmd = 'python3 main.py '
    cmd += '--save_dir {} '.format(save_dir)
    cmd += '--width {} '.format(w)
    cmd += '--seed {} '.format(s)
    cmd += '--dataset {} '.format(d)
    cmd += '--lr {} '.format(lr)
    cmd += '--iterations {} '.format(int(num_iter))
    cmd += '--batch_size_train {} '.format(batch_size)
    cmd += '--alpha {} '.format(alpha)

    #cmd += '--custom_init '
    #cmd += '--traj '

    log_file = save_dir + '.log'
    jobs_file.write('{:s} > {:s} 2>&1 ;\n'.format(cmd, log_file))
jobs_file.close()
