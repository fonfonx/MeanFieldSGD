# averaging functions for beta=1

from math import isnan
import torch
import numpy as np
import copy

def update_avg_net(net, avg_net, num_iter, burning=1000):
    n = num_iter - burning + 1
    if num_iter <= burning:
        return copy.deepcopy(net)
    else:
        with torch.no_grad():
            for (p_avg, p_new) in zip(avg_net.parameters(), net.parameters()):
                p_avg.data = (1 - 1 / n) * p_avg.data + (1 / n) * p_new.data
            return avg_net
