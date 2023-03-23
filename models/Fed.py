#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import random
import math
import torch
from torch import nn
import numpy as np

def FedAvg(w, args):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w)).to(args.device)
    return w_avg
