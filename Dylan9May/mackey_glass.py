#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 20:08:19 2018

@author: DylanGrey
"""
from file_handle import *
import collections
import numpy as np
import matplotlib.pyplot as plt

def mackey_glass(sample_len=1000, tau=17, seed=None, n_samples = 1):
    '''
    mackey_glass(sample_len=1000, tau=17, seed = None, n_samples = 1) -> input
    Generate the Mackey Glass time-series. Parameters are:
        - sample_len: length of the time-series in timesteps. Default is 1000.
        - tau: delay of the MG - system. Commonly used values are tau=17 (mild 
          chaos) and tau=30 (moderate chaos). Default is 17.
        - seed: to seed the random generator, can be used to generate the same
          timeseries at each invocation.
        - n_samples : number of samples to generate
    '''
    delta_t = 10
    history_len = tau * delta_t 
    # Initial conditions for the history of the system
    timeseries = 1.2
    
    if seed is not None:
        np.random.seed(seed)

    samples = []

    for _ in range(n_samples):
        history = collections.deque(1.2 * np.ones(history_len) + 0.2 * \
                                    (np.random.rand(history_len) - 0.5))
        # Preallocate the array for the time-series
        inp = np.zeros((sample_len,1))
        
        for timestep in range(sample_len):
            for _ in range(delta_t):
                xtau = history.popleft()
                history.append(timeseries)
                timeseries = history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - \
                             0.1 * history[-1]) / delta_t
            inp[timestep] = timeseries
        
        # Squash timeseries through tanh
        inp = np.tanh(inp - 1)
        samples.append(inp)
    return samples

r = mackey_glass(seed=4)[0];

plt.plot(r)
plt.show()

#now do processing on it to make it 128 long

one_hots = []
one_hot_idxs = []
maximum = np.amax(r)
minimum = np.amin(r)
for t in r:
    tmp_one_hot = np.zeros(128)
    one_hot_idx = int(round((t[0] - minimum)/(maximum-minimum) * 127))
    tmp_one_hot[ one_hot_idx ] = 1
    one_hots.append(tmp_one_hot)
    one_hot_idxs.append(one_hot_idx)
    
plt.plot(one_hot_idxs)
plt.show()
    
save_var("~/Desktop/mackey_glass_1000.hex", one_hots)


tst = load_var("~/Desktop/mackey_glass_1000.hex")