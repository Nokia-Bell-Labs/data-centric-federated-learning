import sys
import pandas as pd
import numpy as np

import os
from os import listdir
from os.path import isfile, join

from numpy import loadtxt

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from plot_utils import running_avg

plt.rcParams.update({'font.size': 8})

dest = ['ap','iot','dyn']
dest_str = ['AP Training','UCD Training','Centaur']
flpa = ['0.001', '0.1', '1.0', '1000.0']#, '10.0', '100.0', ] '0.01', 


clas_hist = ['500,0,0,0,0,0,0,0,0,0', '251,230,13,4,2,0,0,0,0,0', '125,89,87,47,44,39,34,29,3,3', '57,53,53,52,51,50,49,48,47,40']
clas_hist = ['Hist: ' + c for c in clas_hist]



rounds = list(range(100))

all_dat = []

for f in flpa:
    dat = []
    for d in dest:
        filedir = 'results/' + 'acc_' + d + '_1cifar10_2mobi_3medium_c8|100_e3_a5_b3_g0_flpa' + f + '.csv'
        dat_x = pd.read_csv(filedir, sep=',')
        dat_x = dat_x['accuracy'].tolist()[1:]
        dat_x = running_avg(dat_x)
        dat.append(dat_x)

    all_dat.append(dat)

# import pdb; pdb.set_trace()

# plot
figs, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(5,4))
plt.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.1)

idx = 0
for row in axes:
    for col in row:
        dat = all_dat[idx]
        col.plot(rounds, dat[0], '-', color='b', label=dest_str[0], linewidth=1)
        col.plot(rounds, dat[1], '-', color='y', label=dest_str[1], linewidth=1)
        col.plot(rounds, dat[2], '-', color='r',label=dest_str[2], linewidth=1)
        col.annotate(clas_hist[idx],  xy=(0, 0.9), xycoords='data',size=6) #, textcoords='offset points'
        
        col.set_yticks(np.arange(0.1, 1, 0.1))
        col.set_ylim(0.1, 0.95)
        col.set_title('LDA-Alpha: ' + flpa[idx], loc='left')
        
        if idx == 2 or idx == 3: col.set_xlabel('Rounds')
        if idx == 0 or idx == 2: col.set_ylabel('Test Accuracy')
        idx += 1

col.legend()
        
plt.savefig('imgs/unbalanced.pdf')