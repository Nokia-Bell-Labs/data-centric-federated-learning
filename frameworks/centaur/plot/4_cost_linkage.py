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

from plot_utils import y_fmt, load_mac_latency_energy, load_comm_latency_energy, load_res_from_cfg


SMALL_SIZE = 5
MEDIUM_SIZE = 9
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


encoders = ['mobilenet','mobilenet', 'efficientnet', 'efficientnet']
classifiers = ['small', 'large', 'small', 'large']
keys = ['mobile+S','mobile+L','efficient+S','efficient+L']
post_args = ['1cifar10_2mobi_3small_c8|100_e3_a5_b3_g0', '1cifar10_2mobi_3large_c8|100_e3_a5_b3_g0', '1cifar10_2effi_3small_c5|100_e3_a5_b3_g0', '1cifar10_2effi_3large_c5|100_e3_a5_b3_g0']
# load configuration from cfg file
computation_cfg, communication_cfg = load_res_from_cfg()
iot_freq, ap_freq, count_inference, iot_energy_mac, ap_energy_mac = computation_cfg
up_iot, down_iot, up_ap, down_ap, sample_size, iot_energy_comm, ap_energy_comm = communication_cfg


# plot
idxs = [1, 2]
dests = ['dyn', 'iot', 'ap']
nrows, ncols = 2, 4

figs, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(5,2.5))
plt.subplots_adjust(left=0.1, right=0.9, wspace=0.6, top=0.98, bottom=0.19)

for c in range(ncols):
    
    encoder = encoders[c]
    classifier = classifiers[c]
    post_arg = post_args[c]

    # compute mac and its energy and latency
    mac_dyn, mac_ap, mac_iot, _, _, _ = load_mac_latency_energy(encoder,classifier, post_arg,
                                                iot_freq, ap_freq, iot_energy_mac, ap_energy_mac)

    comm_dyn, comm_ap, comm_iot, _, _, _ = load_comm_latency_energy(encoder, classifier, post_arg,
                                                up_iot, down_iot, up_ap, down_ap,iot_energy_comm, ap_energy_comm, sample_size)
    
    for r in range(nrows):
        axe = axes[r,c]

        if r == 0:
            dyn_x, dyn_y1, dyn_y2 = mac_dyn[0], mac_dyn[2], mac_dyn[4]
            latency_color = 'b'
        elif r == 1:
            dyn_x, dyn_y1, dyn_y2 = mac_dyn[1], mac_dyn[3], mac_dyn[5]
            latency_color = 'g'

        axe.plot(dyn_x, dyn_y1, '-', color=latency_color, label ='Latency', linewidth=1)

        axe.yaxis.set_major_formatter(FuncFormatter(y_fmt))
        axe.xaxis.set_major_formatter(FuncFormatter(y_fmt))

        axe_tx = axe.twinx()
        axe_tx.plot(dyn_x, dyn_y2, ':', color='r', label ='Energy', linewidth=1.5)

        axe_tx.yaxis.set_major_formatter(FuncFormatter(y_fmt))
        
        if r == 1: axe.set_xlabel('MACs\n' + keys[c])
        if c == 0 and r == 0: axe.set_ylabel('Latency (UCD)', color=latency_color)
        if c == 0 and r == 1: axe.set_ylabel('Latency (AP)', color=latency_color)
        if c == 3 and r == 0: axe_tx.set_ylabel('Energy (UCD)', color='r')
        if c == 3 and r == 1: axe_tx.set_ylabel('Energy (AP)', color='r')

plt.savefig('imgs/cost_linkage.pdf')

# import pdb; pdb.set_trace()

#ap_x, ap_y1, ap_y2 = mac_ap[0], mac_ap[2], mac_ap[4]
#iot_x, iot_y1, iot_y2 = mac_iot[0], mac_iot[2], mac_iot[4]
#ap_x, ap_y1, ap_y2 = mac_ap[1], mac_ap[3], mac_ap[5]
#iot_x, iot_y1, iot_y2 = mac_iot[1], mac_iot[3], mac_iot[5]
#col.plot(ap_x, ap_y1, '-', color='b', label ='Latency', linewidth=1)
#col.plot(iot_x, iot_y1, '-', color='b', label ='Latency', linewidth=1)
#col_tx.plot(ap_x, ap_y2, ':', color='r', label ='Energy', linewidth=1.5)
#col_tx.plot(iot_x, iot_y2, ':', color='r', label ='Energy', linewidth=1.5)