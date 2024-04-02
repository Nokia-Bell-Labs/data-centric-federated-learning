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

from plot_utils import load_all_acc

plt.rcParams.update({'font.size': 8})


classifier_focus = 0
training_method_focus = 1

res_folder = 'results'
df_acc = load_all_acc(res_folder)
df_acc = df_acc.loc[~(df_acc['dataset'] == 'emnist')]

# filter out low accuracy
df_acc = df_acc.loc[df_acc['accuracy'] > 0.3]

df_acc['key'] = [d + '\n' + e for d,e in zip(df_acc['encoder'],df_acc['dataset'])]
key_list = df_acc['key'].unique().tolist()

classifier_order = ['S','M','L']

idxs = list(range(3))


if classifier_focus == 1:
    df_acc['classifier_index'] = df_acc['classifier']
    df_acc = df_acc.set_index('classifier_index').loc[classifier_order]

    dests = ['ap', 'iot', 'dyn']
    dests_str = ['AP Training', 'UCD Training', 'Centaur']

    # plot all
    figs, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(5.5,2.6))
    plt.subplots_adjust(wspace=0.3, right=0.78, bottom=0.15)
    # plt.subplots_adjust(left=0.1, , wspace=0.9, )

    for idx,col in zip(idxs,axes):
        dest_acc = df_acc.loc[df_acc['dest'] == dests[idx]]
        for key in key_list:
            dat = dest_acc[dest_acc['key'] == key]
            x = dat['classifier'].tolist()
            y = dat['accuracy'].tolist()
            col.plot(x, y, label=key, marker=".", markersize=10, linewidth=0.5)

        if idx==0: col.set_ylabel('Test Accuracy')
        col.set_xlabel('Classifier Size')
        col.set_ylim(0.3, 1)
        col.set_yticks(np.arange(0.3, 1, 0.1))
        col.title.set_text(dests_str[idx])

    #import pdb; pdb.set_trace()

    handles, labels = col.get_legend_handles_labels()
    figs.legend(handles, labels, loc='center right') 

    plt.savefig('imgs/classifier_acc.pdf')



if training_method_focus == 1:
    
    d_str = ['AP', 'UCD', 'Centaur']
    df_acc.loc[df_acc['dest'] == 'ap', 'dest'] = d_str[0]
    df_acc.loc[df_acc['dest'] == 'iot', 'dest'] = d_str[1]
    df_acc.loc[df_acc['dest'] == 'dyn', 'dest'] = d_str[2]

    df_acc['dest_index'] = df_acc['dest']
    df_acc = df_acc.set_index('dest_index').loc[d_str]

    classifier_str = ['Small', 'Medium', 'Large']

    # plot all
    figs, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(5.5,2.6))
    plt.subplots_adjust(wspace=0.3, right=0.78, bottom=0.15)
    # plt.subplots_adjust(left=0.1, , wspace=0.9, )
    
    markers = ["d", "v", "s", "*", "^", "o"]

    for idx,col in zip(idxs,axes):
        classifier_acc = df_acc.loc[df_acc['classifier'] == classifier_order[idx]]
        for count, key in enumerate(key_list):
            dat = classifier_acc[classifier_acc['key'] == key]
            x = dat['dest'].tolist()
            y = dat['accuracy'].tolist()
            col.plot(x, y, label=key, marker=markers[count], markersize=7, linewidth=0.4, linestyle=(0, (1, 10)))

        if idx==0: col.set_ylabel('Test Accuracy')
        col.set_xlabel('Training Methods')
        col.set_ylim(0.3, 1)
        col.set_yticks(np.arange(0.3, 1, 0.1))
        col.title.set_text(classifier_str[idx])


    handles, labels = col.get_legend_handles_labels()
    figs.legend(handles, labels, loc='center right') 
    
    plt.savefig('imgs/classifier_acc.pdf')