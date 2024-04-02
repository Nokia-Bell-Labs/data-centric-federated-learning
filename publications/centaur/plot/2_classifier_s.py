import sys
import pandas as pd
import numpy as np
sys.path.append(".")

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


def load_all_acc_s(res_folder, focus='basic'):
    onlyfiles = [f for f in listdir(res_folder) if isfile(join(res_folder, f))]
    acc_ap = [filename for filename in onlyfiles if 'acc_ap_1' in filename]
    acc_iot = [filename for filename in onlyfiles if 'acc_iot_1' in filename]
    acc_dyn = [filename for filename in onlyfiles if 'acc_dyn_1' in filename]
    acc_x = acc_ap + acc_iot + acc_dyn

    splitted_list = []
    for acc_name in acc_x:
        # marking based on the filename
        acc_name = acc_name.replace('uci_har', 'ucihar')
        splitted = acc_name.split('_')        
        if 'flpa1000' not in splitted:
            continue        
        print(splitted)
        splitted = [splitted[1], splitted[2][1:], splitted[3][1:], splitted[4][1:]]
        name_base = ['dest','dataset','encoder','classifier']
        
        # read highest accuracy
        acc_name = acc_name.replace('ucihar', 'uci_har')
        acc_temp = pd.read_csv(res_folder + '/' + acc_name)
        
        acc_list = acc_temp['accuracy'].tolist()
        acc_list.sort(reverse=True)
        max_acc = sum(acc_list[:5])/5
        
        # loc this accuracy
        round_idx = int(acc_temp.loc[acc_temp['accuracy'] == acc_list[0], 'round'].tolist()[0])
        splitted = splitted + [max_acc] + [round_idx]
        splitted_list.append(splitted)

    
    columns_name = name_base + ['accuracy','loc']
    df_acc = pd.DataFrame(splitted_list, columns=columns_name)
    df_acc = df_acc.sort_values(columns_name)
    
    # df_acc.loc[df_acc['encoder'] == 'effi','encoder'] = 'efficient'
    # df_acc.loc[df_acc['encoder'] == 'mobi','encoder'] = 'mobile'
    # df_acc.loc[df_acc['encoder'] == 'shuf','encoder'] = 'shuffle'
    df_acc.loc[df_acc['dataset'] == 'motion','dataset'] = 'MotionSense'
    df_acc.loc[df_acc['dataset'] == 'ucihar','dataset'] = 'UCIHAR'
    df_acc.loc[df_acc['dataset'] == 'pamap','dataset'] = 'PAMAP2'
    
    df_acc.loc[df_acc['classifier'] == 'small','classifier'] = 'S'
    df_acc.loc[df_acc['classifier'] == 'medium','classifier'] = 'M'
    df_acc.loc[df_acc['classifier'] == 'large','classifier'] = 'L'

    return df_acc



df_acc = load_all_acc_s(res_folder)
# df_acc = df_acc.loc[~(df_acc['dataset'] == 'motion')]

# filter out low accuracy
# df_acc = df_acc.loc[df_acc['accuracy'] > 0.3]

df_acc['key'] = [e for e in df_acc['dataset']]
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
    
    markers = ["d", "v", "o", "*", "^", "o"]

    for idx,col in zip(idxs,axes):
        classifier_acc = df_acc.loc[df_acc['classifier'] == classifier_order[idx]]
        for count, key in enumerate(key_list):
            dat = classifier_acc[classifier_acc['key'] == key]
            x = dat['dest'].tolist()
            y = dat['accuracy'].tolist()
            col.plot(x, y, label=key, marker=markers[count], markersize=7, linewidth=0.4, linestyle=(0, (1, 10)))

        if idx==0: col.set_ylabel('Test Accuracy')
        col.set_xlabel('Training Methods')
        col.set_ylim(0.5, 1.01)
        col.set_yticks(np.arange(0.5, 1.01, 0.1))
        col.title.set_text(classifier_str[idx])


    handles, labels = col.get_legend_handles_labels()
    figs.legend(handles, labels, loc='center right') 

    plt.savefig('imgs/classifier_acc.pdf', bbox_inches='tight')