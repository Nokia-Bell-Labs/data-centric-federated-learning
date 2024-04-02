import sys
import pandas as pd
import numpy as np

import os

from numpy import loadtxt

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from plot_utils import load_all_acc

plt.rcParams.update({'font.size': 8})


res_folder = 'results'
df_acc = load_all_acc(res_folder, focus='mr')

# import pdb; pdb.set_trace()

# keep alpha=5, beta=3, gamma=0 only
df_acc = df_acc.loc[~(df_acc['clientin'] == '8')]
df_acc = df_acc.sort_values(['clientall', 'clientin','dest'])

df_acc['key'] = [d + '/\n' + e for d,e in zip(df_acc['clientin'],df_acc['clientall'])]

# bar plot
fig, ax1 = plt.subplots(figsize=(5.5,2.8))
plt.gcf().subplots_adjust(bottom=0.19, top=0.97, left=0.09, right=0.98)

barwidth = 0.25

br_acc_ap = df_acc.loc[(df_acc['dest']=='ap')]['accuracy'].tolist()
br_acc_iot = df_acc.loc[(df_acc['dest']=='iot')]['accuracy'].tolist()
br_acc_dyn = df_acc.loc[(df_acc['dest']=='dyn')]['accuracy'].tolist()

x = np.arange(len(br_acc_ap))

# calculate percentage improvement
# iot_up, ap_up = [], []
# for i in range(len(br_acc_dyn)):
#     #ap_up_ = (br_acc_dyn[i] - br_acc_ap[i]) / br_acc_ap[i]
#     #iot_up_ = (br_acc_dyn[i] - br_acc_iot[i]) / br_acc_iot[i]
#     ap_up_ = br_acc_dyn[i] - br_acc_ap[i]
#     iot_up_ = br_acc_dyn[i] - br_acc_iot[i]
#     ap_up.append(ap_up_)
#     iot_up.append(iot_up_)

# dat_ = {'dataset': df_acc['dataset'].tolist()[:len(br_acc_dyn)], 
#  'encoder': df_acc['encoder'].tolist()[:len(br_acc_dyn)],
#  'iot_up': iot_up,
#  'ap_up': ap_up
# }

# print(pd.DataFrame(data=dat_))


# plot
ax1.bar(x-barwidth, br_acc_ap, color ='b', width = barwidth, edgecolor ='black', label ='AP Training')
ax1.bar(x, br_acc_iot, color ='y', width = barwidth, edgecolor ='black', label ='UCD Training')
ax1.bar(x+barwidth, br_acc_dyn, color ='r', width = barwidth, edgecolor ='black', label ='Centaur')

ax1.set_xticks(x, df_acc['key'].unique().tolist()) #rotation=45
ax1.set_ylim(0.4, 1.00)
ax1.set_yticks(np.arange(0.4, 1.1, 0.1))
ax1.legend()

# plt.xlabel('Models and Datasets')
ax1.set_ylabel('Test Accuracy')
ax1.set_xlabel('Participating Clients out of Total Clients')

plt.savefig('imgs/clients_num.pdf')