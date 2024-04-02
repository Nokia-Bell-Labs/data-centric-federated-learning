import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.cm as cm

from plot_utils import y_fmt, load_acc_pure, load_mac_latency_energy, load_comm_latency_energy, load_res_from_cfg
from plot_utils import load_all_acc
from adjustText import adjust_text

SMALL_SIZE = 7.5
MEDIUM_SIZE = 9
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

encoder = 'mobilenet'
classifier = 'medium'
res_folder = 'results'
buff_file = 'results/ds_plot_buff.csv'


if not os.path.isfile(buff_file):
    # Load accuracy
    df_acc = load_all_acc(res_folder, focus='flpa')
    # keep records with new data selection args
    df_acc = df_acc.loc[~((df_acc['alpha'] == '5') & (df_acc['beta'] == '3') & (df_acc['gamma'] == '0'))]
    df_acc['iot_latency'] = 0
    df_acc['ap_latency'] = 0
    df_acc['iot_energy'] = 0
    df_acc['ap_energy'] = 0

    # Load cost
    a = ['1.0', '3.0', '5.0', '10.0']
    b = ['10.0', '5.0', '3.0', '1.0']
    g = ['0.0', '1.0', '3.0', '5.0']

    ab = ['a'+a+'_b'+b for a,b in zip(a,b)]
    filedirs = []
    for i in range(len(ab)):
        for j in range(len(g)):
            filedir = res_folder + '/acc_dyn_1cifar10_2mobi_3medium_c8|100_e3_' + ab[i] + '_g' + g[j] + '_flpa1000.csv'
            filedirs.append(filedir)
    filedirs.append(res_folder + '/acc_ap_1cifar10_2mobi_3medium_c8|100_e3_a5.0_b3.0_g0.0_flpa1000.csv')
    filedirs.append(res_folder + '/acc_iot_1cifar10_2mobi_3medium_c8|100_e3_a5.0_b3.0_g0.0_flpa1000.csv')

    a_list = np.repeat(a, 4).tolist() + ['5.0'] * 2
    b_list = np.repeat(b, 4).tolist() + ['3.0'] * 2
    g_list = g*4 + ['0.0'] * 2

    # load configuration from cfg file
    computation_cfg, communication_cfg = load_res_from_cfg()
    iot_freq, ap_freq, count_inference, iot_energy_mac, ap_energy_mac = computation_cfg
    up_iot, down_iot, up_ap, down_ap, sample_size, iot_energy_comm, ap_energy_comm = communication_cfg

    for idx in range(len(filedirs)):
        # if os.path.isfile(filedir):
        filedir = filedirs[idx]
        alpha = a_list[idx]
        beta = b_list[idx]
        gamma = g_list[idx]
        
        # load and compute the latency and energy cost
        post_arg = filedir.replace('.csv','').replace(res_folder + '/acc_','')
        dest = post_arg[:3].replace('_','')
        post_arg = post_arg.replace('ap_','').replace('iot_','').replace('dyn_','')
        
        mac_dyn, mac_ap, mac_iot, _, _, _ = load_mac_latency_energy(encoder,classifier, post_arg, iot_freq, ap_freq, iot_energy_mac, ap_energy_mac, dests=dest)
        comm_dyn, comm_ap, comm_iot, _, _, _ = load_comm_latency_energy(encoder,classifier, post_arg, up_iot, down_iot, up_ap, down_ap, iot_energy_comm, ap_energy_comm, sample_size, dests=dest)

        sum_x = []
        for k in range(2, 6):
            if mac_dyn is not None: sum_x.append([i+j for i,j in zip(mac_dyn[k], comm_dyn[k])])
            if mac_ap is not None: sum_x.append([i+j for i,j in zip(mac_ap[k], comm_ap[k])])
            if mac_iot is not None: sum_x.append([i+j for i,j in zip(mac_iot[k], comm_iot[k])])

        # put the cost into the table
        bool = (df_acc['dest']==dest) & (df_acc['alpha']==alpha) & (df_acc['beta']==beta) & (df_acc['gamma']==gamma)
        df_acc.loc[bool, 'iot_latency'] = max(sum_x[0])
        df_acc.loc[bool, 'ap_latency'] = max(sum_x[1])
        df_acc.loc[bool, 'iot_energy'] = max(sum_x[2])
        df_acc.loc[bool, 'ap_energy'] = max(sum_x[3])

    df_acc['key'] = df_acc['alpha'] + ','+df_acc['beta'] +','+ df_acc['gamma']
    df_acc['key'] = df_acc['key'].str.replace('\.0','')

    df_acc.to_csv('results/ds_plot_buff.csv')

else:
    print('loading existing buff !!')
    df_acc = pd.read_csv(buff_file, sep=',')


# plot
figs, axes = plt.subplots(nrows=1, ncols=2, figsize=(5,3))
plt.subplots_adjust(left=0.1, right=0.9, wspace=0.35, bottom=0.15, top=0.90)

x0 = df_acc['accuracy'].tolist()
ln = len(x0)
x = x0[1:ln-1]

txts0 = df_acc['key'].tolist()
txts = txts0[1:ln-1]

#import pdb; pdb.set_trace()

for idx in range(len(axes)):
    col = axes[idx]
    if idx == 0:
        y10 = df_acc['iot_latency'].tolist()
        y20 = df_acc['iot_energy'].tolist()
    elif idx == 1:
        y10 = df_acc['ap_latency'].tolist()
        y20 = df_acc['ap_energy'].tolist()
    
    y1 = y10[1:ln-1]
    y2 = y20[1:ln-1]

    colors = cm.rainbow(np.linspace(0, 1, len(x)))
    col.scatter(x, y1, s=8, color=colors)
    
    if y10[0] > 1000: col.scatter([x0[0]], [y10[0]], s=20, color='black',marker=6)
    if y10[ln-1] > 1000: col.scatter([x0[ln-1]], [y10[ln-1]], s=20, color='black',marker=7)

    texts = [col.text(x[i], y1[i], txts[i], size=6) for i in range(len(x))] # ha='center', va='center'  
    adjust_text(texts, ax=col)
    # adjust_text(texts, ax=col, arrowprops=dict(arrowstyle="-", color='k', lw=0.5), avoid_text=True) # avoid_self=True, 
    col.invert_xaxis()
    col.set_xlim(0.93, 0.8)
    #col.set_xticks(np.arange(0.92, 0.80, 0.02))
    col.yaxis.set_major_formatter(FuncFormatter(y_fmt))
    col.set_xlabel('Test Accuracy')
    if idx==0: col.set_title('Workload on TEDs')
    if idx==0: col.set_ylabel('Accumulated Latency (Seconds)')

    col_tx = col.twinx()
    col_tx.scatter(x, y2, s=0.01, color='white', marker='')
    col_tx.set_xlim(0.93, 0.8)
    #col.set_xticks(np.arange(0.92, 0.80, 0.02))
    col_tx.yaxis.set_major_formatter(FuncFormatter(y_fmt))
    col_tx.set_xlabel('Test Accuracy')
    if idx==1: col_tx.set_title('Workload on APs')
    if idx==1: col_tx.set_ylabel('Accumulated Energy (Joule)')

    
plt.savefig('imgs/data_selection_args.pdf')


