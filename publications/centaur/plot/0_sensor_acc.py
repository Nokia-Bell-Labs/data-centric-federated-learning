import sys
sys.path.append(".")

import os
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# from plot_utils import y_fmt, load_acc_pure, load_mac_latency_energy, load_comm_latency_energy, load_res_from_cfg

plt.rcParams.update({'font.size': 9})

def load_accs(post_arg, res_folder):
    filedir = res_folder + "/" + "acc_ap" + post_arg + ".csv"
    dat_full = pd.read_csv(filedir, sep=',')

    filedir = res_folder + "/" + "acc_iot" + post_arg + ".csv"
    dat_clas = pd.read_csv(filedir, sep=',')

    filedir = res_folder + "/" + "acc_dyn" + post_arg + ".csv"
    dat_dyn = pd.read_csv(filedir, sep=',')

    acc_ap = dat_full['accuracy'].tolist()[1:]
    acc_iot = dat_clas['accuracy'].tolist()[1:]
    acc_dyn = dat_dyn['accuracy'].tolist()[1:]    
    return acc_ap, acc_iot, acc_dyn


def load_acc_one(post_arg, res_folder):
    post_arg_small = '1cifar10_2mobi_3small_c8|100_e3_a5_b3_g0'
    post_arg_medium = '1cifar10_2mobi_3medium_c8|100_e3_a5_b3_g0'
    post_arg_large = '1cifar10_2mobi_3large_c8|100_e3_a5_b3_g0'
    post_args = [post_arg_small, post_arg_medium, post_arg_large]

    acc_ap, acc_iot, acc_dyn = [], [], []
    for post_arg in post_args:
        
        filedir = res_folder + "/" + "acc_ap_" + post_arg + ".csv"
        dat_full = pd.read_csv(filedir, sep=',')
        acc_ap.append(dat_full['accuracy'].tolist()[1:])

        filedir = res_folder + "/" + "acc_iot_" + post_arg + ".csv"
        dat_clas = pd.read_csv(filedir, sep=',')
        acc_iot.append(dat_clas['accuracy'].tolist()[1:])

        filedir = res_folder + "/" + "acc_dyn_" + post_arg + ".csv"
        dat_dyn = pd.read_csv(filedir, sep=',')
        acc_dyn.append(dat_dyn['accuracy'].tolist()[1:])

    acc_ap = [(i+j+k)/3 for i,j,k in zip(acc_ap[0], acc_ap[1], acc_ap[2])]
    acc_iot = [(i+j+k)/3 for i,j,k in zip(acc_iot[0], acc_iot[1], acc_iot[2])]
    acc_dyn = [(i+j+k)/3 for i,j,k in zip(acc_dyn[0], acc_dyn[1], acc_dyn[2])]

    return acc_ap, acc_iot, acc_dyn


lebel_dyn = 'Centaur'
res_folder = 'results'
encoders = ['sencoder_p', 'sencoder_u', 'sencoder_m']
classifier = 'medium'

# buff_file = 'results/intro_plot_buff.csv'

post_args = ['_1pamap_2senc_3medium_c8|20_e3_a5_b3_g0_flpa1000_mr0|0']
post_args = post_args+ ['_1uci_har_2senc_3medium_c8|20_e3_a5_b3_g0_flpa1000_mr0|0']
post_args = post_args+ ['_1motion_2senc_3medium_c8|20_e3_a5_b3_g0_flpa1000_mr0|0']
post_args = post_args+ ['pt_1motion_2senc_3medium_c8|20_e3_a5_b3_g0_flpa1000_mr0|0']

# if not os.path.isfile(buff_file):
if True:
    # load configuration from cfg file

    for encoder, post_arg in zip(encoders, post_args):
        print(f"loading {post_arg}")
        acc_ap, acc_iot, acc_dyn = load_accs(post_arg, res_folder)
        print()


    #figs, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,3))
figs, col = plt.subplots(figsize=(5,3))
plt.subplots_adjust(left=0.14, right=0.96, bottom=0.18, top=0.95)


lw = 1
markers = ["d", "v", "s", "^", "o", "*"]
anno_text_size = 8

#####################
# col = axes[0]
x_iot = [x[i] for i in iot_idx]
y1_iot = [y1[i] for i in iot_idx]
x_dyn = [x[i] for i in dyn_idx]
y1_dyn = [y1[i] for i in dyn_idx]
index = list(range(4))
pointsize1 = 8
pointsize2 = 7

for idx in range(3):
    # p1, = col.plot(x_iot[idx], y1_iot[idx], 'y', label='TED training', marker=markers[idx], linestyle='--')
    # p2, = col.plot(x_dyn[idx], y1_dyn[idx], 'r', label=lebel_dyn, marker=markers[idx], linestyle='--')
    # if idx >= 1: 
    #     col.plot([x_iot[idx-1],x_iot[idx]], [y1_iot[idx-1],y1_iot[idx]], 'y', linestyle='--', linewidth=lw)
    #     col.plot([x_dyn[idx-1],x_dyn[idx]], [y1_dyn[idx-1],y1_dyn[idx]], 'r', linestyle='--', linewidth=lw)

    p1, = col.plot(x_iot[idx], y1_iot[idx], 'y', label='TED training', marker='v', markersize= pointsize1, linestyle='')
    p2, = col.plot(x_dyn[idx], y1_dyn[idx], 'r', label=lebel_dyn, marker='s', markersize= pointsize2, linestyle='')
    col.plot([x_iot[idx], x_dyn[idx]], [y1_iot[idx], y1_dyn[idx]], 'black', linestyle=(0, (4, 4)), linewidth=lw)
    if idx ==0:
        col.text(x_iot[idx]*0.96, y1_iot[idx]+5000, "PAMAP2", size=anno_text_size)
    elif idx == 1:
        col.text(x_iot[idx]*0.92, y1_iot[idx]+5000, "UCI_HAR", size=anno_text_size)
    else :
        col.text(x_iot[idx]*0.96, y1_iot[idx]+5000, "MotionSense", size=anno_text_size)


#### avergaged ####
# text_ = 'avg. ~10% higher accuracy'
# col.annotate('', xy=(0.925, 170000), xycoords='data',
#             xytext=(0.625, 170000), textcoords='data',
#             arrowprops=dict(arrowstyle="->, head_width=0.3", color='b', lw=2.5, ls='-'))
# col.annotate(text_, xy=(0.75, 180000), xycoords='data', ha='center', color='b', size=9)

# text_ = 'avg.\n~58%\nlower\nlatency'
# col.annotate('', xy=(0.945, 20000), xycoords='data',
#             xytext=(0.945, 260000), textcoords='data',
#             arrowprops=dict(arrowstyle="->, head_width=0.3", color='b', lw=2.5,ls='-'))
# col.annotate(text_, xy=(0.972, 180000), xycoords='data', ha='center', color='b', size=9)



# text_ = 'Better'
# col.annotate('', xy=(0.4, 0), xycoords='data',
#             xytext=(0.4, -50000), textcoords='data',
#             arrowprops=dict(arrowstyle="->, head_width=0.3", color='black', lw=2.5,ls='-'))
# col.annotate(text_, xy=(0.4, -50000), xycoords='data', ha='center', color='black', size=12)



col.set_xlabel('Test Accuracy')
col.set_xlim(0.55, 1)
col.set_xticks(np.arange(0.50, 1.01, 0.05))
col.set_ylim(0, 120000)

col.set_ylabel('Latency (Seconds)')
col.yaxis.set_major_formatter(FuncFormatter(y_fmt))

# col.legend()
col.legend((p1, p2), ('UCD Training', lebel_dyn), loc='upper left') #, , ncol=2, fontsize=8) , scatterpoints=0.1, markerscale=0.01

# col_tx = col.twinx()

# y1_iot = [y1_energy[i] for i in iot_idx]
# y1_dyn = [y1_energy[i] for i in dyn_idx]

# for idx in range(4):
#     col_tx.plot(x_iot[idx], y1_iot[idx], 'y', linestyle='')
#     col_tx.plot(x_dyn[idx], y1_dyn[idx], 'r', linestyle='')

# col_tx.set_ylabel('Energy Consumption (Joule)')
# col_tx.yaxis.set_major_formatter(FuncFormatter(y_fmt))

plt.tight_layout()

plt.savefig('imgs/intro_sum.pdf', bbox_inches='tight')


#import pdb; pdb.set_trace()


# print gain
cost_down = [(iot-dyn)/iot for iot, dyn in zip(y1_iot, y1_dyn)]
acc_up = [(dyn-iot)/iot for iot, dyn in zip(x_iot, x_dyn)]

#
print(f"encoders: {encoders}")
print(f"acc up: {acc_up}")
print(f"cost down: {cost_down}")
print(sum(cost_down)/4)
print(sum(acc_up)/4)