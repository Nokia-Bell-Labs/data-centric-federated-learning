import os
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from plot_utils import y_fmt, load_acc_pure, load_mac_latency_energy, load_comm_latency_energy, load_res_from_cfg

plt.rcParams.update({'font.size': 9})


lebel_dyn = 'Centaur'
res_folder = 'results'
encoders = ['efficientnet', 'mobilenet', 'shufflenet', 'mnasnet']
classifier = 'medium'

buff_file = 'results/intro_plot_buff.csv'

post_args = ['1cifar10_2' + e[:4] +  '_3medium_c8|100_e3_a5_b3_g0' for e in encoders[1:]]
post_args = ['1cifar10_2effi_3medium_c5|100_e3_a5_b3_g0'] + post_args


if not os.path.isfile(buff_file):
    # load configuration from cfg file
    computation_cfg, communication_cfg = load_res_from_cfg()
    iot_freq, ap_freq, count_inference, iot_energy_mac, ap_energy_mac = computation_cfg
    up_iot, down_iot, up_ap, down_ap, sample_size, iot_energy_comm, ap_energy_comm = communication_cfg


    key_e, key_d, x, y1, y1_energy, y2 = [], [], [], [], [], []
    for encoder, post_arg in zip(encoders, post_args):
        print(f"loading {post_arg}")
        acc_ap, acc_iot, acc_dyn = load_acc_pure(post_arg, res_folder)

        mac_dyn, mac_ap, mac_iot, _, _, _ = load_mac_latency_energy(encoder,classifier, post_arg, iot_freq, ap_freq, iot_energy_mac, ap_energy_mac)
        comm_dyn, comm_ap, comm_iot, _, _, _ = load_comm_latency_energy(encoder,classifier, post_arg, up_iot, down_iot, up_ap, down_ap, iot_energy_comm, ap_energy_comm, sample_size)

        sum_dyn, sum_ap, sum_iot = [], [], []
        for k in range(2, len(mac_dyn)):
            sum_dyn.append([i+j for i,j in zip(mac_dyn[k], comm_dyn[k])])
            sum_ap.append([i+j for i,j in zip(mac_ap[k], comm_ap[k])])
            sum_iot.append([i+j for i,j in zip(mac_iot[k], comm_iot[k])])

        key_e = key_e + [encoder] * 3
        key_d = key_d + ['ap', 'iot', 'dyn']
        x = x + [max(acc_ap), max(acc_iot), max(acc_dyn)]
        y1 = y1 + [max(sum_ap[0]), max(sum_iot[0]), max(sum_dyn[0])]
        y1_energy = y1_energy + [max(sum_ap[2]), max(sum_iot[2]), max(sum_dyn[2])]
        y2 = y2 + [max(sum_ap[1]), max(sum_iot[1]), max(sum_dyn[1])]

    # buffer
    dat_ = {'key_e': key_e,
            'key_d': key_d,
            'x': x,
            'y1': y1,
            'y1_energy': y1_energy,
            'y2': y2
            }
    df_acc = pd.DataFrame(data=dat_)
    df_acc.to_csv(buff_file)

else:
    print('loading existing buff !')
    df_acc = pd.read_csv(buff_file, sep=',')

# import pdb; pdb.set_trace()



key_e = df_acc['key_e']
key_d = df_acc['key_d']
x = df_acc['x']
y1 = df_acc['y1']
y1_energy = df_acc['y1_energy']
y2 = df_acc['y2']

ap_idx = [i for i, j in enumerate(key_d) if j == 'ap']
iot_idx = [i for i, j in enumerate(key_d) if j == 'iot']
dyn_idx = [i for i, j in enumerate(key_d) if j == 'dyn']

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

for idx in range(4):
    # p1, = col.plot(x_iot[idx], y1_iot[idx], 'y', label='TED training', marker=markers[idx], linestyle='--')
    # p2, = col.plot(x_dyn[idx], y1_dyn[idx], 'r', label=lebel_dyn, marker=markers[idx], linestyle='--')
    # if idx >= 1: 
    #     col.plot([x_iot[idx-1],x_iot[idx]], [y1_iot[idx-1],y1_iot[idx]], 'y', linestyle='--', linewidth=lw)
    #     col.plot([x_dyn[idx-1],x_dyn[idx]], [y1_dyn[idx-1],y1_dyn[idx]], 'r', linestyle='--', linewidth=lw)

    p1, = col.plot(x_iot[idx], y1_iot[idx], 'y', label='TED training', marker='v', markersize= pointsize1, linestyle='')
    p2, = col.plot(x_dyn[idx], y1_dyn[idx], 'r', label=lebel_dyn, marker='s', markersize= pointsize2, linestyle='')
    col.plot([x_iot[idx], x_dyn[idx]], [y1_iot[idx], y1_dyn[idx]], 'black', linestyle=(0, (4, 4)), linewidth=lw)

    col.text(x_iot[idx]*0.96, y1_iot[idx]+15000, encoders[idx], size=anno_text_size)


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


#### single ####
text_ = '16.5% higher accuracy'
col.annotate('', xy=(0.925, 100000), xycoords='data',
            xytext=(0.785, 100000), textcoords='data',
            arrowprops=dict(arrowstyle="->, head_width=0.3", color='b', lw=2.5, ls='-'))
col.annotate(text_, xy=(0.825, 110000), xycoords='data', ha='center', color='b', size=9)

text_ = '59.8%\nlower\nlatency'
col.annotate('', xy=(0.93, 105000), xycoords='data',
            xytext=(0.93, 260000), textcoords='data',
            arrowprops=dict(arrowstyle="->, head_width=0.3", color='b', lw=2.5,ls='-'))
col.annotate(text_, xy=(0.96, 180000), xycoords='data', ha='center', color='b', size=9)



# text_ = 'Better'
# col.annotate('', xy=(0.4, 0), xycoords='data',
#             xytext=(0.4, -50000), textcoords='data',
#             arrowprops=dict(arrowstyle="->, head_width=0.3", color='black', lw=2.5,ls='-'))
# col.annotate(text_, xy=(0.4, -50000), xycoords='data', ha='center', color='black', size=12)



col.set_xlabel('Test Accuracy')
col.set_xlim(0.55, 1)
col.set_xticks(np.arange(0.55, 1.01, 0.05))
col.set_ylim(0, 300000)

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

plt.savefig('imgs/intro_sum.pdf')


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