import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from plot_utils import y_fmt, load_acc_one, load_mac_latency_energy, load_comm_latency_energy, load_res_from_cfg

plt.rcParams.update({'font.size': 8})

parser = argparse.ArgumentParser(description="Plot")
parser.add_argument("--mac", type=bool, default=True)
parser.add_argument("--comm", type=bool, default=True)
parser.add_argument("--sum", type=bool, default=False)

args = parser.parse_args()

lebel_pbfl = 'Centaur'

res_folder = 'results'
encoder = 'mobilenet'
classifier = 'medium'
post_arg = '1cifar10_2mobi_3medium_c8|100_e3_a5_b3_g0'


def plot_12_3y(plot_dat_dyn, plot_dat_ap, plot_dat_iot, x_dyn, x_ap, x_iot, title_text, x_text, y_text, xlim_vec, ylim_vec, filename, xinverse=False):
    figs, axes = plt.subplots(nrows=1, ncols=2, figsize=(9,2.5))
    plt.subplots_adjust(left=0.1, right=0.85, wspace=1, bottom=0.15)
    pi = 0

    min_value = min(x_dyn + x_ap + x_iot)
    min_value = round(min_value, 1)
    max_value = max(x_dyn + x_ap + x_iot)
    
    lebel_ap = 'AP Training'
    lebel_iot = 'UCD training'
    lebel_dyn = lebel_pbfl
    linewidth = 1

    for col in axes:
        if not max(plot_dat_ap[pi]) == 0:
            col.plot(x_ap, plot_dat_ap[pi], 'g', label=lebel_ap, linewidth=linewidth)
        if not max(plot_dat_iot[pi]) == 0:
            col.plot(x_iot, plot_dat_iot[pi], 'y', label=lebel_iot, linewidth=linewidth)
        if not max(plot_dat_dyn[pi]) == 0:
            col.plot(x_dyn, plot_dat_dyn[pi], 'r', label=lebel_dyn, linewidth=linewidth)
        col.set_title(title_text[pi])
        col.set_xlabel(x_text[pi])
        col.set_ylabel(y_text[pi])


        if xinverse == True: col.invert_xaxis()
        if xlim_vec is not None: col.set_xlim(xlim_vec[0], xlim_vec[1])
        if ylim_vec is not None: col.set_ylim(ylim_vec[0], ylim_vec[1])
        col.yaxis.set_major_formatter(FuncFormatter(y_fmt))

        col.legend()

        col_tx = col.twinx()
        col_tx2 = col.twinx()

        if not max(plot_dat_ap[pi+2]) == 0:
            col_tx.plot(x_ap, plot_dat_ap[pi+2], 'g', label=lebel_ap, linewidth=linewidth, linestyle='')
        if not max(plot_dat_iot[pi+2]) == 0:
            col_tx.plot(x_iot, plot_dat_iot[pi+2], 'y', label=lebel_iot, linewidth=linewidth, linestyle='')
        if not max(plot_dat_dyn[pi+2]) == 0:
            col_tx.plot(x_dyn, plot_dat_dyn[pi+2], 'r', label=lebel_dyn, linewidth=linewidth, linestyle='')
        col_tx.set_ylabel(y_text[pi+2])
        col_tx.yaxis.set_major_formatter(FuncFormatter(y_fmt))
        
        if not max(plot_dat_ap[pi+4]) == 0:
            col_tx2.plot(x_ap, plot_dat_ap[pi+4], 'g', label=lebel_ap, linewidth=linewidth, linestyle='')
        if not max(plot_dat_iot[pi+4]) == 0:
            col_tx2.plot(x_iot, plot_dat_iot[pi+4], 'y', label=lebel_iot, linewidth=linewidth, linestyle='')
        if not max(plot_dat_dyn[pi+4]) == 0:
            col_tx2.plot(x_dyn, plot_dat_dyn[pi+4], 'r', label=lebel_dyn, linewidth=linewidth, linestyle='')
        col_tx2.set_ylabel(y_text[pi+4])
        col_tx2.yaxis.set_major_formatter(FuncFormatter(y_fmt))

        col_tx2.spines['right'].set_position(('outward', 45))

        pi += 1

    plt.savefig('imgs/' + filename + '.pdf')


def plot_12_2y(plot_dat_dyn, plot_dat_ap, plot_dat_iot, x_dyn, x_ap, x_iot, x_text, y_text, xlim_vec, ylim_vec, filename, xinverse=False):
    figs, axes = plt.subplots(nrows=1, ncols=2, figsize=(5,2.5))
    plt.subplots_adjust(left=0.1, right=0.9, wspace=0.45, bottom=0.15)
    
    pi = 0
    
    min_value = min(x_dyn + x_ap + x_iot)
    min_value = round(min_value, 1)
    if min_value < 0.35: min_value = 0.35
    max_value = max(x_dyn + x_ap + x_iot)
    
    title_text = ['Workload on UCDs', 'Workload on APs']
    lebel_ap = 'AP Training'
    lebel_iot = 'UCD training'
    lebel_dyn = lebel_pbfl

    linewidth = 1

    for index, col in enumerate(axes):
        if not max(plot_dat_ap[pi]) <= 1000:
            col.plot(x_ap, plot_dat_ap[pi], 'g', label=lebel_ap, linewidth=linewidth)
        if not max(plot_dat_iot[pi]) <= 1000:
            col.plot(x_iot, plot_dat_iot[pi], 'y', label=lebel_iot, linewidth=linewidth)
        if not max(plot_dat_dyn[pi]) <= 1000:
            col.plot(x_dyn, plot_dat_dyn[pi], 'r', label=lebel_dyn, linewidth=linewidth)
        col.set_title(title_text[pi])
        col.set_xlabel(x_text[pi])
        if index ==0:
            text_ = 'acc. up=5.78%\ncost down=56.57%'
            col.set_ylabel(y_text[pi])
            col.annotate('', xy=(0.88, 27500), xycoords='data',
                        xytext=(0.82, 60500), textcoords='data',
                        arrowprops=dict(arrowstyle="<->"))#connectionstyle="bar",
                                        #ec="k", shrinkA=2, shrinkB=2))
            col.annotate(text_, xy=(0.86, 44000), xycoords='data',
                        xytext=(-80, -30), textcoords='offset points',
                        arrowprops=dict(arrowstyle="->"))

        if xinverse == True: col.invert_xaxis()
        #if pi > 0: col.set_xlim(min_value, max_value+0.05)
        col.set_xlim(xlim_vec[0], xlim_vec[1])
        if ylim_vec is not None: col.set_ylim(0, ylim_vec[pi])
        col.yaxis.set_major_formatter(FuncFormatter(y_fmt))

        col_tx = col.twinx()

        if not max(plot_dat_ap[pi+2]) <= 1000:
            col_tx.plot(x_ap, plot_dat_ap[pi+2], 'limegreen', label=lebel_ap, linewidth=linewidth, linestyle='')
        if not max(plot_dat_iot[pi+2]) <= 1000:
            col_tx.plot(x_iot, plot_dat_iot[pi+2], 'y', label=lebel_iot, linewidth=linewidth, linestyle='')
        if not max(plot_dat_dyn[pi+2]) <= 1000:
            col_tx.plot(x_dyn, plot_dat_dyn[pi+2], 'r', label=lebel_dyn, linewidth=linewidth, linestyle='')
        #if pi > 0:
        #    col_tx.set_ylabel(y_text[pi+2], color='limegreen')
        #else:
        if index ==1: col_tx.set_ylabel(y_text[pi+2])
        
        col_tx.yaxis.set_major_formatter(FuncFormatter(y_fmt))

        #if index ==1: 
        col.legend()

        pi += 1

    plt.savefig('imgs/' + filename + '.pdf')



# load configuration from cfg file
computation_cfg, communication_cfg = load_res_from_cfg()
iot_freq, ap_freq, count_inference, iot_energy_mac, ap_energy_mac = computation_cfg
up_iot, down_iot, up_ap, down_ap, sample_size, iot_energy_comm, ap_energy_comm = communication_cfg


if args.mac:
    acc_ap, acc_iot, acc_dyn = load_acc_one(post_arg, res_folder)
    plot_dat_dyn, plot_dat_ap, plot_dat_iot, _, _, _ = load_mac_latency_energy(encoder,classifier, post_arg, iot_freq, ap_freq, iot_energy_mac, ap_energy_mac)

    # plot mac
    title_text = ['Computation workload on UCDs', 'Computation workload on APs']
    x_text = ['Test Accuracy'] * 2
    y_text = ['Accumulated MAC operations (Million)'] * 2
    y_text += ['Accumulated Latency (Seconds)'] * 2
    y_text += ['Accumulated Energy (Joule)'] * 2
    xlim_vec, ylim_vec = [0.35, 0.9], None
    filename = 'mac_latency_energy_accuracy'

    plot_12_3y(plot_dat_dyn, plot_dat_ap, plot_dat_iot, acc_dyn, acc_ap, acc_iot, title_text, x_text, y_text, xlim_vec, ylim_vec, filename, xinverse=False)


if args.comm:
    acc_ap, acc_iot, acc_dyn = load_acc_one(post_arg, res_folder)
    plot_dat_dyn, plot_dat_ap, plot_dat_iot, _, _, _ = load_comm_latency_energy(encoder,classifier, post_arg, up_iot, down_iot, up_ap, down_ap, iot_energy_comm, ap_energy_comm, sample_size)

    # plot mac
    title_text = ['Communication workload on UCDs', 'Communication workload on APs']
    x_text = ['Test Accuracy'] * 2
    y_text = ['Accumulated Comm. Amount (Migabytes)'] * 2
    y_text += ['Accumulated Latency (Seconds)'] * 2
    y_text += ['Accumulated Energy (Joule)'] * 2
    xlim_vec, ylim_vec = [0.35, 0.9], None
    filename = 'comm_latency_energy_accuracy'
    
    plot_12_3y(plot_dat_dyn, plot_dat_ap, plot_dat_iot, acc_dyn, acc_ap, acc_iot, title_text, x_text, y_text, xlim_vec, ylim_vec, filename, xinverse=False)


if args.sum:
    acc_ap, acc_iot, acc_dyn = load_acc_one(post_arg, res_folder)

    mac_dyn, mac_ap, mac_iot, _, _, _ = load_mac_latency_energy(encoder,classifier, post_arg, iot_freq, ap_freq, iot_energy_mac, ap_energy_mac)
    comm_dyn, comm_ap, comm_iot, _, _, _ = load_comm_latency_energy(encoder,classifier, post_arg, up_iot, down_iot, up_ap, down_ap, iot_energy_comm, ap_energy_comm, sample_size)

    sum_dyn, sum_ap, sum_iot = [], [], []
    for k in range(2, len(mac_dyn)):
        sum_dyn.append([i+j for i,j in zip(mac_dyn[k], comm_dyn[k])])
        sum_ap.append([i+j for i,j in zip(mac_ap[k], comm_ap[k])])
        sum_iot.append([i+j for i,j in zip(mac_iot[k], comm_iot[k])])
    
    # print gain
    cost_down = (max(sum_iot[2]) - max(sum_dyn[2])) / max(sum_iot[2])
    acc_up = max(acc_dyn) - max(acc_iot)
    print(f"cost down: {cost_down}; accuracy up: {acc_up}")

    x_text = ['Test Accuracy'] * 2
    y_text = ['Accumulated Latency (Seconds)'] * 2
    y_text += ['Accumulated Energy (Joule)'] * 2

    xlim_vec, ylim_vec = [0.35, 0.9], None
    
    filename = 'sum_latency_energy_accuracy'
    plot_12_2y(sum_dyn, sum_ap, sum_iot, acc_dyn, acc_ap, acc_iot, x_text, y_text, xlim_vec, ylim_vec, filename, xinverse=False)