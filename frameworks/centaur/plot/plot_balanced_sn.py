import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from plot.utils import y_fmt, load_acc, load_mac_latency_energy, load_comm_latency_energy

plt.rcParams.update({'font.size': 8})

parser = argparse.ArgumentParser(description="Plot")
parser.add_argument("--plot_acc", type=bool, default=False)
parser.add_argument("--plot_mac", type=bool, default=False)
parser.add_argument("--plot_comm", type=bool, default=False)
parser.add_argument("--plot_sum", type=bool, default=True)

args = parser.parse_args()

res_folder = "results_balanced_sn"
# post_args = "_c8_e3_1mob_a5.0_b3.0_g0.0"  # arguments for dynamic test


# def plot_acc():
#     dat_full, dat_clas, dat_dyn = load_acc(res_folder)

#     # import pdb; pdb.set_trace()
#     min_value = min(dat_dyn['accuracy'].to_list() + dat_clas['accuracy'].to_list() + dat_dyn['accuracy'].to_list())
#     min_value = round(min_value, 1) + 0.2
#     max_value = max(dat_dyn['accuracy'].to_list() + dat_clas['accuracy'].to_list() + dat_dyn['accuracy'].to_list())

#     plt.figure(figsize=(4,3))
#     plt.gcf().subplots_adjust(left=0.15, bottom=0.15)

#     plt.plot(dat_full['round'],dat_full['accuracy'], 'g', label='Access point training')
#     plt.plot(dat_clas['round'],dat_clas['accuracy'], 'y', label='Tiny end device training')
#     plt.plot(dat_dyn['round'],dat_dyn['accuracy'], 'r',  label='Partition-based training')
#     plt.yticks(np.arange(min_value, max_value+0.05, 0.1))
#     plt.ylim(0.3, 0.95)
#     plt.legend()
#     plt.xlabel('Communication Rounds')
#     plt.ylabel('Test Accuracy')
#     plt.savefig(res_folder + '/acc.pdf')



# def plot_22(plot_dat_dyn, plot_dat_ap, plot_dat_iot, x_dyn, x_ap, x_iot, title_text, x_text, y_text, xlim_vec, ylim_vec, filename, xinverse=False):
#     figs, axes = plt.subplots(nrows=2, ncols=2)
#     pi = 0
#     for row in axes:
#         for col in row:
#             if not max(plot_dat_ap[pi]) == 0:
#                 col.plot(x_ap, plot_dat_ap[pi], 'g', label='ap')
#             if not max(plot_dat_iot[pi]) == 0:
#                 col.plot(x_iot, plot_dat_iot[pi], 'y', label='iot')
#             if not max(plot_dat_dyn[pi]) == 0:
#                 col.plot(x_dyn, plot_dat_dyn[pi], 'r', label='dyn')
#             col.set_title(title_text[pi])
#             col.set_xlabel(x_text[pi])
#             col.set_ylabel(y_text[pi])
#             if xinverse == True: col.invert_xaxis()
#             if xlim_vec is not None: col.set_xlim(0, xlim_vec[pi])
#             if ylim_vec is not None: col.set_ylim(0, ylim_vec[pi])
#             col.legend()
#             pi += 1
#     figs.tight_layout()
#     plt.savefig('results/' + filename + '.pdf')


def plot_12_3y(plot_dat_dyn, plot_dat_ap, plot_dat_iot, x_dyn, x_ap, x_iot, x_text, y_text, xlim_vec, ylim_vec, filename, xinverse=False):
    figs, axes = plt.subplots(nrows=1, ncols=2, figsize=(9,2.5))
    plt.subplots_adjust(left=0.1, right=0.85, wspace=0.9, bottom=0.15)
    pi = 0

    min_value = min(x_dyn + x_ap + x_iot)
    min_value = round(min_value, 1)
    max_value = max(x_dyn + x_ap + x_iot)
    
    title_text = ['Workload on Tiny End Devices', 'Workload on Access Points']
    lebel_ap = 'Access point training'
    lebel_iot = 'Tiny end device training'
    lebel_dyn = 'Partition-based training'

    for col in axes:
        if not max(plot_dat_ap[pi]) == 0:
            col.plot(x_ap, plot_dat_ap[pi], 'g', label=lebel_ap)
        if not max(plot_dat_iot[pi]) == 0:
            col.plot(x_iot, plot_dat_iot[pi], 'y', label=lebel_iot)
        if not max(plot_dat_dyn[pi]) == 0:
            col.plot(x_dyn, plot_dat_dyn[pi], 'r', label=lebel_dyn)
        col.set_title(title_text[pi])
        col.set_xlabel(x_text[pi])
        col.set_ylabel(y_text[pi])


        if xinverse == True: col.invert_xaxis()
        if xlim_vec is not None: col.set_xlim(0, xlim_vec[pi])
        if ylim_vec is not None: col.set_ylim(0, ylim_vec[pi])
        col.yaxis.set_major_formatter(FuncFormatter(y_fmt))

        col.legend()

        col_tx = col.twinx()
        col_tx2 = col.twinx()

        if not max(plot_dat_ap[pi+2]) == 0:
            col_tx.plot(x_ap, plot_dat_ap[pi+2], 'g', label=lebel_ap)
        if not max(plot_dat_iot[pi+2]) == 0:
            col_tx.plot(x_iot, plot_dat_iot[pi+2], 'y', label=lebel_iot)
        if not max(plot_dat_dyn[pi+2]) == 0:
            col_tx.plot(x_dyn, plot_dat_dyn[pi+2], 'r', label=lebel_dyn)
        col_tx.set_ylabel(y_text[pi+2])
        col_tx.yaxis.set_major_formatter(FuncFormatter(y_fmt))
        
        if not max(plot_dat_ap[pi+4]) == 0:
            col_tx2.plot(x_ap, plot_dat_ap[pi+4], 'g', label=lebel_ap)
        if not max(plot_dat_iot[pi+4]) == 0:
            col_tx2.plot(x_iot, plot_dat_iot[pi+4], 'y', label=lebel_iot)
        if not max(plot_dat_dyn[pi+4]) == 0:
            col_tx2.plot(x_dyn, plot_dat_dyn[pi+4], 'r', label=lebel_dyn)
        col_tx2.set_ylabel(y_text[pi+4])
        col_tx2.yaxis.set_major_formatter(FuncFormatter(y_fmt))

        col_tx2.spines['right'].set_position(('outward', 45))

        pi += 1

    # figs.tight_layout()
    plt.savefig(res_folder + '/' + filename + '.pdf')


def plot_12_2y(plot_dat_dyn, plot_dat_ap, plot_dat_iot, x_dyn, x_ap, x_iot, x_text, y_text, xlim_vec, ylim_vec, filename, xinverse=False):
    figs, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,2.8))
    plt.subplots_adjust(left=0.1, right=0.85, wspace=0.8, bottom=0.15)
    
    pi = 0
    
    min_value = min(x_dyn + x_ap + x_iot)
    min_value = round(min_value, 1)
    if min_value < 0.35: min_value = 0.35
    max_value = max(x_dyn + x_ap + x_iot)
    
    title_text = ['Workload on Tiny End Devices', 'Workload on Access Points']
    lebel_ap = 'Access point training'
    lebel_iot = 'Tiny end device training'
    lebel_dyn = 'Partition-based training'

    linewidth = 1

    for col in axes:
        if not max(plot_dat_ap[pi]) <= 1000:
            col.plot(x_ap, plot_dat_ap[pi], 'limegreen', label=lebel_ap, linewidth=linewidth)
        if not max(plot_dat_iot[pi]) <= 1000:
            col.plot(x_iot, plot_dat_iot[pi], 'y', label=lebel_iot, linewidth=linewidth)
        if not max(plot_dat_dyn[pi]) <= 1000:
            col.plot(x_dyn, plot_dat_dyn[pi], 'r', label=lebel_dyn, linewidth=linewidth)
        col.set_title(title_text[pi])
        col.set_xlabel(x_text[pi])
        col.set_ylabel(y_text[pi])


        if xinverse == True: col.invert_xaxis()
        if pi > 0: col.set_xlim(max_value+0.05, min_value)
        if ylim_vec is not None: col.set_ylim(0, ylim_vec[pi])
        col.yaxis.set_major_formatter(FuncFormatter(y_fmt))

        col_tx = col.twinx()

        if not max(plot_dat_ap[pi+2]) <= 1000:
            col_tx.plot(x_ap, plot_dat_ap[pi+2], 'g', label=lebel_ap, linewidth=linewidth)
        if not max(plot_dat_iot[pi+2]) <= 1000:
            col_tx.plot(x_iot, plot_dat_iot[pi+2], 'y', label=lebel_iot, linewidth=linewidth)
        if not max(plot_dat_dyn[pi+2]) <= 1000:
            col_tx.plot(x_dyn, plot_dat_dyn[pi+2], 'r', label=lebel_dyn, linewidth=linewidth)
        if pi > 0:
            col_tx.set_ylabel(y_text[pi+2], color='g')
        else:
            col_tx.set_ylabel(y_text[pi+2])
        col_tx.yaxis.set_major_formatter(FuncFormatter(y_fmt))

        col_tx.legend()

        pi += 1

    # figs.tight_layout()
    plt.savefig(res_folder + '/' + filename + '.pdf')


# def plot_mac_latency_xround(iot_freq, ap_freq, count_inference=True):
#     plot_dat_dyn, plot_dat_ap, plot_dat_iot, rounds_dyn, rounds_ap, rounds_iot = load_mac_latency(iot_freq, ap_freq)

#     # plot mac
#     title_text = ['on AP','on IoT','on AP','on IoT']
#     x_text = ['round'] * 4
#     y_text = ['Accumulated MAC operations (Million)', 'Accumulated MAC operations (Million)', 
#                         'Accumulated Latency (Seconds)', 'Accumulated Latency (Seconds)']

#     ylim1 = max(plot_dat_ap[0] + plot_dat_iot[0] + plot_dat_dyn[0] + plot_dat_ap[1] + plot_dat_iot[1] + plot_dat_dyn[1])
#     ylim2 = max(plot_dat_ap[2] + plot_dat_iot[2] + plot_dat_dyn[2] + plot_dat_ap[3] + plot_dat_iot[3] + plot_dat_dyn[3])
#     ylim_vec = [ylim1,ylim1,ylim2,ylim2]

#     xlim_vec = [50] * 4
    
#     filename = 'mac_latency'
#     plot_22(plot_dat_dyn, plot_dat_ap, plot_dat_iot, rounds_dyn, rounds_ap, rounds_iot, title_text, x_text, y_text, xlim_vec, ylim_vec, filename)





def plot_mac_latency_energy_xaccuracy(iot_freq, ap_freq, iot_energy_mac, ap_energy_mac, count_inference=True):
    
    dat_full, dat_clas, dat_dyn = load_acc(res_folder)
    acc_ap = dat_full['accuracy'].tolist()[1:]
    acc_iot = dat_clas['accuracy'].tolist()[1:]
    acc_dyn = dat_dyn['accuracy'].tolist()[1:]

    plot_dat_dyn, plot_dat_ap, plot_dat_iot, _, _, _ = load_mac_latency_energy(iot_freq, ap_freq, iot_energy_mac, ap_energy_mac)
    
    # plot mac
    x_text = ['Test Accuracy'] * 2
    y_text = ['Accumulated MAC operations (Million)'] * 2
    y_text += ['Accumulated Latency (Seconds)'] * 2
    y_text += ['Accumulated Energy (Joule)'] * 2

    xlim_vec, ylim_vec = None, None

    filename = 'mac_latency_energy_accuracy'
    plot_12_3y(plot_dat_dyn, plot_dat_ap, plot_dat_iot, acc_dyn, acc_ap, acc_iot, x_text, y_text, xlim_vec, ylim_vec, filename, xinverse=True)




# def plot_comm_latency_xround(up_iot, down_iot, up_ap, down_ap, sample_size):
#     plot_dat_dyn, plot_dat_ap, plot_dat_iot, rounds_dyn, rounds_ap, rounds_iot = load_comm_latency(up_iot, down_iot, up_ap, down_ap, sample_size)

#     # plot mac
#     figs, axes = plt.subplots(nrows=2, ncols=2)
#     title_text = ['on AP','on IoT','on AP','on IoT']
#     x_text = ['round'] * 4
#     y_text = ['Accumulated Comm. Amount (Migabytes)', 'Accumulated Comm. Amount (Migabytes)', 
#                         'Accumulated Latency (Seconds)', 'Accumulated Latency (Seconds)']

#     #ylim1 = max(plot_dat_ap[0] + plot_dat_iot[0] + plot_dat_dyn[0] + plot_dat_ap[1] + plot_dat_iot[1] + plot_dat_dyn[1])
#     #ylim2 = max(plot_dat_ap[2] + plot_dat_iot[2] + plot_dat_dyn[2] + plot_dat_ap[3] + plot_dat_iot[3] + plot_dat_dyn[3])
#     #ylim_vec = [ylim1, ylim1, ylim2, ylim2]
#     ylim_vec = None

#     xlim_vec = [50] * 4

#     filename = 'comm_latency'
#     plot_22(plot_dat_dyn, plot_dat_ap, plot_dat_iot, rounds_dyn, rounds_ap, rounds_iot, title_text, x_text, y_text, xlim_vec, ylim_vec, filename)



def plot_comm_latency_energy_xaccuracy(up_iot, down_iot, up_ap, down_ap, iot_energy_comm, ap_energy_comm, sample_size):
    dat_full, dat_clas, dat_dyn = load_acc(res_folder)
    acc_ap = dat_full['accuracy'].tolist()[1:]
    acc_iot = dat_clas['accuracy'].tolist()[1:]
    acc_dyn = dat_dyn['accuracy'].tolist()[1:]

    plot_dat_dyn, plot_dat_ap, plot_dat_iot, _, _, _ = load_comm_latency_energy(up_iot, down_iot, up_ap, down_ap, 
                                                    iot_energy_comm, ap_energy_comm, sample_size)

    # plot mac
    x_text = ['Test Accuracy'] * 2
    y_text = ['Accumulated Comm. Amount (Migabytes)'] * 2
    y_text += ['Accumulated Latency (Seconds)'] * 2
    y_text += ['Accumulated Energy (Joule)'] * 2

    xlim_vec, ylim_vec = None, None
    
    filename = 'comm_latency_energy_accuracy'
    plot_12_3y(plot_dat_dyn, plot_dat_ap, plot_dat_iot, acc_dyn, acc_ap, acc_iot, x_text, y_text, xlim_vec, ylim_vec, filename, xinverse=True)



def plot_sum_latency_energy_xaccuracy(iot_freq, ap_freq, iot_energy_mac, ap_energy_mac, count_inference,
                                        up_iot, down_iot, up_ap, down_ap, iot_energy_comm, ap_energy_comm, sample_size):

    dat_full, dat_clas, dat_dyn = load_acc(res_folder)
    acc_ap = dat_full['accuracy'].tolist()[1:]
    acc_iot = dat_clas['accuracy'].tolist()[1:]
    acc_dyn = dat_dyn['accuracy'].tolist()[1:]
    mac_dyn, mac_ap, mac_iot, _, _, _ = load_mac_latency_energy(iot_freq, ap_freq, iot_energy_mac, ap_energy_mac)
    comm_dyn, comm_ap, comm_iot, _, _, _ = load_comm_latency_energy(up_iot, down_iot, up_ap, down_ap, 
                                                    iot_energy_comm, ap_energy_comm, sample_size)

    sum_dyn, sum_ap, sum_iot = [], [], []
    for k in range(2, len(mac_dyn)):
        sum_dyn.append([i+j for i,j in zip(mac_dyn[k], comm_dyn[k])])
        sum_ap.append([i+j for i,j in zip(mac_ap[k], comm_ap[k])])
        sum_iot.append([i+j for i,j in zip(mac_iot[k], comm_iot[k])])

    x_text = ['Test Accuracy'] * 2
    y_text = ['Accumulated Latency (Seconds)'] * 2
    y_text += ['Accumulated Energy (Joule)'] * 2

    xlim_vec, ylim_vec = None, None
    
    filename = 'sum_latency_energy_accuracy'
    plot_12_2y(sum_dyn, sum_ap, sum_iot, acc_dyn, acc_ap, acc_iot, x_text, y_text, xlim_vec, ylim_vec, filename, xinverse=True)



def res(configs, type, feature):
    return float(configs[configs['type'] == type][feature])




# plot accuracy only
if args.plot_acc:
    plot_acc()


cfg = pd.read_csv('client_configurations.csv', sep=',')


# plot mac, latency, energy, accuracy
iot_frequency = res(cfg, 'iot', 'frequency(MHz)')
ap_frequency = res(cfg, 'ap', 'frequency(MHz)')
acceleration = res(cfg, 'ap', 'acceleration') # how many times faster the accelerator is than ap CPU
count_inference = True

iot_energy_mac = iot_frequency * res(cfg, 'iot', 'power(mWpMHz)') # times = milliWatt/Mhz https://www.prnewswire.com/news-releases/ultra-reliable-arm-cortex-m4f-microcontroller-from-maxim-integrated-offers-industrys-lowest-power-consumption-and-smallest-size-for-industrial-healthcare-and-iot-sensor-applications-301080651.html#:~:text=Lowest%20Power%3A%2040%C2%B5W%2FMHz%20of,5mm%2Dx%2D5mm%20TQFN.
ap_energy_mac = ap_frequency * res(cfg, 'ap', 'power(mWpMHz)') # times = milliWatt/MHz
# https://www.researchgate.net/figure/Energy-consumption-for-different-parts-of-the-mobile-phone_tbl1_224248235#:~:text=The%20power%20consumption%20of%20wearable,%5B18%5D.%20...



# # plot communication amount, latency, energy, accuracy
up_iot = res(cfg, 'iot', 'uplink(Mbit/s)') # BLE5 2 Mbit/s
down_iot = res(cfg, 'iot', 'downlink(Mbit/s)') # BLE5 2 Mbit/s
up_ap = res(cfg, 'ap', 'uplink(Mbit/s)') # WIFI 10 Mbit/s
down_ap = res(cfg, 'iot', 'downlink(Mbit/s)') # WIFI 100 Mbit/s
sample_size = 30 # KiloBytes

iot_energy_comm = res(cfg, 'iot', 'energy_comm(W)') # Watt BLE5
ap_energy_comm = res(cfg, 'ap', 'energy_comm(W)') # Watt WIFI


if args.plot_mac:
    #plot_mac_latency_xround(iot_frequency, ap_frequency*acceleration, count_inference)
    plot_mac_latency_energy_xaccuracy(iot_frequency, ap_frequency*acceleration, iot_energy_mac, ap_energy_mac, count_inference)

if args.plot_comm:
    # plot_comm_latency_xround(up_iot, down_iot, up_ap, down_ap, sample_size)
    plot_comm_latency_energy_xaccuracy(up_iot, down_iot, up_ap, down_ap, iot_energy_comm, ap_energy_comm, sample_size)

if args.plot_sum:
    plot_sum_latency_energy_xaccuracy(iot_frequency, ap_frequency*acceleration, iot_energy_mac, ap_energy_mac, count_inference,
                                        up_iot, down_iot, up_ap, down_ap, iot_energy_comm, ap_energy_comm, sample_size)