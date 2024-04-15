import sys
import os

import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

from measures.mac_comm_counter import mac_latency_energy_counter, comm_latency_energy_counter

linestyle_tuple = {
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
     }


# https://stackoverflow.com/questions/40566413/matplotlib-pyplot-auto-adjust-unit-of-y-axis
def y_fmt(y, pos):
    decades = [1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9 ]
    suffix  = ["G", "M", "K", "" , "m" , "u", "n"  ]
    if y == 0:
        return str(0)
    for i, d in enumerate(decades):
        if np.abs(y) >= d:
            val = round(y/float(d),3) # fix unlimited 0.99999 issue
            signf = len(str(val).split(".")[1])
            if signf == 0:
                return '{val:d}{suffix}'.format(val=int(val), suffix=suffix[i])
            else:
                if signf == 1:
                    if str(val).split(".")[1] == "0":
                       return '{val:d}{suffix}'.format(val=int(str(round(val))), suffix=suffix[i])
                tx = "{"+"val:.{signf}f".format(signf = signf) +"}{suffix}"
                return tx.format(val=val, suffix=suffix[i])
            #return y
    return y


def load_acc_pure(post_arg, res_folder):
    filedir = res_folder + "/" + "acc_ap_" + post_arg + ".csv"
    dat_full = pd.read_csv(filedir, sep=',')

    filedir = res_folder + "/" + "acc_iot_" + post_arg + ".csv"
    dat_clas = pd.read_csv(filedir, sep=',')

    filedir = res_folder + "/" + "acc_dyn_" + post_arg + ".csv"
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


def load_all_acc(res_folder, focus='basic'):
    onlyfiles = [f for f in listdir(res_folder) if isfile(join(res_folder, f))]
    print(onlyfiles)
    1/0
    acc_ap = [filename for filename in onlyfiles if 'acc_ap_1' in filename]
    acc_iot = [filename for filename in onlyfiles if 'acc_iot_1' in filename]
    acc_dyn = [filename for filename in onlyfiles if 'acc_dyn_1' in filename]
    acc_x = acc_ap + acc_iot + acc_dyn

    if focus == 'vanilla':
        acc_vanilla = [filename for filename in onlyfiles if 'acc_apvanilla_1' in filename]
        acc_x = acc_vanilla

    splitted_list = []
    for acc_name in acc_x:
        # marking based on the filename
        splitted = acc_name.split('_')
        
        if focus == 'basic':
            if len(splitted) != 10: continue
            splitted = [splitted[1], splitted[2][1:], splitted[3][1:], splitted[4][1:],
            splitted[5].split('|')[0][1:], splitted[5].split('|')[1],
            splitted[6][1:], splitted[7][1:], splitted[8][1:], splitted[9][1:].replace('.csv','')]
            name_base = ['dest','dataset','encoder','classifier','clientin','clientall', 'epoch', 'alpha','beta','gamma']
        elif focus == 'flpa':
            if len(splitted) != 11: continue
            splitted = [splitted[1], splitted[2][1:], splitted[3][1:], splitted[4][1:],
            splitted[5].split('|')[0][1:], splitted[5].split('|')[1],
            splitted[6][1:], splitted[7][1:], splitted[8][1:], splitted[9][1:], splitted[10][4:].replace('.csv','')]
            name_base = ['dest','dataset','encoder','classifier','clientin','clientall', 'epoch', 'alpha','beta','gamma','flpa']
        elif focus == 'mr':
            if len(splitted) != 12: continue
            splitted = [splitted[1], splitted[2][1:], splitted[3][1:], splitted[4][1:],
            splitted[5].split('|')[0][1:], splitted[5].split('|')[1],
            splitted[6][1:], splitted[7][1:], splitted[8][1:], splitted[9][1:], splitted[10][4:], splitted[11].split('|')[0][2:], splitted[11].split('|')[1].replace('.csv','')]
            name_base = ['dest','dataset','encoder','classifier','clientin','clientall', 'epoch', 'alpha','beta','gamma','flpa','mr1','mr2']
        elif focus == 'vanilla':
            splitted = [splitted[1], splitted[2][1:], splitted[3][1:], splitted[4][1:],
            splitted[5].split('|')[0][1:], splitted[5].split('|')[1],
            splitted[6][1:], splitted[7][1:], splitted[8][1:], splitted[9][1:], splitted[10][4:], splitted[11].split('|')[0][2:], splitted[11].split('|')[1].replace('.csv','')]
            name_base = ['dest','dataset','encoder','classifier','clientin','clientall', 'epoch', 'alpha','beta','gamma','flpa','mr1','mr2']

        # read highest accuracy
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

    df_acc.loc[df_acc['encoder'] == 'effi','encoder'] = 'efficient'
    df_acc.loc[df_acc['encoder'] == 'mobi','encoder'] = 'mobile'
    df_acc.loc[df_acc['encoder'] == 'shuf','encoder'] = 'shuffle'
    df_acc.loc[df_acc['dataset'] == 'cifar10','dataset'] = 'cf10'
    df_acc.loc[df_acc['dataset'] == 'cifar100','dataset'] = 'cf100'
    
    df_acc.loc[df_acc['classifier'] == 'small','classifier'] = 'S'
    df_acc.loc[df_acc['classifier'] == 'medium','classifier'] = 'M'
    df_acc.loc[df_acc['classifier'] == 'large','classifier'] = 'L'

    return df_acc



def running_avg(arr, window_size=3):  
    i = 0
    first = arr[0]
    last = arr[len(arr)-1]
    
    moving_averages = []
    
    while i < len(arr) - window_size + 1:
        
        window = arr[i : i + window_size]
        window_average = round(sum(window) / window_size, 2)
        moving_averages.append(window_average)
        i += 1
    
    moving_averages = [first] + moving_averages + [last]
    return moving_averages



def load_mac_latency_energy(encoder, classifier, post_args, iot_freq, ap_freq, iot_energy_mac, ap_energy_mac, dests=['dyn','ap','iot'], count_inference=True):
    plot_dat_dyn, plot_dat_ap, plot_dat_iot, rounds_dyn, rounds_ap, rounds_iot = None, None, None, None, None, None
    
    if 'dyn' in dests:
        plot_dat_dyn, rounds_dyn = mac_latency_energy_counter(encoder, classifier, post_args, 'dyn', 
                                        iot_freq, ap_freq, iot_energy_mac, ap_energy_mac, count_inference)
    if 'ap' in dests:
        plot_dat_ap, rounds_ap = mac_latency_energy_counter(encoder, classifier, post_args, 'ap', 
                                        iot_freq, ap_freq, iot_energy_mac, ap_energy_mac, count_inference)
    if 'iot' in dests:
        plot_dat_iot, rounds_iot = mac_latency_energy_counter(encoder, classifier, post_args, 'iot', 
                                        iot_freq, ap_freq, iot_energy_mac, ap_energy_mac, count_inference)
    
    return plot_dat_dyn, plot_dat_ap, plot_dat_iot, rounds_dyn, rounds_ap, rounds_iot



def load_comm_latency_energy(encoder,classifier, post_args, up_iot, down_iot, up_ap, down_ap, iot_energy_comm, ap_energy_comm,sample_size, dests=['dyn','ap','iot']):
    plot_dat_dyn, plot_dat_ap, plot_dat_iot, rounds_dyn, rounds_ap, rounds_iot = None, None, None, None, None, None

    if 'dyn' in dests:
        plot_dat_dyn, rounds_dyn = comm_latency_energy_counter(encoder,classifier, post_args, 'dyn', 
                                        up_iot, down_iot, up_ap, down_ap, iot_energy_comm, ap_energy_comm, sample_size)
    if 'ap' in dests:
        plot_dat_ap, rounds_ap = comm_latency_energy_counter(encoder,classifier, post_args, 'ap', 
                                        up_iot, down_iot, up_ap, down_ap, iot_energy_comm, ap_energy_comm, sample_size)
    if 'iot' in dests:
        plot_dat_iot, rounds_iot = comm_latency_energy_counter(encoder,classifier, post_args, 'iot', 
                                        up_iot, down_iot, up_ap, down_ap, iot_energy_comm, ap_energy_comm, sample_size)

    return plot_dat_dyn, plot_dat_ap, plot_dat_iot, rounds_dyn, rounds_ap, rounds_iot



def res(configs, type, feature):
    return float(configs[configs['type'] == type][feature])


def load_res_from_cfg():
    
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
    
    computation_cfgs = iot_frequency, ap_frequency*acceleration, count_inference, iot_energy_mac, ap_energy_mac
    communication_cfgs = up_iot, down_iot, up_ap, down_ap, sample_size, iot_energy_comm, ap_energy_comm
    return computation_cfgs, communication_cfgs
            


