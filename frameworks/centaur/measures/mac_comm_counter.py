import torch
import torchvision
import pandas as pd
import numpy as np
from fvcore.nn import flop_count_table, flop_count_str, FlopCountAnalysis

from hiefed.client import init_network

# functions used to compute MAC operations, communication amounts, latency, energy cost
res_folder = "results"

# train K M B text to numbers (used in plots)
def text_to_num(text, bad_data_val = 0):
    d = {
        'K': 1000,
        'M': 1000000,
        'G': 1000000000
    }
    if not isinstance(text, str):
        return bad_data_val

    elif text[-1] in d: # separate out the K, M, or B
        num, magnitude = text[:-1], text[-1]
        return int(float(num) * d[magnitude])
    else:
        try: # catch exceptions
            return float(text)
        except Exception as e:
            return None


# Dynamically generate column names
# The max column count a line in the file could have
def read_file_imbalanced_col(data_file):
   
    largest_column_count = 0

    # Loop the data lines
    with open(data_file, 'r') as temp_f:
        lines = temp_f.readlines()

        for l in lines:
            column_count = len(l.split(',')) + 1
            largest_column_count = column_count if largest_column_count < column_count else largest_column_count

    if largest_column_count > 20: largest_column_count = 20 + 3   # avoiding Error tokenizing data. C error: Buffer overflow caught - possible malformed input file.

    # Generate column names (will be 0, 1, 2, ..., largest_column_count - 1)
    column_names = ['dest','round','cid']
    column_names = column_names + ['samples_e'+str(i) for i in range(1, largest_column_count-2)]

    #print(data_file)
    #if data_file == 'results/sample_number_dyn_1cifar10_2mobi_3medium_c8|100_e3_a5_b3_g0_flpa1000_mr0.1|0.4.csv':
    #    import pdb; pdb.set_trace()
    df = pd.read_csv(data_file, header=None, skiprows=1, delimiter=',', names=column_names, lineterminator='\n')

    return df


# get flops using fvcore.nn library (from facebook AI)
def get_flops_parameters(Net, input):
    flops = FlopCountAnalysis(Net, input)

    flops_ = flop_count_table(flops)
    flops_ = flops_.replace(' ','')
    flops_ = flops_.replace('|',';')
    flops_ = flops_.replace(';\n;','\n')
    flops_ = flops_.replace('-','')
    flops_ = flops_.replace('\n:;:;:\n','\n')

    flops_ = flops_[1:]
    flops_ = flops_[:-1]

    with open(res_folder + "/model_flops.csv","w") as f:
        f.write(flops_)
    
    dat = pd.read_csv(res_folder + "/model_flops.csv", sep=";")
    return dat


def get_input_size(encoder, classifier):
    '''input sizes for encoders when computing FLOPs and weights'''
    
    if  encoder == 'sencoder_p':
        input = torch.rand(16, 1, 27, 200)
    elif  encoder == 'sencoder_u' or encoder == 'sencoder_m':
        input = torch.rand(16, 1, 9, 128)
    else:
        input = torch.rand(16, 3, 3, 3)
    
    if encoder == 'mobilenet' or encoder == 'efficientnet':
        classifier_str = 'classifier.0'
        encoder_str = 'features'

    elif  encoder == 'mnasnet':
        classifier_str = 'classifier'
        encoder_str = 'layers'
    
    elif encoder == 'shufflenet':
        classifier_str = 'fc'
        encoder_str = None
    elif  encoder == 'sencoder_p' or encoder == 'sencoder_u' or encoder == 'sencoder_m':
        classifier_str = 'classifier'
        encoder_str = 'encoder'
    else:
        raise NameError("input size does not defined!")
    
    return input, classifier_str, encoder_str


def mac_counter(encoder, classifier, num_classes=10):
    '''MAC operations counter'''
    if  encoder == 'sencoder_p':
        num_classes = 13
    elif  encoder == 'sencoder_u' or encoder == 'sencoder_m':
        num_classes = 6
    Net = init_network(encoder=encoder, classifier=classifier, num_classes=num_classes, pre_trained=False)
    input, classifier_str, encoder_str = get_input_size(encoder, classifier)
    
    # get flops
    dat = get_flops_parameters(Net, input)
    for i in range(len(dat['#flops'])):
        dat['#flops'][i] = text_to_num(dat['#flops'][i])
    dat['#flops'] = dat['#flops'].apply(pd.to_numeric)
    
    # forward MAC operation computation (1 FLOP = 2 MACs)
    mac_fw_full = 2 * dat.loc[dat["module"] == 'model']['#flops']
    
    mac_fw_clas = 2 * (int(dat.loc[dat["module"] == classifier_str]['#flops']))

    mac_fw_first = dat['#flops'][2]
    if encoder_str is None:
        mac_fw_encd = 2 * (mac_fw_full - mac_fw_clas)
    else:
        mac_fw_encd = 2 * dat.loc[dat["module"] == encoder_str]['#flops']

    mac_fw_encd_rest = mac_fw_encd - mac_fw_first

    # backward MAC operation computation 
    # https://www.lesswrong.com/posts/fnjKpBoWJXcSDwhZk/what-s-the-backward-forward-flop-ratio-for-neural-networks
    mac_bw_first = mac_fw_first
    mac_bw_encd_rest = 2 * mac_fw_encd_rest
    mac_bw_encd = mac_bw_first + mac_bw_encd_rest
    mac_bw_clas = 2 * mac_fw_clas
    mac_bw_full = mac_bw_encd + mac_bw_clas
    
    mac_fw_encd, mac_bw_encd, mac_fw_clas, mac_bw_clas= int(mac_fw_encd), int(mac_bw_encd), int(mac_fw_clas), int(mac_bw_clas)
    if encoder == 'sencoder_p':
        mac_fw_encd, mac_bw_encd, mac_fw_clas, mac_bw_clas = mac_fw_encd/200, mac_bw_encd/200, mac_fw_clas/200, mac_bw_clas/200
    elif encoder == 'sencoder_u' or encoder == 'sencoder_m':
        mac_fw_encd, mac_bw_encd, mac_fw_clas, mac_bw_clas = mac_fw_encd/42, mac_bw_encd/42, mac_fw_clas/42, mac_bw_clas/42
    return mac_fw_encd, mac_bw_encd, mac_fw_clas, mac_bw_clas




def para_counter(encoder, classifier, num_classes=10):
    '''Weight parameters counter'''
    if  encoder == 'sencoder_p':
        num_classes = 13
    elif  encoder == 'sencoder_u' or encoder == 'sencoder_m':
        num_classes = 6
    Net = init_network(encoder=encoder, classifier=classifier, num_classes=num_classes, pre_trained=False)
    input, classifier_str, encoder_str = get_input_size(encoder, classifier)

    # get flops
    dat = get_flops_parameters(Net, input)

    for i in range(len(dat['#parametersorshape'])):
        dat['#parametersorshape'][i] = text_to_num(dat['#parametersorshape'][i])
    dat['#parametersorshape'] = dat['#parametersorshape'].apply(pd.to_numeric)

    # classifier parameters
    para_clas = dat.loc[dat["module"] == classifier_str]['#parametersorshape']

    # encoder parameters
    if encoder_str is None:
        # model para
        para_model = dat.loc[dat["module"] == 'model']['#parametersorshape']
        para_encd = int(para_model) - int(para_clas)
    else:
        para_encd = dat.loc[dat["module"] == encoder_str]['#parametersorshape']

    return int(para_encd), int(para_clas)




# def get_mac_latency_energy_per(res, encoder='mobilenetv3'):
#    '''MAC operations, latency, and energy counter in realtime'''
#     if encoder == 'mobilenetv3':
#         mac_fw_encd, mac_bw_encd, mac_fw_clas, mac_bw_clas = get_mac_mobilenet()

#     mac_ap = mac_fw_encd + mac_bw_encd + mac_fw_clas + mac_bw_clas
#     mac_iot = mac_fw_encd + mac_fw_clas + mac_bw_clas

#     # compute latency (1 MAC operation = 2 instructions)
#     ap_freq = float(res.loc[res['type'] == 'ap']['frequency(MHz)'])
#     iot_freq = float(res.loc[res['type'] == 'iot']['frequency(MHz)'])
#     ap_latency = 2 * mac_ap / ap_freq
#     iot_latency = 2 * mac_iot / iot_freq

#     # compute energy Joule
#     ap_power = float(res.loc[res['type'] == 'ap']['power(mWpMHz)'])
#     iot_power = float(res.loc[res['type'] == 'iot']['power(mWpMHz)'])
#     ap_energy = mac_ap * ap_power
#     iot_energy = mac_iot * iot_power

#     return ap_latency, iot_latency, ap_energy, iot_energy





# def get_comm_latency_energy_per(res, encoder='mobilenetv3'):
#    '''Communication amount, latency, and energy counter in realtime'''
#     if encoder == 'mobilenetv3':
#         para_encd, para_clas = get_para_mobilenet()

#     para_ap = para_encd + para_clas
#     para_iot = para_clas

#     # float numbers to Migabytes (MB now)
#     para_ap = para_ap*4/(1024*1024)
#     para_iot = para_iot*4/(1024*1024)
    
#     # compute latency (1 bytes = 2 bits)
#     up_ap = float(res.loc[res['type'] == 'ap']['uplink(Mbit/s)'])
#     down_ap = float(res.loc[res['type'] == 'iot']['downlink(Mbit/s)'])
#     up_iot = float(res.loc[res['type'] == 'ap']['uplink(Mbit/s)'])
#     down_iot = float(res.loc[res['type'] == 'iot']['downlink(Mbit/s)'])
#     ap_latency = para_ap*2/up_ap + para_ap*2/down_ap
#     iot_latency = para_iot*2/up_iot + para_iot*2/down_iot

#     # compute energy Joule
#     ap_energy = float(res.loc[res['type'] == 'ap']['energy_comm(W)'])
#     iot_energy = float(res.loc[res['type'] == 'iot']['energy_comm(W)'])
#     ap_energy = para_ap*ap_energy
#     iot_energy = para_iot*iot_energy

#     return ap_latency, iot_latency, ap_energy, iot_energy




# compute accumulated MAC operation and latency during FL for AP and IoT
def mac_latency_energy_counter(encoder, classifier, post_args, destine, iot_freq, ap_freq, iot_energy_mac, ap_energy_mac, count_inference, rounds=100):
    filedir = res_folder + "/sample_number_" + destine + "_" + post_args + ".csv"
    # filedir = res_folder + "/sample_number_" + destine + post_args + ".csv"
    dat_sn = read_file_imbalanced_col(filedir)

    # compute mac
    #mac_fw_encd, mac_bw_encd, mac_fw_clas, mac_bw_clas = get_mac_mobilenet()
    mac_fw_encd, mac_bw_encd, mac_fw_clas, mac_bw_clas = mac_counter(encoder, classifier)

    mac_list, mac_acm_ap_list, mac_acm_iot_list = [], [], []

    # loop to counter MAC operations of AP and IoT
    for i in range(len(dat_sn['dest'])):
        if dat_sn.loc[i, 'dest'] == 'ap':
            if count_inference == True:
                mac_ = mac_fw_encd + mac_fw_clas + mac_bw_encd + mac_bw_clas
            else:
                mac_ =  mac_bw_encd + mac_bw_clas
            mac = np.nansum(dat_sn.loc[i,'samples_e1':]) * mac_ / 1000000
        
        elif dat_sn.loc[i, 'dest'] == 'iot':
            if count_inference == True:
                mac_ = mac_fw_encd + mac_fw_clas + mac_bw_clas
            else:
                mac_ =  mac_bw_clas
            mac = np.nansum(dat_sn.loc[i,'samples_e1':]) * mac_ / 1000000
        mac_list.append(mac)

    

    # average clients on each round and accumulate
    dat_sn = dat_sn.assign(mac=mac_list)
    dat_sn = dat_sn.groupby(['round', 'dest'],as_index=False).mean()
    if len(dat_sn['round']) < rounds: # fill if rounds missing
        extra_df = pd.DataFrame(list(range(1,rounds+1)),columns=['round'])
        dat_sn = pd.merge(dat_sn, extra_df, on='round', how="right")

    rounds = dat_sn['round']

    mac_acm_ap, mac_acm_iot = 0, 0
    for i in range(len(dat_sn['dest'])):
        if dat_sn.loc[i, 'dest'] == 'ap':
            mac_acm_ap += dat_sn.loc[i, 'mac']
        elif dat_sn.loc[i, 'dest'] == 'iot':
            mac_acm_iot += dat_sn.loc[i, 'mac']
        
        mac_acm_ap_list.append(mac_acm_ap)
        mac_acm_iot_list.append(mac_acm_iot)

    dat_sn = dat_sn.assign(mac_acm_ap=mac_acm_ap_list)
    dat_sn = dat_sn.assign(mac_acm_iot=mac_acm_iot_list)

    # compute latency (1 MAC operation = 2 instructions) https://stackoverflow.com/questions/9242024/multiply-add-a-a2-b-instruction-on-cpu
    ap_latency = [2*i/ap_freq for i in mac_acm_ap_list]
    iot_latency = [2*i/iot_freq for i in mac_acm_iot_list]

    # compute energy Joule
    ap_energy = [i*ap_energy_mac for i in ap_latency]
    iot_energy = [i*iot_energy_mac for i in iot_latency]

    plot_dat = [mac_acm_iot_list, mac_acm_ap_list, iot_latency, ap_latency, iot_energy, ap_energy]

    return plot_dat, rounds





# compute accumulated parameters to transmit and the latency during FL
def comm_latency_energy_counter(encoder, classifier, post_args, destine, up_iot, down_iot, up_ap, down_ap, iot_energy_comm, ap_energy_comm, sample_size, rounds=100):

    filedir = res_folder + "/sample_number_" + destine + "_" + post_args +  ".csv"
    # filedir = res_folder + "/sample_number_" + destine + post_args +  ".csv"
    dat_sn = read_file_imbalanced_col(filedir)

    para_encd, para_clas = para_counter(encoder, classifier)

    # loop to count passed parameters (one way) of AP and IoT (float numbers)
    para_list = []
    for i in range(len(dat_sn['dest'])):
        if dat_sn.loc[i, 'dest'] == 'ap':
            para = (para_encd + para_clas)
        elif dat_sn.loc[i, 'dest'] == 'iot':
            para = para_clas
            if destine == 'dyn':
                para += 0.5 * para_encd # 0.5 rate for cancelling upload out
        para_list.append(para)

    # loop to count passed samples from IoT to AP
    if not sample_size == 0:
        sample_list = []
        for i in range(len(dat_sn['dest'])):
            if dat_sn.loc[i, 'dest'] == 'ap':
                samples_s = sample_size * np.nanmean(dat_sn.loc[i, 'samples_e1':]) # here because AP can cache some data sample, so we estimate use averaged number of samples over epochs
            elif dat_sn.loc[i, 'dest'] == 'iot':
                samples_s = 0
            sample_list.append(samples_s)

    # merge clients data as one by averaging
    dat_sn = dat_sn.assign(para=para_list)
    dat_sn = dat_sn.assign(samples=sample_list)
    dat_sn = dat_sn.groupby(['round', 'dest'],as_index=False).mean()

    # fill if rounds missing
    if len(dat_sn['round']) < rounds: 
        extra_df = pd.DataFrame(list(range(1,rounds+1)), columns=['round'])
        dat_sn = pd.merge(dat_sn, extra_df, on='round', how="right")
        dat_sn['para'].fillna(0, inplace=True)
        dat_sn['samples'].fillna(0, inplace=True)
        destine_other = 'iot' if destine == 'ap' else 'ap'
        dat_sn['dest'].fillna(destine_other, inplace=True)

    rounds = dat_sn['round']

    # all parameters passed from IoT to AP need to be passed to server
    paralist = dat_sn['para'].to_list()
    for i in range(len(rounds)):
        if dat_sn['dest'][i] == 'ap':
            paralist[i] = paralist[i] + paralist[i-1]
    dat_sn['para'] = paralist

    # move sample size up for 1 row (the samples trained on AP is passed by IoT)
    dat_sn['samples'] = dat_sn['samples'].to_list()[1:] + [0]

    # add sample sizes to para size for later computation
    dat_sn['para'] = dat_sn['para'] + 0.5 * (dat_sn['samples']*1024/4) # KB to B to Float; 0.5 rate for concel download out

    # accumulate passed parameters (one way)
    para_acm_ap, para_acm_iot = 0, 0
    para_acm_ap_list, para_acm_iot_list = [], []
    for i in range(len(dat_sn['dest'])):
        if dat_sn.loc[i, 'dest'] == 'ap':
            para_acm_ap += dat_sn.loc[i, 'para']
        elif dat_sn.loc[i, 'dest'] == 'iot':
            para_acm_iot += dat_sn.loc[i, 'para']

        para_acm_ap_list.append(para_acm_ap)
        para_acm_iot_list.append(para_acm_iot)

    # float numbers to Migabytes (MB now)
    para_acm_ap_list = [i*4/(1024*1024) for i in para_acm_ap_list]
    para_acm_iot_list = [i*4/(1024*1024) for i in para_acm_iot_list]

    dat_sn = dat_sn.assign(para_acm_ap=para_acm_ap_list)
    dat_sn = dat_sn.assign(para_acm_iot=para_acm_iot_list)
    
    # compute latency (1 bytes = 2 bits)
    iot_latency = [(i*2/up_iot + i*2/down_iot) for i in para_acm_iot_list]
    ap_latency = [(i*2/up_ap + i*2/down_ap) for i in para_acm_ap_list]

    # compute energy Joule
    ap_energy = [i*ap_energy_comm for i in ap_latency]
    iot_energy = [i*iot_energy_comm for i in iot_latency]

    plot_dat = [para_acm_iot_list, para_acm_ap_list, iot_latency, ap_latency, iot_energy, ap_energy]

    return plot_dat, rounds