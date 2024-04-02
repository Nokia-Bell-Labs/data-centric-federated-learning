import random
import pandas as pd

def get_online_state(res, cid, server_round, mrate_min, mrate_max):
    '''This function gets ap online time and iot online time'''
    
    if mrate_min == 0 and mrate_max == 0:
        iot_online_prob = float(res.loc[res['type'] == 'iot']['coverage_prob'])
    else:
        assert(mrate_max > mrate_min, 'Conflict: Defined mobility rate Max is smaller than Min!')
        #loading mobility for the client cid
        cprofile = pd.read_csv("client_profiles/"+cid+".csv")
        
        # scalling
        prob_list = cprofile["probability"].tolist()
        A = (mrate_max-mrate_min) / (max(prob_list)-min(prob_list))
        B =  mrate_min - A*min(prob_list)
        prob_scaled = [p*A+B for p in prob_list]
        print(f"connectivity_probability: {prob_scaled}")
        
        iot_online_prob = prob_scaled[server_round%len(cprofile)] #each server round corresonds to 1 time unit of client profile file
    
    ap_online_prob = float(res.loc[res['type'] == 'ap']['coverage_prob'])
    ap_offline_interval = float(res.loc[res['type'] == 'ap']['coverage_interval(h)'])
    iot_offline_interval = float(res.loc[res['type'] == 'iot']['coverage_interval(h)'])

    ap_offline_prob = 1 - ap_online_prob
    iot_offline_prob = 1 - iot_online_prob

    ap_offline = True
    ap_time = 0 - ap_offline_interval
    while ap_offline:
        if ap_offline_prob < random.uniform(0, 1):
            ap_offline = False
        ap_time += ap_offline_interval

    iot_offline = True
    iot_time = 0 - iot_offline_interval
    while iot_offline:
        if iot_offline_prob < random.uniform(0, 1):
            iot_offline = False
        iot_time += iot_offline_interval

    return ap_time, iot_time


def get_sample_limits(res, sample_size):
    '''This function gets storage sample numbers'''
    ap_storage = float(res.loc[res['type'] == 'ap']['storage(MB)'])
    iot_storage = float(res.loc[res['type'] == 'iot']['storage(MB)'])
    ap_storage_sn = int(ap_storage * 1024 / sample_size)
    iot_storage_sn = int(iot_storage * 1024 / sample_size)

    return ap_storage_sn, iot_storage_sn

