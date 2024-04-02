import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.cm as cm

from plot_utils import y_fmt, linestyle_tuple, load_acc_pure, load_all_acc, load_mac_latency_energy, load_comm_latency_energy, load_res_from_cfg

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
buff_file = 'results/mrate_plot_buff.csv'

if not os.path.isfile(buff_file):
    # Load accuracy
    df_acc = load_all_acc(res_folder, focus='mr')

    df_acc['iot_latency'] = 0
    df_acc['ap_latency'] = 0
    df_acc['iot_energy'] = 0
    df_acc['ap_energy'] = 0

    # Load cost
    dests = ['ap', 'iot', 'dyn']

    mr1_list = ['0.1', '0.4', '0.7']
    mr2_list = ['0.4', '0.7', '1.0']
    post_args = [('1cifar10_2mobi_3medium_c8|100_e3_a5_b3_g0_flpa1000_mr' + mr1 + '|' + mr2) for mr1, mr2 in zip(mr1_list, mr2_list)]

    df_acc['mr1'] = mr1_list * 3
    df_acc['mr2'] = mr2_list * 3

    # load configuration from cfg file
    computation_cfg, communication_cfg = load_res_from_cfg()
    iot_freq, ap_freq, count_inference, iot_energy_mac, ap_energy_mac = computation_cfg
    up_iot, down_iot, up_ap, down_ap, sample_size, iot_energy_comm, ap_energy_comm = communication_cfg

    for dest in dests:
        for post_arg, mr1, mr2 in zip(post_args, mr1_list, mr2_list):
            # load and compute the latency and energy cost
            mac_dyn, mac_ap, mac_iot, _, _, _ = load_mac_latency_energy(encoder,classifier, post_arg, iot_freq, ap_freq, iot_energy_mac, ap_energy_mac, dests=dest)
            comm_dyn, comm_ap, comm_iot, _, _, _ = load_comm_latency_energy(encoder,classifier, post_arg, up_iot, down_iot, up_ap, down_ap, iot_energy_comm, ap_energy_comm, sample_size, dests=dest)

            sum_x = []
            for k in range(2, 6):
                if mac_dyn is not None: sum_x.append([i+j for i,j in zip(mac_dyn[k], comm_dyn[k])])
                if mac_ap is not None: sum_x.append([i+j for i,j in zip(mac_ap[k], comm_ap[k])])
                if mac_iot is not None: sum_x.append([i+j for i,j in zip(mac_iot[k], comm_iot[k])])

            # put the cost into the table
            bool_ = (df_acc['dest']==dest) & (df_acc['mr1']==mr1) & (df_acc['mr2']==mr2)
            df_acc.loc[bool_, 'iot_latency'] = max(sum_x[0])
            df_acc.loc[bool_, 'ap_latency'] = max(sum_x[1])
            df_acc.loc[bool_, 'iot_energy'] = max(sum_x[2])
            df_acc.loc[bool_, 'ap_energy'] = max(sum_x[3])

    df_acc['key'] = '$\lambda$=' + df_acc['mr1'] + '~'+df_acc['mr2']

    df_acc.to_csv(buff_file)

else:
    print('loading existing buff !')
    df_acc = pd.read_csv(buff_file, sep=',')

ap_acc_up = [i-j for i,j in zip(df_acc.loc[df_acc['dest'] == 'dyn', 'accuracy'].tolist(), df_acc.loc[df_acc['dest'] == 'ap', 'accuracy'].tolist())]
iot_acc_up = [i-j for i,j in zip(df_acc.loc[df_acc['dest'] == 'dyn', 'accuracy'].tolist(), df_acc.loc[df_acc['dest'] == 'iot', 'accuracy'].tolist())]

ap_cost_down = [(j-i)/j for i,j in zip(df_acc.loc[df_acc['dest'] == 'dyn', 'ap_latency'].tolist(), df_acc.loc[df_acc['dest'] == 'ap', 'ap_latency'].tolist())]
iot_cost_down = [(j-i)/j for i,j in zip(df_acc.loc[df_acc['dest'] == 'dyn', 'iot_latency'].tolist(), df_acc.loc[df_acc['dest'] == 'iot', 'iot_latency'].tolist())]

dat_ = {'key': df_acc['key'].tolist()[:3],
        'ap_acc_up': ap_acc_up,
        'iot_acc_up': iot_acc_up,
        'ap_cost_down': ap_cost_down,
        'iot_cost_down': iot_cost_down,
        }
print(pd.DataFrame(data = dat_))



# plot
figs, axes = plt.subplots(nrows=1, ncols=2, figsize=(5,3))
plt.subplots_adjust(left=0.11, right=0.89, wspace=0.4, bottom=0.15, top=0.9)

x = df_acc['accuracy'].tolist()
len_ = len(x)
txt = df_acc['key'].tolist()
dest = df_acc['dest'].tolist()

anno_text_size = 6
point_size = 30
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)


def plot_arrow(ax, x_, y_, x_dyn, y_dyn):
    # x_, y_ -> x_dyn, y_dyn
    for i in range(len(x_)):
        xpos = (x_[i]+x_dyn[i])/2
        ypos = (y_[i]+y_dyn[i])/2
        xdir = x_dyn[i] - x_[i]
        ydir = y_dyn[i] - y_[i]
        
        # X,Y,dX,dY = xpos, ypos, xdir, ydir
        # ax.annotate("", xytext=(xpos,ypos),xy=(xpos+0.001*xdir,ypos+0.001*ydir), arrowprops=dict(arrowstyle="->", color='black'), size = 8)
        ax.plot([x_[i], x_dyn[i]], [y_[i], y_dyn[i]], color='black', linewidth=0.3, linestyle=(0, (5, 10)))


for idx in range(len(axes)):
    col = axes[idx]
    if idx == 0:
        y1 = df_acc['iot_latency'].tolist()
        y2 = df_acc['iot_energy'].tolist()
    elif idx == 1:
        y1 = df_acc['ap_latency'].tolist()
        y2 = df_acc['ap_energy'].tolist()

    # import pdb; pdb.set_trace()
    
    idx_ap = [i for i, x in enumerate(dest) if x == "ap"]
    idx_iot = [i for i, x in enumerate(dest) if x == "iot"]
    idx_dyn = [i for i, x in enumerate(dest) if x == "dyn"]
    
    x_dyn = [x[i] for i in idx_dyn]
    y_dyn = [y1[i] for i in idx_dyn]
    if max(y_dyn) > 1000: 
        plt1 = col.scatter(x_dyn, y_dyn, s=point_size, color='r', marker='s')
        #texts = [col.text(x[i], y1[i], txt[i], size=6) for i in idx_iot]
        #adjust_text(texts, ax=col)
    
    x_ap = [x[i] for i in idx_ap]
    y_ap = [y1[i] for i in idx_ap]
    if max(y_ap) > 1000: 
        plot_arrow(col, x_ap, y_ap, x_dyn, y_dyn)
        plt2 = col.scatter(x_ap, y_ap, s=point_size, color='b', marker='d')
        #texts = [col.text(x[i], y1[i], txt[i], size=6) for i in idx_ap]
        #adjust_text(texts, ax=col, ha='center',va='center', autoalign=False, expand_points=(1.5,2))
        col.text(0.81, 1520, r'$\lambda$=0.1~0.4', size=anno_text_size, bbox=props)
        col.text(0.803, 1250, r'$\lambda$=0.4~0.7', size=anno_text_size, bbox=props)
        col.text(0.835, y1[2], r'$\lambda$=0.7~1.0', size=anno_text_size, bbox=props)
        
        col.legend([plt2, plt1], ['AP Training','Centaur'])
        

    x_iot = [x[i] for i in idx_iot]
    y_iot = [y1[i] for i in idx_iot]
    if max(y_iot) > 1000: 
        plot_arrow(col, x_iot, y_iot, x_dyn, y_dyn)
        plt3 = col.scatter(x_iot, y_iot, s=point_size, color='y', marker='<')
        #texts = [col.text(x[i], y1[i], txt[i], size=6) for i in idx_iot]
        #adjust_text(texts, ax=col)
        col.text(0.80, 95000, r'$\lambda$=0.1~0.4', size=anno_text_size, bbox=props)
        col.text(0.81, 67500, r'$\lambda$=0.4~0.7', size=anno_text_size, bbox=props)
        col.text(0.795, 35000, r'$\lambda$=0.7~1.0', size=anno_text_size, bbox=props)

        col.legend([plt3, plt1], ['UCD Training','Centaur'],loc="center right", bbox_to_anchor=(1,0.76))
        


    # adjust_text(texts, ax=col, arrowprops=dict(arrowstyle="-", color='k', lw=0.5), avoid_text=True) # avoid_self=True, 
    # col.invert_xaxis()
    col.set_xlim(0.79, 0.91)
    #col.set_xticks(np.arange(0.92, 0.80, 0.02))
    col.yaxis.set_major_formatter(FuncFormatter(y_fmt))
    col.set_xlabel('Test Accuracy')
    if idx==0: col.set_title('Workload on UCDs')
    if idx==0: col.set_ylabel('Accumulated Latency (Seconds)')


    col_tx = col.twinx()
    
    x_ = [x[i] for i in idx_ap]
    y_ = [y2[i] for i in idx_ap]
    if max(y_) > 1000: col_tx.scatter(x_, y_, s=1, color='b',marker='')
    
    x_ = [x[i] for i in idx_iot]
    y_ = [y2[i] for i in idx_iot]
    if max(y_) > 1000: col_tx.scatter(x_, y_, s=1, color='y',marker='')
    
    x_ = [x[i] for i in idx_dyn]
    y_ = [y2[i] for i in idx_dyn]
    if max(y_) > 1000: col_tx.scatter(x_, y_, s=1, color='r',marker='')
    
    #idx_iot = [i for i, x in enumerate(dest) if x == "iot"]
    #col.scatter(x[idx_iot], y1[idx_iot], s=8, colors='y')
    

    #col_tx.set_xlim(0.79, 0.91)
    #col.set_xticks(np.arange(0.92, 0.80, 0.02))
    col_tx.yaxis.set_major_formatter(FuncFormatter(y_fmt))
    col_tx.set_xlabel('Test Accuracy')
    if idx==1: col_tx.set_title('Workload on APs')
    if idx==1: col_tx.set_ylabel('Accumulated Energy (Joule)')

plt.savefig('imgs/mobility.pdf')