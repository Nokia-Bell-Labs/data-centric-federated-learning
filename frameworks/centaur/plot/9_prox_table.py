import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

avg_file = 'results/acc_dyn_1cifar10_2mobi_3medium_c8|100_e3_a5_b3_g0.csv'

base = [1, 0.1, 0.01, 0.001]
grow = 3
perc = 0.01
avg_mu = ['1e-07', '2e-07', '3e-07', '4e-07', '5e-07', '6e-07', '7e-07', '8e-07', '9e-07', '1e-06', '1.1e-06', '1.2e-06']


# get all file names
grow_pos = [perc*i for i in range(1, grow+1)]
grow_minus = [i*-1 for i in grow_pos[::-1]]
grow_plus = grow_minus + [0] + grow_pos

all_mu = []
base_mu = []
for b in base:
    all_mu = all_mu + [g*b + b for g in grow_plus] # all ranged mu
    base_mu = base_mu + [b] * (grow*2 + 1) # all base mu

filename_list = []

# build file names for prox
for mu in all_mu:
    filename = 'results/acc_dynprox' + str(round(mu, 8)) + '_1cifar10_2mobi_3medium_c8|100_e3_a5_b3_g0_flpa1000_mr0|0.csv'
    filename_list.append(filename)

# build filenames for avg (tiny mu)
for mu in avg_mu:
    filename = 'results/acc_dynprox' + mu + '_1cifar10_2mobi_3medium_c8|100_e3_a5_b3_g0_flpa1000_mr0|0.csv'
    filename_list.append(filename)
all_mu = all_mu + avg_mu
base_mu = base_mu + [0]*len(avg_mu)

# read all prox
prox_mu_list = []
prox_acc_list = []
prox_bmu_list = []
for mu, bmu, filename in zip(all_mu, base_mu, filename_list):
    if os.path.isfile(filename):
        prox_acc = pd.read_csv(filename, sep=',')
        prox_acc = prox_acc['accuracy'].tolist()[1:]
        if len(prox_acc) > 95:
            prox_bmu_list.append(bmu)
            prox_mu_list.append(mu)
            prox_acc_list.append(prox_acc)

# read avg
avg_file = 'results/acc_dyn_1cifar10_2mobi_3medium_c8|100_e3_a5_b3_g0.csv'
avg_acc = pd.read_csv(avg_file, sep=',')
avg_acc = avg_acc['accuracy'].tolist()[1:]
prox_bmu_list.append(0)
prox_mu_list.append(0)
prox_acc_list.append(avg_acc)

# prox5_file = 'results/acc_dynprox5.0_1cifar10_2mobi_3medium_c8|100_e3_a5_b3_g0_flpa1000_mr0|0.csv'
# prox5_acc = pd.read_csv(prox5_file, sep=',')
# import pdb; pdb.set_trace()
# prox5_acc = avg_acc['accuracy'].tolist()[1:]
# prox_bmu_list.append(5)
# prox_mu_list.append(5)
# prox_acc_list.append(prox5_acc)

mean_list = []
smoothness_list = []
for bmu, mu, acc in zip(prox_bmu_list, prox_mu_list, prox_acc_list):
    smoothness = round(np.std(np.diff(acc))/np.abs(np.mean(np.diff(acc))),4)
    #smoothness = round(np.std(np.diff(acc)),4)
    # https://stats.stackexchange.com/questions/24607/how-to-measure-smoothness-of-a-time-series-in-r
    mean = round(np.max(acc), 6)
    mean_list.append(mean)
    smoothness_list.append(smoothness)
    # print("bmu={0:8}, mu={1:10},  mean={2:8},  smothness={3:8}".format(bmu, mu, mean, smoothness))

d = {'bmu': prox_bmu_list,
     'mu': prox_mu_list,
     'acc': mean_list,
     'smoothness': smoothness_list,
}

df = pd.DataFrame(data=d)
print(df)
print('='*50)

df = df.groupby(by=['bmu'], as_index=False).agg({'acc': ['count', 'mean', 'std'], 'smoothness': ['mean', 'std']})
print(df)
# import pdb; pdb.set_trace()





# x = list(range(100))
# #import pdb; pdb.set_trace()

# figs, col = plt.subplots(figsize=(5,5))
# col.plot(x, avg_acc, label='FedAvg')
# # col.plot(x, prox5_acc, label=r'FedProx: $\mu$=5')
# col.plot(x, prox1_acc, label=r'FedProx: $\mu$=1')
# col.plot(x, prox01_acc, label=r'FedProx: $\mu$=0.1')
# col.plot(x, prox001_acc, label=r'FedProx: $\mu$=0.01')
# col.plot(x, prox0001_acc, label=r'FedProx: $\mu$=0.001')


# col.set_xlabel('Rounds')
# col.set_ylabel('Test Accuracy')
# col.title.set_text('Centaur')
# col.legend()

# plt.savefig('imgs/prox.pdf')
