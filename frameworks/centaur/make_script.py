import sys
import os

run_all_accuracy_tests = 1
run_unbelanced_tests = 1
run_data_selection_arg_tests = 1
run_client_number_arg_tests = 1
run_mobility_tests = 1
run_fed_prox_tests = 1
run_vanilla_tests = 1

# shell script for testing encoders, classifiers, and datasets
GPUs = '7,6,5,3,2,1,4,0'
num_gpus = 8
num_cpus = 50
gpus_per_client = 0.25
fc = 8



if run_all_accuracy_tests == 1:
    destine = ['ap','iot','dyn']
    encoders = ['mobilenet', 'shufflenet', 'mnasnet', 'efficientnet']
    classifiers = ['medium', 'small', 'large']
    datasets = ['cifar10', 'cifar100', 'emnist']

    with open('run_ecd.sh', 'w') as f:
        f.write('')

    for c in classifiers:
        for d in datasets:
            for e in encoders:
                for dest in destine:
                    with open('run_ecd.sh', 'a') as f:
                        f.write("CUDA_VISIBLE_DEVICES={} python main.py --save_results=True --destine={} --dataset={} --encoder={} --classifier={} 2>&1 | tee log/{}_{}_{}_{}.txt\n".format(GPUs,dest,d,e,c,dest,d,e,c))

    print("Done with creating [run_all_accuracy_tests] scripts.")


if run_unbelanced_tests == 1:
    # shell script for testing unbelanced data
    destine = ['ap','iot','dyn']
    fl_partitioning_alpha = ['0.001', '0.01', '0.1']
    classifier = 'medium'

    with open('run_flpa.sh', 'w') as f:
        f.write('')

    for pa in fl_partitioning_alpha:
        for dest in destine:
            with open('run_flpa.sh', 'a') as f:
                f.write("CUDA_VISIBLE_DEVICES={} python main.py --save_results=True --destine={} --dataset=cifar10 --encoder=mobilenet --classifier={} --fl_partitioning_alpha={} 2>&1 | tee log/{}_flpa{}.txt\n".format(GPUs,dest,classifier,pa,dest,pa))

    print("Done with creating [run_unbelanced_tests] scripts.")


if run_data_selection_arg_tests == 1:
    # shell script for testing data selection arguments
    alpha = ['1', '3', '5', '10']
    beta = ['1', '3', '5', '10']
    beta.reverse()
    gamma = ['0', '1', '3', '5']
    classifier = 'medium'

    with open('run_ds.sh', 'w') as f:
        f.write('')

    for a,b in zip(alpha, beta):
        for g in gamma:
            with open('run_ds.sh', 'a') as f:
                f.write("CUDA_VISIBLE_DEVICES={} python main.py --save_results=True --destine=dyn --dataset=cifar10 --encoder=mobilenet --classifier={} --alpha={} --beta={} --gamma={} 2>&1 | tee log/dyn_a{}_b{}_g{}.txt\n".format(GPUs,classifier,a,b,g,a,b,g))

    print("Done with creating [run_data_selection_arg_tests] scripts.")


if run_client_number_arg_tests == 1:
    # shell script for testing the number of clients
    classifier = 'medium'
    destine = ['ap','iot','dyn']
    min_fit_clients_ratio = [0.1, 0.2, 0.5]
    min_available_clients = [10, 100, 1000]

    with open('run_client.sh', 'w') as f:
        f.write('')

    for ac in min_available_clients:
        for fc_ratio in min_fit_clients_ratio:
            min_fit_clients = int(fc_ratio*ac)
            if min_fit_clients == 1:
                continue

            num_client_gpus = 0.25
            num_client_cpus = 2
            
            # print("{}={}/{}".format(num_client_gpus,num_gpus,min_fit_clients))

            for dest in destine:
                with open('run_client.sh', 'a') as f:
                    f.write("CUDA_VISIBLE_DEVICES={} python main.py --save_results=True --classifier={} --destine={} --dataset=cifar10 --encoder=mobilenet --num_client_gpus={} --num_client_cpus={} --min_available_clients={} --min_fit_clients={} 2>&1 | tee log/{}_fc{}_ac{}.txt\n".format(GPUs,classifier,dest,num_client_gpus,num_client_cpus,ac,min_fit_clients,dest,min_fit_clients,ac))

    print("Done with creating [run_client_number_arg_tests] scripts.")
    

if run_mobility_tests == 1:
    classifier = 'medium'
    destine = ['ap','iot','dyn']
    mrate_mins = [0.1, 0.4, 0.7]
    mrate_maxs = [0.4, 0.7, 1.0]
    
    with open('run_mobility.sh', 'w') as f:
        f.write('')
        
    for mrate_min, mrate_max in zip(mrate_mins, mrate_maxs):
        for dest in destine:
            with open('run_mobility.sh', 'a') as f:
                f.write("CUDA_VISIBLE_DEVICES={} python main.py --save_results=True --classifier={} --destine={} --dataset=cifar10 --encoder=mobilenet --mrate_min={} --mrate_max={} 2>&1 | tee log/{}_mrs{}_mrl{}.txt\n".format(GPUs,classifier,dest,mrate_min,mrate_max,dest,mrate_min,mrate_max))

    print("Done with creating [run_mobility_tests] scripts.")


if run_fed_prox_tests == 1:

    fed_prox_mu = [1, 0.1, 0.01, 0.001]
    fed_prox_mu_97 = [0.97*i for i in fed_prox_mu]
    fed_prox_mu_98 = [0.98*i for i in fed_prox_mu]
    fed_prox_mu_99 = [0.99*i for i in fed_prox_mu]
    fed_prox_mu_101 = [1.01*i for i in fed_prox_mu]
    fed_prox_mu_102 = [1.02*i for i in fed_prox_mu]
    fed_prox_mu_103 = [1.03*i for i in fed_prox_mu]

    fed_prox_mu = fed_prox_mu_101 + fed_prox_mu_102 + fed_prox_mu_103 + fed_prox_mu_97 + fed_prox_mu_98 + fed_prox_mu_99

    with open('run_fedprox.sh', 'w') as f:
        f.write('')

    for mu in fed_prox_mu:
        with open('run_fedprox.sh', 'a') as f:
                f.write("CUDA_VISIBLE_DEVICES={} python main.py --save_results=True --destine=dyn --dataset=cifar10 --encoder=mobilenet --classifier=medium --fed_prox_mu={} 2>&1 | tee log/mu{}.txt\n".format(GPUs,str(mu),str(mu)))

    print("Done with creating [run_fed_prox_tests] scripts.")


if run_vanilla_tests == 1:
    destine = ['ap']
    encoders = ['mobilenet', 'shufflenet', 'mnasnet', 'efficientnet']
    classifiers = ['large']
    datasets = ['cifar10', 'cifar100', 'emnist']

    with open('run_vanilla.sh', 'w') as f:
        f.write('')

    for d in datasets:
        for e in encoders:
            for clas in classifiers:
                for dest in destine:
                    with open('run_vanilla.sh', 'a') as f:
                        if e == 'efficientnet':
                            f.write("CUDA_VISIBLE_DEVICES={} python main.py --save_results=True --vanilla=True --destine={} --dataset={} --encoder={} --classifier=large --base_rate=1 --sample_size=1 --num_rounds=200 --alpha=1000 --beta=0.001 --min_fit_clients=5 --num_client_gpus=1 2>&1 | tee log/vanilla_{}_{}_{}_small.txt\n".format(GPUs,dest,d,e,dest,d,e))
                        else:
                            f.write("CUDA_VISIBLE_DEVICES={} python main.py --save_results=True --vanilla=True --destine={} --dataset={} --encoder={} --classifier={} --base_rate=1 --sample_size=1 --num_rounds=200 --alpha=1000 --beta=0.001 --min_fit_clients={} 2>&1 | tee log/vanilla_{}_{}_{}_{}.txt\n".format(GPUs,dest,d,e,clas,fc,dest,d,e,clas))
    
    print("Done with creating [run_vanilla_tests] scripts.")
    

print("ALL DONE.")