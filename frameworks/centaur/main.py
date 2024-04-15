import argparse

import ray
import torch
import torchvision

from collections import OrderedDict
from typing import Dict, Callable, Optional, Tuple, List
from pathlib import Path
import numpy as np
import flwr as fl
from flwr.common.typing import Scalar

from hiefed.fed_criterion import FedAvg_criterion
from hiefed.dataset_utils import getCIFAR10, getCIFAR100, do_fl_partitioning, Saver,  getEMNIST, getUCIHAR, getPAMAP2, getMotion
from hiefed.client import get_params, set_params, RayClient, init_network, test

from oam import create_client_profile

import running_args
from running_args import fit_config
from running_args import get_evaluate_fn
from torchinfo import summary


# Start Ray simulation (a _default server_ will be created)
# This example does:
# 1. Downloads CIFAR-10, CIFAR100-100, or EMNIST
# 2. Partitions the dataset into N splits, where N is the total number of
#    clients. We refere to this as `pool_size`. The partition can be IID or non-IID
# 3. Starts a Ray-based simulation where a % of clients are sample each round.
# 4. After the M rounds end, the global model is evaluated on the entire testset.
#    Also, the global model is evaluated on the valset partition residing in each
#    client. This is useful to get a sense on how well the global model can generalise
#    to each client's data.
if __name__ == "__main__":

    # parse input arguments
    args = running_args.get_parser().parse_args()

    client_resources = {
        "num_gpus": args.num_client_gpus,
        "num_cpus": args.num_client_cpus,
    } 

    # download dataset
    if args.dataset == 'cifar10':
        train_path, testset = getCIFAR10()
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_path, testset = getCIFAR100()
        num_classes = 100
    elif args.dataset == 'emnist':
        train_path, testset = getEMNIST()             
        num_classes = 47
    elif args.dataset == 'pamap':
        train_path, testset = getPAMAP2()                
        num_classes = 13
    elif args.dataset == 'uci_har':
        train_path, testset = getUCIHAR()                
        num_classes = 6
    elif args.dataset == 'motion':
        train_path, testset = getMotion()                
        num_classes = 6
    # TODO: elif args.dataset == 'femnist':
    #     train_path, testset = getFEMNIST()
    #     data_loader = torch.utils.data.DataLoader(train_path, batch_size=len(train_path))    
    #     train_images, train_labels = iter(data_loader).next()
    #     print(train_images.shape, train_labels.shape)
    #     labels = np.array(train_labels).astype(int)
    #     _unique, _counts = np.unique(labels, return_counts=True)
    #     print(np.asarray((_unique, _counts)).T)
    #     num_classes = 10
    else:
        raise NameError('Dataset does not supported!')

    # partition dataset (use a large `alpha` to make it IID;
    # a small value (e.g. 1) will make it non-IID)
    # This will create a new directory called "federated": in the directory where
    # CIFAR-10 lives. Inside it, there will be N=pool_size sub-directories each with
    # its own train/set split.
    fed_dir = do_fl_partitioning(
        train_path, pool_size=args.min_available_clients, 
        alpha=args.fl_partitioning_alpha, 
        num_classes=num_classes, val_ratio=0.1
    )
    model = init_network(encoder=args.encoder, classifier=args.classifier,                          
                         num_classes=num_classes, pre_trained=args.pre_trained)
    if args.dataset == 'pamap':
        summary(model.cuda(), input_size=(1, 1, 27, 200), device="cuda:0")    
    elif args.dataset == 'uci_har' or args.dataset == 'motion':
        summary(model.cuda(), input_size=(1, 1, 9, 128), device="cuda:0")    
    # Get model weights as a list of NumPy ndarray's
    weights = get_params(model)

    # Serialize ndarrays to `Parameters`
    parameters = fl.common.ndarrays_to_parameters(weights)

    # when debug mode is on, choose only two clients
    if args.debug:
        args.min_fit_clients = 2
        args.save_results = False

    # initial saver
    saver = Saver(args)
    
    # create client profiles for spatio-temporal coverage
    create_client_profile(num_clients = args.min_available_clients, temporal = "day", spatial = 10)


    # configure the strategy
    client_ratio = args.min_fit_clients/args.min_available_clients
    strategy = FedAvg_criterion(
        fraction_fit=client_ratio,
        fraction_evaluate=client_ratio,
        min_fit_clients=args.min_fit_clients,
        min_available_clients=args.min_available_clients,  # All clients should be available
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(testset, saver, num_classes),  # centralised testset evaluation of global model
        initial_parameters=parameters,
    )

    def client_fn(cid: str):
        # create a single client instance
        return RayClient(cid, fed_dir, saver, 
                            args.encoder, args.classifier, num_classes, args.pre_trained, args.client_configs)

    # (optional) specify ray config
    ray_config = {"include_dashboard": False,
                    "local_mode": args.debug, # local mode on for ray
                }

    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.min_available_clients,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        ray_init_args=ray_config,
    )
