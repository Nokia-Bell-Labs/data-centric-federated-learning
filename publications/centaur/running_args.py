import argparse
from typing import Dict, Callable, Optional, Tuple, List
import flwr as fl
from flwr.common.typing import Scalar
import torch
import torchvision
from hiefed.client import get_params, set_params, RayClient, init_network, test

import os
import numpy as np

###
# This file is used for setting up the experiments
###

def get_parser():
    parser = argparse.ArgumentParser(description="Flower Simulation of Centaur with PyTorch")

    parser.add_argument("--num_client_cpus", type=float, default=4, help="the number of CPUs for each client")
    parser.add_argument("--num_client_gpus", type=float, default=1, help="the number of GPUs for each client")
    parser.add_argument("--num_rounds", type=int, default=100, help="the total number of FL training rounds")
    parser.add_argument("--num_epochs", type=int, default=3, help="the number of basic local epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="the size of local batch")
    parser.add_argument("--sample_size", type=int, default=30, help="the average size of one local sample ()in KB")

    parser.add_argument("--min_fit_clients", type=int, default=8, help="the number of participating clients in each round")
    parser.add_argument("--min_available_clients", type=int, default=20, help="the number of total clients")

    parser.add_argument("--dataset", type=str, default="pamap", help="the dataset to use [cifar10], [cifar100], [emnist], [pamap], [uci_har], or [motion]")
    parser.add_argument("--encoder", type=str, default="sencoder_p", help="the encoder to use [mobilenet], [efficientnet], [shufflenet], [mnasnet], [sencoder_p], [sencoder_u], or [sencoder_m]")
    parser.add_argument("--classifier", type=str, default="medium", help="the classifier size to use [small], [medium], or [large]")
    parser.add_argument("--pre_trained", type=bool, default=False, help="whether the encoder's weights are pre-trained [True] or [False]")

    parser.add_argument("--debug", type=bool, default=False, help="debug mode [True] or [False]")
    parser.add_argument("--vanilla", type=bool, default=False, help="whether to start vanilla training mode [True] or [False]")
    parser.add_argument("--save_results", type=bool, default=True, help="whether to save results or not [True] or [False]")
    parser.add_argument("--save_models", type=bool, default=False, help="whether to save the best trained model or not [True] or [False]")

    parser.add_argument("--fed_prox_mu", type=float, default=0, help="The mu parameter in FedProx")
    parser.add_argument("--mrate_min", type=float, default=0, help="the lowest connection rate of the range in mobility model")
    parser.add_argument("--mrate_max", type=float, default=0, help="the highest connection rate of the range in mobility model")
    
    parser.add_argument("--fl_partitioning_alpha", type=float, default=1000, help="the LDA-alpha parameter used to partition dataset in FL (LDA)")
    parser.add_argument("--alpha", type=float, default=5, help="alpha value in data selection (for dropping, the smaller this number is, the larger portion of data it gets)")
    parser.add_argument("--beta", type=float, default=3, help="beta value in data selection (for access point)")
    parser.add_argument("--gamma", type=float, default=0, help="gamma value in data selection (for norm-based selection)")

    parser.add_argument("--keep_localtraining", type=bool, default=True, help="keep training going when IoT offline [True] or [False]")
    parser.add_argument("--base_rate", type=float, default=.5, help="the basic rate of local data used to training")
    parser.add_argument("--step_rate", type=float, default=.2, help="the step rate of local data used to training when offline")

    parser.add_argument("--destine", type=str, default="dyn", help="[dyn]=our Centaur training; [ap]=access point training; [iot]=ultra-constrained edge device training")
    parser.add_argument("--client_configs", type=str, default="client_configurations.csv", help="file directory of client configuration")

    return parser



def fit_config(server_round: int) -> Dict[str, Scalar]:
    """
    This function will be used for the server to pass new configuration values
    to the client at each round. This function will be called by the chosen strategy 
    and must return a dictionary of configuration key values pairs.
    """    
    args = get_parser().parse_args()
    config = {
        "server_round": str(server_round),
        "epochs": str(args.num_epochs),
        "batch_size": str(args.batch_size),
        "sample_size": str(args.sample_size),
        "alpha": str(args.alpha),
        "beta": str(args.beta),
        "gamma": str(args.gamma),
        "destine": str(args.destine),
        "dataset": str(args.dataset),
        "fed_prox_mu": str(args.fed_prox_mu),
        "mrate_min": str(args.mrate_min),
        "mrate_max": str(args.mrate_max),
        "keep_localtraining": str(args.keep_localtraining),
        "classifier": str(args.classifier),
        "base_rate": str(args.base_rate),
        "step_rate": str(args.step_rate),
        "debug": str(args.debug),
    }
    return config



def get_evaluate_fn(
    testset, saver, num_classes,
) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """
    All built-in strategies support centralized evaluation by providing an evaluation function
    during initialization. An evaluation function is any function that can take 
    the current global model parameters as input and return evaluation results.
    """
    parser = get_parser()
    args = get_parser().parse_args()
    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, float]]:
        """Use the entire test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = init_network(encoder=args.encoder, classifier=args.classifier, num_classes=num_classes, pre_trained=args.pre_trained)
        set_params(model, parameters)
        model.to(device)

        testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        loss, accuracy = test(model, testloader, device=device)

        # save the accuracy
        saver.accuracy_saver(server_round, accuracy, parser.parse_args().destine)

        if args.save_models:
            f_dir = "./saved_models/"+args.dataset+"/"
            if os.path.exists(f_dir):            
                best_acc = np.load(f_dir+"best_acc.npy")
                if accuracy > best_acc:
                    np.save(f_dir+"best_acc.npy", accuracy)
                    torch.save(model.state_dict(), f_dir+"best_model.pt")
                    print("##### Saved Accuracy:", accuracy, best_acc)
            else:
                os.makedirs(f_dir)
                np.save(f_dir+"best_acc.npy", accuracy)


        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate