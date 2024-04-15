import traceback
import sys
import math
import time
import gc
import os
import argparse

import numpy as np
import pandas as pd
import ray

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import flwr as fl
from flwr.common.typing import Scalar
from torchinfo import summary


from collections import OrderedDict
from typing import Dict, Callable, Optional, Tuple, List
from pathlib import Path

from hiefed.dataset_utils import get_dataloader
from measures.coverage import get_online_state, get_sample_limits

from hiefed import sensor_data

from third_party import autograd_hacks


def init_network(encoder="", classifier="", num_classes=10, pre_trained=False):
    """This function initialize the network to be used in training"""

    # define the encoder part
    if encoder == "mobilenet":
        Net = torchvision.models.mobilenet_v3_small(pretrained=pre_trained)
        encoder_output_size = Net.classifier[0].in_features

    elif encoder == "squeezenet":
        Net = torchvision.models.squeezenet1_1(pretrained=pre_trained)

    elif encoder == "shufflenet":
        Net = torchvision.models.shufflenet_v2_x0_5(pretrained=pre_trained)
        encoder_output_size = Net.fc.in_features

    elif encoder == "mnasnet":
        Net = torchvision.models.mnasnet0_5(pretrained=pre_trained)
        encoder_output_size = Net.classifier[1].in_features

    elif encoder == "efficientnet":
        Net = torchvision.models.efficientnet_b0(pretrained=pre_trained)
        encoder_output_size = Net.classifier[1].in_features
    elif encoder == "sencoder_p":
        encoder_output_size = 100
    elif encoder == "sencoder_u":
        encoder_output_size = 100
    elif encoder == "sencoder_m":
        encoder_output_size = 100        
    else:
        print(encoder)
        raise NameError("Encoder does not defined!")

    
    # define the classifier part
    if encoder == "squeezenet":
        torch.nn.init.xavier_uniform_(Net.classifier[1].weight)
    elif encoder == "sencoder_u" or encoder == "sencoder_m":
        Net_enc = sensor_data.SenCoder_UCIHAR()                
        if classifier == 'small':
            Net_clf = sensor_data.SenFier_S(num_classes=num_classes, cl_type=classifier,
                                   cl_inp_size=encoder_output_size)        
            Net = sensor_data.SenNet(encoder=Net_enc, classifier=Net_clf)
            if pre_trained:
                if encoder == "sencoder_u":
                    dataset = "motion"
                elif encoder == "sencoder_m":
                    dataset = "uci_har"
                f_dir = "./saved_models/"+dataset+"/"
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                Net.load_state_dict(torch.load(f_dir+"best_model.pt", map_location=torch.device(device)))               
            torch.nn.init.xavier_uniform_(Net.classifier.fcs[0].weight)
        else:
            Net_clf = sensor_data.SenFier_ML(num_classes=num_classes, cl_type=classifier,
                                   cl_inp_size=encoder_output_size)      
            Net = sensor_data.SenNet(encoder=Net_enc, classifier=Net_clf)
            if pre_trained:
                if encoder == "sencoder_u":
                    dataset = "motion"
                elif encoder == "sencoder_m":
                    dataset = "uci_har"
                f_dir = "./saved_models/"+dataset+"/"
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")            
                Net.load_state_dict(torch.load(f_dir+"best_model.pt", map_location=torch.device(device)))                               
            torch.nn.init.xavier_uniform_(Net.classifier.fcml1[0].weight)
            torch.nn.init.xavier_uniform_(Net.classifier.fcml2[0].weight)            
        
        # summary(Net.cuda(), input_size=(10, 1, 6, 128), device="cuda:0")        
    elif encoder == "sencoder_p":
        Net_enc = sensor_data.SenCoder_PAMAP2()                
        if classifier == 'small':
            Net_clf = sensor_data.SenFier_S(num_classes=num_classes, cl_type=classifier,
                                   cl_inp_size=encoder_output_size)        
            torch.nn.init.xavier_uniform_(Net_clf.fcs[0].weight)
        else:
            Net_clf = sensor_data.SenFier_ML(num_classes=num_classes, cl_type=classifier,
                                   cl_inp_size=encoder_output_size)        
            torch.nn.init.xavier_uniform_(Net_clf.fcml1[0].weight)
            torch.nn.init.xavier_uniform_(Net_clf.fcml2[0].weight)            
        Net = sensor_data.SenNet(encoder=Net_enc, classifier=Net_clf)
        # summary(Net.cuda(), input_size=(10, 1, 6, 128), device="cuda:0")        
    else:
        if classifier == 'small':
            NetClassifier = torch.nn.Sequential(
                            torch.nn.Linear(encoder_output_size, num_classes),
                        )
            if encoder == "shufflenet":
                Net.fc = NetClassifier
                torch.nn.init.xavier_uniform_(Net.fc[0].weight)            
            else:
                Net.classifier = NetClassifier
                torch.nn.init.xavier_uniform_(Net.classifier[0].weight)

        elif classifier == 'medium':
            NetClassifier = torch.nn.Sequential(
                            torch.nn.Linear(encoder_output_size, 64),
                            torch.nn.Dropout(p=0.2, inplace=True),
                            torch.nn.Linear(64, num_classes),
                        )
            
            if encoder == "shufflenet":
                Net.fc = NetClassifier
                torch.nn.init.xavier_uniform_(Net.fc[0].weight)
                torch.nn.init.xavier_uniform_(Net.fc[2].weight)
            else:
                Net.classifier = NetClassifier
                torch.nn.init.xavier_uniform_(Net.classifier[0].weight)
                torch.nn.init.xavier_uniform_(Net.classifier[2].weight)

        elif classifier == 'large':
            NetClassifier = torch.nn.Sequential(
                            torch.nn.Linear(encoder_output_size, 128),
                            torch.nn.Dropout(p=0.2, inplace=True),
                            torch.nn.Linear(128, num_classes),
                        )
            
            if encoder == "shufflenet":
                Net.fc = NetClassifier
                torch.nn.init.xavier_uniform_(Net.fc[0].weight)
                torch.nn.init.xavier_uniform_(Net.fc[2].weight)
            else:
                Net.classifier = NetClassifier
                torch.nn.init.xavier_uniform_(Net.classifier[0].weight)
                torch.nn.init.xavier_uniform_(Net.classifier[2].weight)


    return Net


def get_params(model: torch.nn.ModuleList) -> List[np.ndarray]:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model: torch.nn.ModuleList, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


# Compute gradient norm
def get_gradient_norm(losses, net):
    # losses for a group of samples
    # Implementation of https://github.com/idiap/importance-sampling/blob/9c9cab2ac91081ae2b64f99891504155057c09e3/importance_sampling/layers/scores.py#L117

    # set requires_grad so only gradients of the last layer
    len_net = 0
    for param in net.parameters():
        param.requires_grad = False
        len_net += 1

    idx = 0
    for param in net.parameters():
        if idx >= len_net-2:
            param.requires_grad = True
        idx += 1

    # for each sample, get the norm of gradients
    grad_norm = []
    for loss in losses:
        loss.backward(retain_graph=True)
        grad_norm.append(torch.sqrt(torch.sum(torch.square(net.classifier[2].weight.grad))).detach().cpu().numpy())
        net.classifier[2].weight.grad.data.zero_()

    # return the requires_grad
    for param in net.parameters(): param.requires_grad = True

    return grad_norm



def load_state(fed_dir, cid, state_str, hist_bins=0):
    with open(str(fed_dir / cid / (state_str + "_state.npy")), 'rb') as f:
        state = np.load(f)

    if hist_bins:
        print(f"-------------------------\nHistogram of {state_str} state")
        hist = np.histogram(state, bins=hist_bins)
        for bin in range(len(hist[0])):
            count = math.ceil(hist[0][bin] / 3)
            beg, end = hist[1][bin], hist[1][bin+1]
            print("{0:.3f}-{1:.3f}: {2}".format(beg, end, "|"*count))

    return state


def save_state(fed_dir, cid, state, state_str='loss'):
    with open(str(fed_dir / cid / (state_str + "_state.npy")), 'wb') as f:
        np.save(f, state)
        # print("{} state updated to {}".format(state_str, str(fed_dir / cid)))


# this function build the importance state array measure the importance of sample using loss
def importance_state_init(net, config, fed_dir, cid, num_workers, device: str):
    inittrainloader = get_dataloader(fed_dir, cid, is_train=True, 
                        batch_size=int(config["batch_size"]), dataset=str(config["dataset"]), workers=num_workers)
    gamma = float(config["gamma"])

    #_# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    #_# We use 'none', becuase we need per-sample loss values.  
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    #net.eval()
    net.train()

    losses = np.array([])
    norms = np.array([])
    # with torch.no_grad():
    for images, labels in inittrainloader:
        images, labels = images.to(device), labels.to(device)
        per_sample_losses = criterion(net(images), labels)
        losses = np.append(losses, per_sample_losses.detach().cpu().numpy())

        loss = torch.mean(per_sample_losses)
        if not gamma == 0: norms = np.append(norms, get_gradient_norm(per_sample_losses, net))

        del images
        del labels

    #losses = losses.tolist()
    save_state(fed_dir, cid, losses, 'loss')
    if not gamma == 0: save_state(fed_dir, cid, norms, 'norm')
    
    return len(losses)


# train on ap or iot, data selection and layer selection setup
def train_swap(net, config, cid, online_time):
    server_round=int(config["server_round"])
    alpha=float(config["alpha"])
    beta=float(config["beta"])
    gamma=float(config["gamma"])
    destine_set = str(config["destine"])
    keep_localtraining = str(config["keep_localtraining"])
    classifier = str(config["classifier"])
    base_rate = float(config["base_rate"])
    step_rate = float(config["step_rate"])

    ap_online_time, iot_online_time = online_time    

    # if destine_set == "dyn":
    # TODO: Strategy, crrently: every two rounds on ap or iot
    if server_round % 2 == 0:
        destine = "ap"

        # load cids from the buffer
        with open('log/cids_buffer.csv', 'r') as f:
            pre_cids = f.read()
            iot_storage_sn = [i.split('|')[1] for i in pre_cids.split(',')[:-1]]
            iot_storage_sn = int(iot_storage_sn[0])

            cids = [i.split('|')[0] for i in pre_cids.split(',')[:-1]]
            iot_sample_gain_rates = [i.split('|')[2] for i in pre_cids.split(',')[:-1]]
            iot_sample_gain_rate = float(iot_sample_gain_rates[cids.index(cid)])

    else:
        destine = "iot"
        iot_storage_sn = sys.maxsize

    # get encoder len and classifier len
    clas_len = 2 if classifier == 'small' else 4
    total_len = 0
    for param in net.parameters(): total_len += 1
    encd_len = total_len - clas_len

    # layer_selection for training partitioned model
    if destine == "iot":
        idx = 0
        for param in net.parameters():
            idx += 1
            if idx <= encd_len:
                param.requires_grad = False
                if idx == encd_len:
                   print(f"freezing layers til layer={idx} (out of {total_len})")
    elif destine == "ap":
        for param in net.parameters():
            param.requires_grad = True
    else:
        raise ValueError("Layer_selection function: destine does not support")

    # print(f"-----test here---{destine}----{iot_online_time}")
    epochs = int(config["epochs"])
    sample_gain_rate = base_rate

    if keep_localtraining == 'True':
        if destine == 'iot':
            # get epoch number for iot
            if not iot_online_time == 0:
                epochs += int(iot_online_time)

            # get sample_gain_rate for iot
            sample_gain_rate = base_rate + int(iot_online_time)*step_rate
            if sample_gain_rate > 1: sample_gain_rate = 1
            print(f"IoT epochs: {epochs}; sample gain rate: {base_rate} -> {sample_gain_rate}")

        elif destine == 'ap':
            # get epoch number for ap
            # if not ap_online_time == 0:
            #     epochs += int(ap_online_time)
            epochs += int((iot_sample_gain_rate - base_rate) / step_rate)
            sample_gain_rate = iot_sample_gain_rate
            print(f"AP epochs: {epochs}; sample gain rate (from iot): {base_rate} -> {sample_gain_rate}")

    data_selection = (alpha, beta, gamma), sample_gain_rate, destine, iot_storage_sn

    # on iot training only, no need to do data selection
    if destine_set == "iot" and destine == "iot":
        data_selection = None

    # training round
    bo1 = str(config["destine"]) == 'ap' and destine == 'ap'
    bo2 = str(config["destine"]) == 'iot' and destine == 'iot'
    bo3 = str(config["destine"]) == 'dyn'
    train_round_bool = bo1 or bo2 or bo3

    return destine, data_selection, epochs, sample_gain_rate, train_round_bool




def train(net, config, online_time, fed_dir, cid, num_workers, device: str):
    """This train function invoked by clients' fit function."""

    destine, data_selection, epochs, sample_gain_rate, train_round_bool = train_swap(net, config, cid, online_time)
    len_traindata_local = []
    server_params = get_params(net)

    if not train_round_bool:
        len_traindata_local = [450]
        print(f"Train_round_bool: {train_round_bool}, Skipping...")
        time.sleep(5)
    else:
        print(f"Train_round_bool: {train_round_bool}, Training...")
        criterion_none = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        net.train()

        for ep in range(epochs):
            # get train dataset based on data selection
            # trainloader is updated every epoch, to reselect data points
            trainloader = get_dataloader(fed_dir, cid, is_train=True,
                data_selection=data_selection, batch_size=int(config["batch_size"]), dataset=str(config["dataset"]), workers=num_workers)

            gamma = float(config['gamma'])
            fed_prox_mu = float(config['fed_prox_mu'])

            # get index of training samples and loss state of all samples
            sample_index = trainloader.dataset.get_index()
            len_traindata_local.append(len(sample_index)) # append the len of dataset
            loss_state = load_state(fed_dir, cid, 'loss')
            if not gamma == 0: norm_state = load_state(fed_dir, cid, 'norm')

            idx = 0
            for images, labels in trainloader:
                # forward
                images, labels = images.to(device), labels.to(device)
                if len(labels) == 1: continue # note: skip one-input-only batch, because batchnorm will panic

                optimizer.zero_grad()

                per_sample_losses = criterion_none(net(images), labels)
                losses = per_sample_losses.detach().cpu().numpy()
                if not gamma == 0: norms = get_gradient_norm(per_sample_losses, net)
                for i in range(len(labels)):
                    loss_state[sample_index[idx]] = losses[i]
                    if not gamma == 0: norm_state[sample_index[idx]] = norms[i]
                    idx += 1

                # backward
                if fed_prox_mu < 0.00001:
                    loss = torch.mean(per_sample_losses)
                
                else:
                    proximal_term = 0.0
                    local_params = get_params(net)
                    for i in range(len(server_params)):
                        w = local_params[i].flatten()
                        w_t = server_params[i].flatten()
                        proximal_term += np.linalg.norm(w-w_t)

                    print(f"Proximal Term is {proximal_term}, checking with mu={fed_prox_mu}")
                    loss = torch.mean(per_sample_losses) + fed_prox_mu * proximal_term  

                loss.backward()
                optimizer.step()

                del loss
                del images
                del labels

            # save the updated loss state
            save_state(fed_dir, cid, loss_state, 'loss')
            if not gamma == 0: save_state(fed_dir, cid, norm_state, 'norm')

    return len_traindata_local, destine, sample_gain_rate, train_round_bool



def test(net, testloader, device: str):
    criterion = torch.nn.CrossEntropyLoss()
    correct_sum, total, loss_sum = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)            
            outputs = net(images)            
            loss = criterion(outputs, labels)
            loss_sum += loss.cpu().detach().item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct = (predicted == labels).sum()
            correct_sum += correct.cpu().detach().item()
            del loss
            del correct
            del images
            del labels
    accuracy = correct_sum / total
    return loss_sum, accuracy


# Flower client that will be spawned by Ray
# Adapted from Pytorch quickstart example
class RayClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_dir_data: str, saver, encoder, classifier, num_classes, pre_trained, res_configs):
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

        # instantiate model
        self.net = init_network(encoder=encoder, classifier=classifier, num_classes=num_classes, pre_trained=pre_trained)

        # determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # setup saver and resource configs
        self.saver = saver
        self.encoder = encoder
        self.res_configs = res_configs


    def get_resourse_requires(self, config, full=False):
        res = pd.read_csv(self.res_configs, sep=',')
        
        # number of samples can be saved and trained
        sample_size = float(config['sample_size'])
        ap_storage_sn, iot_storage_sn = get_sample_limits(res, sample_size)

        # when it is online
        server_round = int(config['server_round'])
        mrate_min = float(config['mrate_min'])
        mrate_max = float(config['mrate_max'])
        ap_online_time, iot_online_time = get_online_state(res, self.cid, server_round, mrate_min, mrate_max)

        # TODO: not fully implementated now
        # if full: 
        #     # get training latency and energy per sample training
        #     mac_latency_energy = get_mac_latency_energy_per(res, self.encoder)
        #     ap_latency_mac, iot_latency_mac, ap_energy_mac, iot_energy_mac = mac_latency_energy

        #     # get communication latency and energy per communication round (model only)
        #     comm_latency_energy = get_comm_latency_energy_per(res, self.encoder)
        #     ap_latency_comm, iot_latency_comm, ap_energy_comm, iot_energy_comm = comm_latency_energy

        return ap_storage_sn, iot_storage_sn, [ap_online_time, iot_online_time]


    def get_parameters(self, config):
        return get_params(self.net)


    def fit(self, parameters, config):
        gc.collect()
        torch.cuda.empty_cache()

        if not self.res_configs == None:
            _, iot_storage_sn, online_time = self.get_resourse_requires(config)

        print(f"------fit() on client cid={self.cid}------")

        # self.set_parameters(parameters)
        set_params(self.net, parameters)

        # load data for this client and get trainloader
        if config['debug'] == 'False':
            num_workers = len(ray.worker.get_resource_ids()["CPU"])
        else:
            num_workers = 2

        # send model to device
        self.net.to(self.device)

        # initialize sample importance
        importance_state_init(self.net, config, self.fed_dir, self.cid, num_workers=num_workers, device=self.device)

        # train
        len_traindata, destine, sample_gain_rate, train_round_bool  = train(self.net, config, online_time, self.fed_dir, self.cid, num_workers=num_workers, device=self.device)

        # buffer limits of sample numbers for iot
        if destine == 'iot': 
            self.saver.cids_buffer_append(self.cid, iot_storage_sn, sample_gain_rate)

        # save the number of samples
        if train_round_bool:
            self.saver.sample_number_saver(destine, self.cid, len_traindata, config)

        # return local model and statistics
        return get_params(self.net), len_traindata[0], {}


    def evaluate(self, parameters, config):
        try:
            print(f"evaluate() on client cid={self.cid}")
            set_params(self.net, parameters)

            # load data for this client and get trainloader
            num_workers = 4
            valloader = get_dataloader(
                self.fed_dir, self.cid, is_train=False, batch_size=50, dataset=str(config['dataset']), workers=num_workers
            )

            # send model to device
            self.net.to(self.device)

            # evaluate
            loss, accuracy = test(self.net, valloader, device=self.device)
            
            # restart cids buffer
            if int(config['server_round']) % 2 == 0:
                self.saver.cids_buffer_reset()

            # return statistics
            return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}
    
        except Exception as e:
            print(f"{e}")
            print(traceback.format_exc())
            print(sys.exc_info()[2])