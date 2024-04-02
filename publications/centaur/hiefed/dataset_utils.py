from pathlib import Path
import numpy as np
import random
import time
import os

from itertools import compress

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import shutil
from PIL import Image
from torchvision.datasets import VisionDataset
from typing import Callable, Optional, Tuple, Any

from hiefed.data_selection import CDFSelection
from hiefed import sensor_data

from third_party.dataset_partition_flwr import create_lda_partitions


def get_dataset(path_to_data: Path, cid: str, partition: str, data_selection, dataset='cifar10'):

    # generate path to cid's data
    path_to_data = path_to_data / cid
    if dataset == 'cifar10' or dataset == 'cifar100':
        transform = cifar10Transformation()
    elif dataset == 'emnist':
        transform = emnistTransformation()
    elif dataset == 'uci_har':
        transform = uciharTransformation()
    elif dataset == 'motion':
        transform = motionTransformation()
    elif dataset == 'pamap':
        transform = pamapTransformation()
    else:
        raise NameError('Defined transformation error!')

    if dataset == 'uci_har' or dataset == 'pamap' or dataset == 'motion':        
        return TorchSensor_FL(path_to_data, partition, data_selection=data_selection, transform=transform)
    return TorchVision_FL(path_to_data, partition, data_selection=data_selection, transform=transform)


def get_dataloader(
    path_to_data: str, cid: str, is_train: bool, batch_size: int, dataset:str, workers: int,
    data_selection=None
):
    """Generates trainset/valset object and returns appropiate dataloader."""
    partition = "train" if is_train else "val"
    dataset = get_dataset(Path(path_to_data), cid, partition, data_selection, dataset)

    # we use as number of workers all the cpu cores assigned to this actor
    kwargs = {"num_workers": workers, "pin_memory": True, "drop_last": False}
    return DataLoader(dataset, batch_size=batch_size, **kwargs)


def get_random_id_splits(total: int, val_ratio: float, shuffle: bool = True):
    """splits a list of length `total` into two following a
    (1-val_ratio):val_ratio partitioning.

    By default the indices are shuffled before creating the split and
    returning.
    """

    if isinstance(total, int):
        indices = list(range(total))
    else:
        indices = total

    split = int(np.floor(val_ratio * len(indices)))
    # print(f"Users left out for validation (ratio={val_ratio}) = {split} ")
    if shuffle:
        np.random.shuffle(indices)
    return indices[split:], indices[:split]


def do_fl_partitioning(path_to_dataset, pool_size, alpha, num_classes, val_ratio=0.0):
    """Torchvision (e.g. CIFAR-10) datasets using LDA."""

    images, labels = torch.load(path_to_dataset)
    idx = np.array(range(len(images)))
    dataset = [idx, labels]
    # create_lda_partitions function https://github.com/adap/flower/blob/525f98d78618b315999a88f3ae68415305df6558/src/py/flwr/dataset/utils/common.py#L376
    partitions, _ = create_lda_partitions(
        dataset, num_partitions=pool_size, concentration=alpha, accept_imbalanced=True
    )

    # Show label distribution for first partition (purely informative)
    partition_zero = partitions[0][1]
    hist, _ = np.histogram(partition_zero, bins=list(range(num_classes + 1)))
    print(
        f"Class histogram for 0-th partition (alpha={alpha}, {num_classes} classes): {hist}"
    )

    # now save partitioned dataset to disk
    # first delete dir containing splits (if exists), then create it
    splits_dir = path_to_dataset.parent / "federated"
    if splits_dir.exists():
        shutil.rmtree(splits_dir)
    Path.mkdir(splits_dir, parents=True)

    for p in range(pool_size):

        labels = partitions[p][1]
        image_idx = partitions[p][0]
        imgs = images[image_idx]

        # create dir
        Path.mkdir(splits_dir / str(p))

        if val_ratio > 0.0:
            # split data according to val_ratio
            train_idx, val_idx = get_random_id_splits(len(labels), val_ratio)
            val_imgs = imgs[val_idx]
            val_labels = labels[val_idx]

            with open(splits_dir / str(p) / "val.pt", "wb") as f:
                torch.save([val_imgs, val_labels], f)

            # remaining images for training
            imgs = imgs[train_idx]
            labels = labels[train_idx]

        with open(splits_dir / str(p) / "train.pt", "wb") as f:
            torch.save([imgs, labels], f)

    return splits_dir


def cifar10Transformation():

    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


def emnistTransformation():

    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((0.1751), (0.3332)) # for the balanced split
        ]
    )

def uciharTransformation():

    return transforms.Compose(
        [            
            transforms.ToTensor()            
        ]
    )

def motionTransformation():

    return transforms.Compose(
        [            
            transforms.ToTensor()            
        ]
    )


def pamapTransformation():

    return transforms.Compose(
        [            
            transforms.ToTensor()            
        ]
    )


# def femnistTransformation():

#     return transforms.Compose(
#         [
#             # transforms.Resize(224),
#             transforms.ToTensor(),
#             transforms.Normalize((0.13107656), (0.28947565))            
#         ]
#     )


""" Data Selection to AP, IoT """
def dataSelection(data, targets, path_to_client, data_selection):

    cdf_args, sample_gain_rate, destine, sample_limits = data_selection
    alpha, beta, gamma = cdf_args

    # Loss State CDF-based selection
    path_to_loss_state = path_to_client / "loss_state.npy"
    with open(str(path_to_loss_state), 'rb') as f:
        loss_state = np.load(f)
    loss_selec = CDFSelection(alpha, beta, full_queue=loss_state)

    # sample gain considering spatio-temporal coverage
    sample_num = int(len(targets) * sample_gain_rate)
    data = data[:sample_num]
    targets = targets[:sample_num]

    # select sample based on its loss
    drop_list, ap_list = [], []
    for idx in range(len(targets)):
        prob_drop, prob_ap = loss_selec.query_prob(loss_state[idx])
        drop_bool = True if prob_drop > random.uniform(0, 1) else False
        ap_bool = True if prob_ap > random.uniform(0, 1) else False
        drop_list.append(drop_bool)
        ap_list.append(ap_bool)

    # Norm State CDF-based selection if required
    if not gamma == 0:
        path_to_norm_state = path_to_client / "norm_state.npy"
        with open(str(path_to_norm_state), 'rb') as f:
            norm_state = np.load(f)
        norm_selec = CDFSelection(0, gamma, full_queue=norm_state)

        iot_list = np.logical_not(np.logical_or(np.array(drop_list), np.array(ap_list)))
        iot_index = list(compress(range(len(iot_list)), iot_list))

        # reselect if based on its norm if required
        for idx in iot_index:
            _, prob_ap = norm_selec.query_prob(norm_state[idx])
            ap_bool = True if prob_ap > random.uniform(0, 1) else False
            ap_list[idx] = ap_bool

    # get the index list (for ap or iot)
    if destine == "ap":
        dest_index = list(compress(range(len(ap_list)), ap_list))
        if sample_limits < len(dest_index):
            dest_index = [dest_index[i] for i in sorted(random.sample(range(len(dest_index)), sample_limits))]

    elif destine == "iot":
        iot_list = np.logical_not(np.logical_or(np.array(drop_list), np.array(ap_list)))
        dest_index = list(compress(range(len(iot_list)), iot_list))

    else:
        raise Exception('destine not supported [ap or iot]')    
    data = data[dest_index]        
    targets = targets[dest_index]
    index = dest_index
    print(f"data selection done with dest: {destine}, sample_sum: {len(targets)}")

    return data, targets, index

class TorchSensor_FL(VisionDataset):
    """This is just a trimmed down version of torchvision.datasets.MNIST.

    Use this class by either passing a path to a torch file (.pt)
    containing (data, targets) or pass the data, targets directly
    instead.
    """
    def __init__(
        self,
        path_to_client=None,
        partition=None,
        data=None,
        targets=None,
        loss_state=None,
        data_selection=None,
        transform: Optional[Callable] = None,
    ) -> None:
        path_to_data = path_to_client / (partition + ".pt")
        path = path_to_data.parent if path_to_data else None
        super(TorchSensor_FL, self).__init__(path, transform=transform)
        self.transform = transform

        if path_to_data:
            # load data and targets (path_to_data points to an specific .pt file)
            self.data, self.targets = torch.load(path_to_data)
        else:
            self.data = data
            self.targets = targets

        # initialize the index
        self.index = list(range(len(self.targets)))

        # Data Selection
        if data_selection is not None:
            data, targets, index = dataSelection(self.data, self.targets, path_to_client, data_selection)
            self.data = data
            self.targets = targets
            self.index = index


    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        ts_data, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            ts_data = self.transform(ts_data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return ts_data, target

    def __len__(self) -> int:
        return len(self.data)

    def get_index(self):
        return self.index

class TorchVision_FL(VisionDataset):
    """This is just a trimmed down version of torchvision.datasets.MNIST.

    Use this class by either passing a path to a torch file (.pt)
    containing (data, targets) or pass the data, targets directly
    instead.
    """
    def __init__(
        self,
        path_to_client=None,
        partition=None,
        data=None,
        targets=None,
        loss_state=None,
        data_selection=None,
        transform: Optional[Callable] = None,
    ) -> None:
        path_to_data = path_to_client / (partition + ".pt")
        path = path_to_data.parent if path_to_data else None
        super(TorchVision_FL, self).__init__(path, transform=transform)
        self.transform = transform

        if path_to_data:
            # load data and targets (path_to_data points to an specific .pt file)
            self.data, self.targets = torch.load(path_to_data)
        else:
            self.data = data
            self.targets = targets

        # initialize the index
        self.index = list(range(len(self.targets)))

        # Data Selection
        if data_selection is not None:
            data, targets, index = dataSelection(self.data, self.targets, path_to_client, data_selection)
            self.data = data
            self.targets = targets
            self.index = index


    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if not isinstance(img, Image.Image):  # if not PIL image
            if not isinstance(img, np.ndarray):  # if torch tensor
                img = img.numpy()
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def get_index(self):
        return self.index

def getUCIHAR(path_to_data="./data"):
    """Downloads UCIHAR dataset and generates a unified training set (it will
    be partitioned later using the LDA partitioning mechanism."""

    # download dataset and load train set
    train_set = sensor_data.UCIHAR(root=path_to_data, train=True, download=True)

    # fuse all data splits into a single "training.pt"
    data_loc = Path(path_to_data) / "uci_har"
    training_data = data_loc / "training.pt"
    print("Generating unified UCIHAR dataset")
    torch.save([train_set.data, np.array(train_set.targets)], training_data)

    test_set = sensor_data.UCIHAR(
        root=path_to_data, train=False, transform=uciharTransformation()
    )

    # returns path where training data is and testset
    return training_data, test_set

def getMotion(path_to_data="./data"):
    """Downloads MotionSense dataset and generates a unified training set (it will
    be partitioned later using the LDA partitioning mechanism."""

    # download dataset and load train set
    train_set = sensor_data.MotionSense(root=path_to_data, train=True, download=True)

    # fuse all data splits into a single "training.pt"
    data_loc = Path(path_to_data) / "motion"
    training_data = data_loc / "training.pt"
    print("Generating unified MotionSense dataset")
    torch.save([train_set.data, np.array(train_set.targets)], training_data)

    test_set = sensor_data.MotionSense(
        root=path_to_data, train=False, transform=motionTransformation()
    )

    # returns path where training data is and testset
    return training_data, test_set


def getPAMAP2(path_to_data="./data"):
    """Downloads PAMAP2 dataset and generates a unified training set (it will
    be partitioned later using the LDA partitioning mechanism."""

    # download dataset and load train set
    train_set = sensor_data.PAMAP2(root=path_to_data, train=True, download=True)

    # fuse all data splits into a single "training.pt"
    data_loc = Path(path_to_data) / "pamap"
    training_data = data_loc / "training.pt"
    print("Generating unified PAMAP dataset")
    torch.save([train_set.data, np.array(train_set.targets)], training_data)

    test_set = sensor_data.PAMAP2(
        root=path_to_data, train=False, transform=pamapTransformation()
    )    
    # returns path where training data is and testset
    return training_data, test_set

def getCIFAR10(path_to_data="./data"):
    """Downloads CIFAR10 dataset and generates a unified training set (it will
    be partitioned later using the LDA partitioning mechanism."""

    # download dataset and load train set
    train_set = datasets.CIFAR10(root=path_to_data, train=True, download=True)

    # fuse all data splits into a single "training.pt"
    data_loc = Path(path_to_data) / "cifar-10-batches-py"
    training_data = data_loc / "training.pt"
    print("Generating unified CIFAR10 dataset")
    torch.save([train_set.data, np.array(train_set.targets)], training_data)

    test_set = datasets.CIFAR10(
        root=path_to_data, train=False, transform=cifar10Transformation()
    )

    # returns path where training data is and testset
    return training_data, test_set


def getCIFAR100(path_to_data="./data"):
    """Downloads CIFAR100 dataset and generates a unified training set (it will
    be partitioned later using the LDA partitioning mechanism."""

    # download dataset and load train set
    train_set = datasets.CIFAR100(root=path_to_data, train=True, download=True)

    # fuse all data splits into a single "training.pt"
    data_loc = Path(path_to_data) / "cifar-100-python"
    training_data = data_loc / "training.pt"
    print("Generating unified CIFAR100 dataset")
    torch.save([train_set.data, np.array(train_set.targets)], training_data)

    test_set = datasets.CIFAR100(
        root=path_to_data, train=False, transform=cifar10Transformation()
    )

    # returns path where training data is and testset
    return training_data, test_set



def getEMNIST(path_to_data="./data"):
    """Downloads EMNIST dataset and generates a unified training set (it will
    be partitioned later using the LDA partitioning mechanism."""

    # download dataset and load train set
    train_set = datasets.EMNIST(
        root=path_to_data, split='balanced', train=True, transform=emnistTransformation(), download=True)

    data_loc = Path(path_to_data) / "EMNIST/processed"
    training_data = data_loc / "training_balanced.pt"
    if not os.path.exists(data_loc): os.makedirs(data_loc)
    print("Generating unified emnist dataset")
    torch.save([train_set.data, np.array(train_set.targets)], training_data)

    test_set = datasets.EMNIST(
        root=path_to_data, split='balanced', train=False, transform=emnistTransformation()
    )

    # returns path where training data is and testset
    return training_data, test_set

# def getFEMNIST(path_to_data="./data"):
#     """Downloads FEMNIST dataset and generates a unified training set (it will
#     be partitioned later using the LDA partitioning mechanism."""

#     # download dataset and load train set
#     training_data = FEMNIST(
#         root=path_to_data, train=True, transform=femnistTransformation(), download=True)

#     test_set = FEMNIST(
#         root=path_to_data, train=False, transform=femnistTransformation()
#     )

#     # returns path where training data is and testset
#     return training_data, test_set



class Saver:
    def __init__(self, args):
        self.sample_number = []
        self.accuracy = []
        self.sn_start = False
        self.acc_start = False

        self.destine = args.destine
        self.num_epochs = args.num_epochs
        self.save = args.save_results
        
        self.perid = "_1{}_2{}_3{}_c{}|{}_e{}_a{}_b{}_g{}_flpa{}_mr{}|{}".format(
                                        args.dataset,
                                        args.encoder[:4],
                                        args.classifier,
                                        args.min_fit_clients,
                                        args.min_available_clients,
                                        args.num_epochs,
                                        args.alpha,
                                        args.beta,
                                        args.gamma,
                                        args.fl_partitioning_alpha,
                                        args.mrate_min,
                                        args.mrate_max,
                                        )

        if args.vanilla:
            self.perid = 'vanilla' + self.perid

        if args.pre_trained:
            self.perid = 'pt' + self.perid
            
        if args.fed_prox_mu != 0:
            self.perid = 'prox' + str(args.fed_prox_mu) + self.perid

        if self.save == True:
            sn_filedir = "test_results/sample_number_" + self.destine + self.perid + ".csv"
            self.sn_filedir = sn_filedir

            samples_key = ['samples_e'+str(i) for i in list(range(1,int(self.num_epochs)+1))]
            samples_key = ','.join(samples_key)

            with open(sn_filedir, "w") as f:
                f.write(f"dest,round,cid,{samples_key}\n")

        # reset cids buffer
        with open("log/cids_buffer.csv", 'w') as f:
            f.write('')

    # buffer for saving the pre cids list
    def cids_buffer_append(self, cid, iot_storage_sn, sample_gain_rate):
        with open("log/cids_buffer.csv", 'a') as f:
            f.write(f"{cid}|{iot_storage_sn}|{sample_gain_rate},")
            time.sleep(0.1)
        
    def cids_buffer_reset(self):
        with open('log/cids_buffer.csv', 'w') as f:
            f.write('')

    def sample_number_saver(self, destine, cid, len_traindata, config):
        if self.save == True:
            samples_number = [str(i) for i in len_traindata]
            samples_number = ','.join(samples_number)
            server_round = config["server_round"]

            with open(self.sn_filedir, "a") as f:
                f.write(f"{destine},{server_round},{cid},{samples_number}\n")


    def accuracy_saver(self, curound, accuracy, destine):
        if self.save == True:
            acc_filedir = "test_results/acc_" + destine + self.perid + ".csv" # config["destine"] not working
            
            if self.acc_start == False:
                self.acc_start = True
                with open(acc_filedir, "w") as f:
                    f.write(f"round,accuracy\n")
                    f.write(f"{str(curound)},{str(accuracy)}\n")
            else:
                with open(acc_filedir, "a") as f:
                    f.write(f"{str(curound)},{str(accuracy)}\n")