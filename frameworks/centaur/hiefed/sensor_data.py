import os
import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from io import BytesIO

import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url
from torchvision.datasets.vision import VisionDataset


class PAMAP2(VisionDataset):
    """`PAMAP2 <https://github.com/microsoft/PersonalizedFL>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``pamap2-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "pamap"
    url = "https://wjdcloud.blob.core.windows.net/dataset/cycfed/pamap.tar.gz"
    filename = "pamap.tar.gz"
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train

        if download:
            if not os.path.exists(self.root + "/" + self.base_folder):
                self.download()

        file_path = os.path.join(self.root, self.base_folder, 'x.npy')
        self.data_all = np.load(file_path).astype(np.float32)        
        file_path = os.path.join(self.root, self.base_folder, 'y.npy')
        self.targets_all = np.load(file_path)

        indices = np.random.RandomState(seed=0).permutation(len(self.data_all))            
        self.data_all = self.data_all[indices]
        self.targets_all = self.targets_all[indices]       
        
        split_ind =int(len(self.data_all)*0.8)
        self.data = self.data_all[:split_ind]
        self.targets = self.targets_all[:split_ind]
        d_mean = self.data.mean((0,1))
        d_var = self.data.std((0,1))        
        if self.train:
            self.data = (self.data - d_mean) / d_var                                                                   
        else:
            self.data = self.data_all[split_ind:]
            self.targets = self.targets_all[split_ind:]
            self.data = (self.data - d_mean) / d_var                                                                   
        self.data = np.expand_dims(self.data, 3)
        self.data = np.einsum("itsc->istc", self.data)         
        print(self.data.shape)             
        print(self.targets.shape)             
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def download(self) -> None:
        download_and_extract_archive(self.url, self.root, filename=self.filename)

class UCIHAR(VisionDataset):
    """`UCIHAR <https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``uci_har`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    # data_class_names = ["Walking", "Sairs-Up", "Sairs-Down", "Sitting", "Standing", "Lying"]

    base_folder = "uci_har"
    url = "https://raw.githubusercontent.com/mmalekzadeh/dana/master/dana/datasets/UCIHAR/"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train

        if download:
            if not os.path.exists(self.root + "/" + self.base_folder):
                self.download()

        
        file_path = os.path.join(self.root, self.base_folder, "X_train.npy")
        self.data = np.load(file_path)
        file_path = os.path.join(self.root, self.base_folder, "y_train.npy")
        self.targets = np.load(file_path)
        
        indices = np.random.RandomState(seed=0).permutation(len(self.data))            
        self.data = self.data[indices]
        self.targets = self.targets[indices]                        
                
        d_mean = self.data.mean((0,1))
        d_var = self.data.std((0,1))
        if self.train:
            self.data = (self.data - d_mean) / d_var                         
            self.data = np.expand_dims(self.data, 3)
            self.data = np.einsum("itsc->istc", self.data) 
            print(self.data.shape)                  
        else:
            file_path = os.path.join(self.root, self.base_folder, "X_test.npy")
            self.data = np.load(file_path)
            file_path = os.path.join(self.root, self.base_folder, "y_test.npy")
            self.targets = np.load(file_path)            

            self.data = (self.data - d_mean) / d_var            
            self.data = np.expand_dims(self.data, 3)
            self.data = np.einsum("itsc->istc", self.data)            
            print(self.data.shape)                       

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def download(self) -> None:
        dl_dir = self.root + "/" + self.base_folder
        if not os.path.exists(dl_dir):
            os.makedirs(dl_dir)
        np.save(
            dl_dir + "/X_train.npy",
            np.load(BytesIO(requests.get(self.url + "X_train.npy").content)),
        )
        np.save(
            dl_dir + "/y_train.npy",
            np.load(BytesIO(requests.get(self.url + "y_train.npy").content)).argmax(1),
        )
        np.save(
            dl_dir + "/X_test.npy",
            np.load(BytesIO(requests.get(self.url + "X_test.npy").content)),
        )
        np.save(
            dl_dir + "/y_test.npy",
            np.load(BytesIO(requests.get(self.url + "y_test.npy").content)).argmax(1),
        )




class MotionSense(VisionDataset):
    """`MotionSense <https://github.com/mmalekzadeh/motion-sense>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``motionsense`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "motion"
    url = "https://raw.githubusercontent.com/mmalekzadeh/motion-sense/master/data/A_DeviceMotion_data.zip"
    filename = "A_DeviceMotion_data.zip"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train
        self.root = root

        if download:
            if not os.path.exists(self.root + "/" + self.base_folder):
                self.download()

        file_path = os.path.join(self.root, self.base_folder, "X_train.npy")
        self.data = np.load(file_path).astype(np.float32) 
        file_path = os.path.join(self.root, self.base_folder, "y_train.npy")
        self.targets = np.load(file_path).astype(int)

        indices = np.random.RandomState(seed=0).permutation(len(self.data))
        self.data = self.data[indices]
        self.targets = self.targets[indices]

        d_mean = self.data.mean((0,1))
        d_var = self.data.std((0,1))
        if self.train:
            self.data = (self.data - d_mean) / d_var
            self.data = np.expand_dims(self.data, 3)
            self.data = np.einsum("itsc->istc", self.data)
            print(self.data.shape)
        else:
            file_path = os.path.join(self.root, self.base_folder, "X_test.npy")
            self.data = np.load(file_path).astype(np.float32) 
            file_path = os.path.join(self.root, self.base_folder, "y_test.npy")
            self.targets = np.load(file_path).astype(int)

            self.data = (self.data - d_mean) / d_var
            d_mean = self.data.mean((0,1))
            d_var = self.data.std((0,1))
            self.data = np.expand_dims(self.data, 3)
            self.data = np.einsum("itsc->istc", self.data)
            print(self.data.shape)



    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def download(self) -> None:
        dl_dir = self.root + "/" + self.base_folder
        download_and_extract_archive(self.url, dl_dir, filename=self.filename)
        sub_url= "https://raw.githubusercontent.com/mmalekzadeh/motion-sense/master/data/data_subjects_info.csv"
        download_url(sub_url, dl_dir, filename="data_subjects_info.csv")
         
        def get_ds_infos():
            """
            Read the file includes data subject information.

            Data Columns:
            0: code [1-24]
            1: weight [kg]
            2: height [cm]
            3: age [years]
            4: gender [0:Female, 1:Male]

            Returns:
                A pandas DataFrame that contains inforamtion about data subjects' attributes
            """

            dss = pd.read_csv(dl_dir+"/data_subjects_info.csv")
            print("[INFO] -- Data subjects' information is imported.")

            return dss

        def set_data_types(data_types=["userAcceleration"]):
            """
            Select the sensors and the mode to shape the final dataset.

            Args:
                data_types: A list of sensor data type from this list: [attitude, gravity, rotationRate, userAcceleration]

            Returns:
                It returns a list of columns to use for creating time-series from files.
            """
            dt_list = []
            for t in data_types:
                if t != "attitude":
                    dt_list.append([t+".x",t+".y",t+".z"])
                else:
                    dt_list.append([t+".roll", t+".pitch", t+".yaw"])

            return dt_list


        def creat_time_series(dt_list, act_labels, trial_codes, mode="mag", labeled=True):
            """
            Args:
                dt_list: A list of columns that shows the type of data we want.
                act_labels: list of activites
                trial_codes: list of trials
                mode: It can be "raw" which means you want raw data
                for every dimention of each data type,
                [attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)].
                or it can be "mag" which means you only want the magnitude for each data type: (x^2+y^2+z^2)^(1/2)
                labeled: True, if we want a labeld dataset. False, if we only want sensor values.

            Returns:
                It returns a time-series of sensor data.

            """
            num_data_cols = len(dt_list) if mode == "mag" else len(dt_list*3)

            if labeled:
                train_dataset = np.zeros((0,num_data_cols+7)) # "7" --> [act, code, weight, height, age, gender, trial]
                test_dataset = np.zeros((0,num_data_cols+7))
            else:
                train_dataset = np.zeros((0,num_data_cols))
                test_dataset = np.zeros((0,num_data_cols))

            ds_list = get_ds_infos()

            print("[INFO] -- Creating Time-Series")
            for sub_id in ds_list["code"]:
                for act_id, act in enumerate(act_labels):
                    for trial in trial_codes[act_id]:
                        fname = dl_dir+'/A_DeviceMotion_data/'+act+'_'+str(trial)+'/sub_'+str(int(sub_id))+'.csv'
                        raw_data = pd.read_csv(fname)
                        raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                        vals = np.zeros((len(raw_data), num_data_cols))
                        for x_id, axes in enumerate(dt_list):
                            if mode == "mag":
                                vals[:,x_id] = (raw_data[axes]**2).sum(axis=1)**0.5
                            else:
                                vals[:,x_id*3:(x_id+1)*3] = raw_data[axes].values
                            vals = vals[:,:num_data_cols]
                        if labeled:
                            lbls = np.array([[act_id,
                                    sub_id-1,
                                    ds_list["weight"][sub_id-1],
                                    ds_list["height"][sub_id-1],
                                    ds_list["age"][sub_id-1],
                                    ds_list["gender"][sub_id-1],
                                    trial
                                ]]*len(raw_data))
                            vals = np.concatenate((vals, lbls), axis=1)
                        if trial < 10:
                            train_dataset = np.append(train_dataset,vals, axis=0)
                        else:
                            test_dataset = np.append(test_dataset,vals, axis=0)
            cols = []
            for axes in dt_list:
                if mode == "raw":
                    cols += axes
                else:
                    cols += [str(axes[0][:-2])]

            if labeled:
                cols += ["act", "id", "weight", "height", "age", "gender", "trial"]

            train_dataset = pd.DataFrame(data=train_dataset, columns=cols)
            test_dataset = pd.DataFrame(data=test_dataset, columns=cols)
            return train_dataset, test_dataset
        #________________________________


        ACT_LABELS = ["dws","ups", "wlk", "jog", "std", "sit"]
        TRIAL_CODES = {
            ACT_LABELS[0]:[1,2,11],
            ACT_LABELS[1]:[3,4,12],
            ACT_LABELS[2]:[7,8,15],
            ACT_LABELS[3]:[9,16],
            ACT_LABELS[4]:[6,14],
            ACT_LABELS[5]:[5,13]
        }

        ## Here we set parameter to build labeld time-series from dataset of "(A)DeviceMotion_data"
        ## attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
        sdt = ["userAcceleration", "rotationRate", "attitude"]
        print("[INFO] -- Selected sensor data types: "+str(sdt))
        act_labels = ACT_LABELS
        print("[INFO] -- Selected activites: "+str(act_labels))
        trial_codes = [TRIAL_CODES[act] for act in act_labels]
        dt_list = set_data_types(sdt)
        train_ts, test_ts = creat_time_series(dt_list, act_labels, trial_codes, mode="raw", labeled=True)
        print("[INFO] -- Shape of training time-Series dataset:"+str(train_ts.shape))
        print("[INFO] -- Shape of test time-Series dataset:"+str(test_ts.shape))
        # print(train_ts.head())

        def time_series_to_section(dataset, num_act_labels, sliding_window_size, step_size_of_sliding_window):
            data = dataset.iloc[: , 0:9]
            act_labels = dataset.iloc[: , 9]

            ## We want the Rows of matrices show each Feature and the Columns show time points.
            # data = data.T
            
            size_data = data.shape[0]
            size_features = data.shape[1]                

            number_of_secs = round(((size_data - sliding_window_size)/step_size_of_sliding_window))

            ##  Create a 3D matrix for Storing Snapshots
            secs_data = np.zeros((number_of_secs , sliding_window_size, size_features))
            act_secs_labels = np.zeros((number_of_secs))

            k=0
            for i in range(0 ,(size_data)-sliding_window_size  , step_size_of_sliding_window):
                j = i // step_size_of_sliding_window
                if(j>=number_of_secs):
                    break
                if(not (act_labels[i] == act_labels[i+sliding_window_size-1]).all()):
                    continue
                secs_data[k] = data.iloc[i:i+sliding_window_size, 0:size_features]
                act_secs_labels[k] = act_labels[i].astype(int)
                k = k+1
            secs_data = secs_data[0:k]
            act_secs_labels = act_secs_labels[0:k]
            return secs_data, act_secs_labels
        ##________________________________________________________________

        num_features = len(sdt)*3
        num_act_labels = len(ACT_LABELS)
        ## This Variable Defines the Size of Sliding Window
        ## ( e.g. 100 means in each snapshot we just consider 100 consecutive observations of each sensor)
        sliding_window_size = 128 # 50 Equals to 1 second for MotionSense Dataset (it is on 50Hz samplig rate)
        ## Here We Choose Step Size for Building Diffrent Snapshots from Time-Series Data
        ## ( smaller step size will increase the amount of the instances and higher computational cost may be incurred )
        step_size_of_sliding_window = 50
        print("--> Sectioning Training and Test datasets: shape of each section will be: (",num_features,"x",sliding_window_size,")")
        train_data, act_train_labels = time_series_to_section(train_ts.copy(),num_act_labels, sliding_window_size, step_size_of_sliding_window)

        test_data, act_test_labels = time_series_to_section(test_ts.copy(),num_act_labels,sliding_window_size,step_size_of_sliding_window,)
        print("--> Shape of Training Sections:", train_data.shape)
        print("--> Shape of Test Sections:", test_data.shape)


        np.save(dl_dir + "/X_train.npy", train_data )
        np.save(dl_dir + "/y_train.npy", act_train_labels )
        np.save(dl_dir + "/X_test.npy", test_data )
        np.save(dl_dir + "/y_test.npy", act_test_labels )
        

######################################################
######################################################
# class SenCoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(3, 9), stride=(3,1)),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(1, 3)),
#         )        
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(1, 9), stride=(1,1)),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(1, 3)),
#         )        
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 7), stride=(3,1)),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(1, 3)),
#         )
        
#         self.flatten = nn.Sequential(
#          nn.Flatten()
#         )

#     def forward(self, x):          
#         out = self.conv1(x)        
#         out = self.conv2(out) 
#         out = self.conv3(out)        
#         out = self.flatten(out) 
#         return out


# class SenFier(nn.Module):
#     def __init__(self, num_classes, cl_type, cl_inp_size):
#         super().__init__()

#         self.cl_type = cl_type
        
#         fc_size = 64
#         if self.cl_type == "large":
#             fc_size = 128

#         self.fcs = nn.Sequential(
#             nn.Linear(in_features=cl_inp_size, out_features=num_classes)
#         )

#         self.fcml1 = nn.Sequential(
#             nn.Linear(in_features=cl_inp_size, out_features=fc_size),
#             nn.ReLU(),
#             nn.Dropout(0.5)
#         )
#         self.fcml2 = nn.Sequential(
#             nn.Linear(in_features=fc_size, out_features=fc_size),
#             nn.ReLU(),
#             )
#         self.fcml = nn.Sequential(
#             nn.Linear(in_features=fc_size, out_features=num_classes)
#         )

#     def forward(self, x):
#         if self.cl_type == "small":
#             out = self.fcs(x)
#             return out
#         out = self.fcml1(x)
#         out = self.fcml2(out)
#         out = self.fcml(out)
#         return out


# class SenNet(nn.Module):
#     def __init__(self, encoder=None, classifier=None):
#         super(SenNet, self).__init__()
#         if encoder is not None:
#             self.encoder = encoder
#         if classifier is not None:
#             self.classifier = classifier

#     def forward(self, x):
#         embed = self.encoder(x)
#         # embed = embed.reshape(-1, 100 * 1)
#         out = self.classifier(embed)
#         return out

######################################################
class SenCoder_UCIHAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_chan = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.num_chan, kernel_size=(1, 5), stride=(1,1)),            
            nn.Sequential(nn.BatchNorm2d(num_features=self.num_chan)),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_chan, out_channels=self.num_chan, kernel_size=(1, 5), stride=(1,2)),
            nn.Sequential(nn.BatchNorm2d(num_features=self.num_chan)),
            nn.LeakyReLU(),            
            # nn.MaxPool2d(kernel_size=(1, 2), stride=(1,2)),
        )        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_chan, out_channels=self.num_chan, kernel_size=(1, 5), stride=(1,2)),
            nn.Sequential(nn.BatchNorm2d(num_features=self.num_chan)),
            nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=(1, 2),  stride=(1,2)),            
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_chan, out_channels=self.num_chan, kernel_size=(3, 3), stride=(1,1)),
            nn.Sequential(nn.BatchNorm2d(num_features=self.num_chan)),
            nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=(1, 2),  stride=(1,2)),            
        )        
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_chan, out_channels=self.num_chan, kernel_size=(3, 3), stride=(1,2)),
            nn.Sequential(nn.BatchNorm2d(num_features=self.num_chan)),
            nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=(1, 2),  stride=(1,2)),            
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_chan, out_channels=100, kernel_size=(1, 3), stride=(2,2)),
            nn.Sequential(nn.BatchNorm2d(num_features=100)),
            nn.LeakyReLU(),            
        )
        self.pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 5)), nn.Dropout(0.5)
        )

    def forward(self, x):         
        out = self.conv1(x)        
        out = self.conv2(out)        
        out = self.conv3(out)
        out = self.conv4(out)        
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.pool(out)
        return out


class SenCoder_PAMAP2(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_chan = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.num_chan, kernel_size=(1, 5), stride=(1,1)),            
            nn.Sequential(nn.BatchNorm2d(num_features=self.num_chan)),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_chan, out_channels=self.num_chan, kernel_size=(1, 5), stride=(1,2)),
            nn.Sequential(nn.BatchNorm2d(num_features=self.num_chan)),
            nn.LeakyReLU(),            
            # nn.MaxPool2d(kernel_size=(1, 2), stride=(1,2)),
        )        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_chan, out_channels=self.num_chan, kernel_size=(1, 5), stride=(1,2)),
            nn.Sequential(nn.BatchNorm2d(num_features=self.num_chan)),
            nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=(1, 2),  stride=(1,2)),            
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_chan, out_channels=self.num_chan, kernel_size=(3, 5), stride=(2,2)),
            nn.Sequential(nn.BatchNorm2d(num_features=self.num_chan)),
            nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=(1, 2),  stride=(1,2)),            
        )        
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_chan, out_channels=self.num_chan, kernel_size=(3, 5), stride=(2,2)),
            nn.Sequential(nn.BatchNorm2d(num_features=self.num_chan)),
            nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=(1, 2),  stride=(1,2)),            
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_chan, out_channels=100, kernel_size=(1, 5), stride=(2,2)),
            nn.Sequential(nn.BatchNorm2d(num_features=100)),
            nn.LeakyReLU(),            
        )
        self.pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3)), nn.Dropout(0.5)
        )

    def forward(self, x):         
        out = self.conv1(x)        
        out = self.conv2(out)        
        out = self.conv3(out)
        out = self.conv4(out)        
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.pool(out)
        return out


class SenFier_S(nn.Module):
    def __init__(self, num_classes, cl_type, cl_inp_size):
        super().__init__()

        self.fcs = nn.Sequential(
            nn.Linear(in_features=cl_inp_size, out_features=num_classes)
        )
        
    def forward(self, x):
        out = self.fcs(x)
        return out
        
class SenFier_ML(nn.Module):
    def __init__(self, num_classes, cl_type, cl_inp_size):
        super().__init__()

        self.cl_type = cl_type
        fc_size = 64
        if self.cl_type == "large":
            fc_size = 128

        self.fcml1 = nn.Sequential(
            nn.Linear(in_features=cl_inp_size, out_features=fc_size),            
            nn.Dropout(p=0.2, inplace=True)
            )
        self.fcml2 = nn.Sequential(
            nn.Linear(in_features=fc_size, out_features=num_classes)
        )

    def forward(self, x):
        out = self.fcml1(x)
        out = self.fcml2(out)        
        return out


class SenNet(nn.Module):
    def __init__(self, encoder=None, classifier=None):
        super(SenNet, self).__init__()        
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        embed = self.encoder(x)
        embed = embed.reshape(-1, 100 * 1)
        out = self.classifier(embed)
        return out
