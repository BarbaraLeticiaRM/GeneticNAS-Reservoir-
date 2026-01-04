import os
import torch
import torchvision
import torchvision.transforms as transforms
from modules.cut_out import Cutout, Cutout1D
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os 

class PTDataset(Dataset):
    def __init__(self, pt_file_path, transform=None):
        super().__init__()
        dataset = torch.load(pt_file_path)  
        self.data = dataset['data']
        self.labels = dataset['labels']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

def get_dataset(config):
    dataset_name = config.get('dataset_name')
    data_path = config.get('data_path')
    if dataset_name == 'MIT':
        return get_data(config, os.path.join(data_path, 'MIT'), dataset_name='MIT')
    elif dataset_name == 'EEG':
        return get_data(config, os.path.join(data_path, 'EEG'), dataset_name='EEG')
    else:
        raise Exception('unkown dataset type')


def get_data(config, data_path, dataset_name):
    if dataset_name == 'MIT':
        def data_transforms_mit(use_cutout, cutout_length):

            MIT_MEAN = -0.407281
            MIT_STD = 0.470996

            transform = transforms.Compose([
                transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32).reshape(1, -1) if isinstance(x, np.ndarray) else x.clone().detach().float().reshape(1, -1)),
                transforms.Lambda(lambda x: (x - MIT_MEAN) / MIT_STD),
            ])

            if use_cutout:
                train_transform.transforms.append(Cutout1D(cutout_length=cutout_length))

            return transform

        train_file = 'dataset/MIT/dataset_mit_train.pt'
        val_file = 'dataset/MIT/dataset_mit_val.pt'
        test_file = 'dataset/MIT/dataset_mit_test.pt'

        transform = data_transforms_mit(use_cutout=config.get('cutout'), cutout_length=config.get('length'))

        train_dataset = PTDataset(train_file, transform=transform)
        val_dataset = PTDataset(val_file, transform=transform)
        test_dataset = PTDataset(test_file, transform=transform)

        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.get('batch_size'), shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.get('batch_size_val'), shuffle=False, num_workers=4)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=4)

        n_class = 4


    elif dataset_name == 'EEG':
        def data_transforms_eeg(use_cutout, cutout_length):
            EEG_MEAN =  0.000152
            EEG_STD = 8.069389

            transform = transforms.Compose([
                transforms.Lambda(
                    lambda x: torch.tensor(x, dtype=torch.float32) 
                            if isinstance(x, np.ndarray) 
                            else x.clone().detach().float()
                ),
                transforms.Lambda(lambda x: (x - EEG_MEAN) / EEG_STD),
            ])

            if use_cutout:
                transform.transforms.append(Cutout1D(cutout_length=cutout_length))

            return transform

        train_file = 'dataset/EEG/dataset_eeg_train.pt'
        val_file = 'dataset/EEG/dataset_eeg_val.pt'
        test_file = 'dataset/EEG/dataset_eeg_test.pt'

        transform = data_transforms_eeg(use_cutout=config.get('cutout'), cutout_length=config.get('length'))

        train_dataset = PTDataset(train_file, transform=transform)
        val_dataset = PTDataset(val_file, transform=transform)
        test_dataset = PTDataset(test_file, transform=transform)


        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.get('batch_size'), shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.get('batch_size_val'), shuffle=False, num_workers=0)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)

        n_class = 109


    else:
        raise Exception('unkown dataset' + dataset_name)

    return trainloader, val_loader, testloader, n_class


