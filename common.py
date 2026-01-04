import os
import pickle
import datetime
from enum import Enum


class ModelType(Enum):
    CNN = 0


def make_log_dir(config):
    log_dir = os.path.join('.', 'logs', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


# def load_final(model, search_dir):
#     ind_file = os.path.join(search_dir, 'best_individual.pickle')
#     ind = pickle.load(open(ind_file, "rb"))
#     model.set_individual(ind)
#     return ind


import os
import pickle
import torch # Importar torch caso você adicione o carregamento de pesos

def load_final(model, search_dir):
    ind_file = os.path.join(search_dir, 'best_individual.pickle')
   
    with open(ind_file, "rb") as f:
        ind = pickle.load(f)
    
    model.set_individual(ind)
    
    return ind

def get_model_type(dataset_name):
    if dataset_name in ['MIT', 'EEG']:
        return ModelType.CNN
    else:
        raise Exception('unkown model for dataset:' + dataset_name)
