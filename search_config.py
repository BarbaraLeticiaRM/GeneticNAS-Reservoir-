#######################################
#           User input                #
#######################################
import argparse

parser = argparse.ArgumentParser(description='PyTorch GNAS')
parser.add_argument('--dataset_name', type=str, choices=['MIT', 'EEG'], help='the working data',
                    default='MIT')
parser.add_argument('--config_file', type=str, help='location of the config file')
parser.add_argument('--search_dir', type=str, help='the log dir of the search')
parser.add_argument('--model_dir', type=str, help='the log dir of the test')
parser.add_argument('--final', type=bool, help='location of the config file', default=False)
parser.add_argument('--data_path', type=str, default='/dataset/', help='location of the dataset')
parser.add_argument('--reservoir_size', type=int, default=360, help='')
parser.add_argument('--sparsity', type=float, default=0.05, help='')
parser.add_argument('--spectral_radius', type=float, default=0.95, help='')
parser.add_argument('--reservoir_operation', type=str, choices=['Reservoir', 'Reservoir-LSTM','Reservoir-CNN', 'None'], default='None', help='')

args = parser.parse_args()  
