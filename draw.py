import os
import numpy as np

from gnas.search_space.individual import Individual, MultipleBlockIndividual
from gnas.search_space.operation_space import CnnNodeConfig


def add_node(graph, node_id, label, shape='box', style='filled'):
    if label.startswith('Input') or label.startswith('x'):
        color = 'skyblue'
    elif label.startswith('Output'):
        color = 'pink'
    elif 'Tanh' in label or 'Add' in label or 'Concat' in label:
        color = 'yellow'
    elif 'ReLU' in label or 'Dw' in label or 'Conv' in label:
        color = 'orange'
    elif 'Sigmoid' in label or 'Identity' in label:
        color = 'greenyellow'
    elif label == 'avg':
        color = 'seagreen3'
    elif 'Reservoir' in label:
        color = 'pink'
    else:
        color = 'white'

    if not any(label.startswith(word) for word in ['x', 'avg', 'h']):
        label = f"{label}\n({node_id})"

    graph.add_node(
        node_id, label=label, color='black', fillcolor=color,
        shape=shape, style=style,
    )


def _draw_individual(ocl, individual, path=None):
    import pygraphviz as pgv
    graph = pgv.AGraph(directed=True, layout='dot')  # not work?

    ofset = len(ocl[0].inputs)
    for i in range(len(ocl[0].inputs)):
        add_node(graph, i, 'x[' + str(i) + ']')

    input_list = []

    for i, (oc, op) in enumerate(zip(individual.generate_node_config(), ocl)):
        if isinstance(op, CnnNodeConfig):
            input_a = oc[0]
            input_b = oc[1]
            input_list.append(input_a)
            input_list.append(input_b)
            op_a = oc[4]
            op_b = oc[5]
            add_node(graph, (i + ofset) * 10, ocl[i].op_list[op_a])
            add_node(graph, (i + ofset) * 10 + 1, ocl[i].op_list[op_b])
            graph.add_edge(input_a, (i + ofset) * 10)
            graph.add_edge(input_b, (i + ofset) * 10 + 1)
            add_node(graph, (i + ofset), 'Add')
            graph.add_edge((i + ofset) * 10, (i + ofset))
            graph.add_edge((i + ofset) * 10 + 1, (i + ofset))
            c = graph.add_subgraph([(i + ofset) * 10, (i + ofset) * 10 + 1, (i + ofset)],
                                   name='cluster_block:' + str(i), label='Block ' + str(i))
            # c.attr(label='block:'+str(i))
        else:
            raise Exception('unkown node type')
        
    input_list = np.unique(input_list)
    op_inputs = [int(i) for i in np.linspace(ofset, ofset + individual.get_n_op() - 1, individual.get_n_op()) if
                 i not in input_list]
    concat_node = i + 1 + ofset
    add_node(graph, concat_node, 'Concat')
    for i in op_inputs:
        graph.add_edge(i, concat_node)
    graph.layout(prog='dot')
    if path is not None:
        graph.draw(path + '.png')



def draw_cell(ocl, individual):
    _draw_individual(ocl, individual, path=None)


def draw_network(ss, individual, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(individual, Individual):
        _draw_individual(ss.ocl, individual, path)
    elif isinstance(individual, MultipleBlockIndividual):
        [_draw_individual(ocl, inv, path + str(i)) for i, (inv, ocl) in
         enumerate(zip(individual.individual_list, ss.ocl))]



import torch
import os
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
from models import model_cnn
from data import get_dataset
from common import ModelType, get_model_type, load_final
from config import get_config
import gnas
from modules.drop_module import DropModuleControl
from search_config import args

model_type = get_model_type(dataset_name='MIT')
log_dir_ind = 'logs_sem_reservoir/teste2'

working_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Working device: {working_device}")

config = get_config(model_type)
config.update({'data_path': './data', 'dataset_name': 'MIT', 'working_device': str(working_device)})

_, testloader, n_param = get_dataset(config)

with open(os.path.join(log_dir_ind, 'best_individual.pickle'), 'rb') as f:
    best_individual = pickle.load(f)

if model_type == ModelType.CNN:
    n_cell_type = gnas.SearchSpaceType(config.get('n_block_type') - 1)
    dp_control = DropModuleControl(config.get('drop_path_keep_prob'))
    ss = gnas.get_gnas_cnn_search_space(config.get('n_nodes'), dp_control, n_cell_type)
    net = model_cnn.Net(config.get('n_blocks'), config.get('n_channels'), n_param,
                    config.get('dropout'),
                    ss, aux=config.get('aux_loss')).to(working_device)
else:
    raise ValueError(f"Tipo de modelo não suportado: {model_type}")


net.set_individual(best_individual)

draw_network(ss, best_individual, os.path.join(log_dir_ind, 'best_graph_360'))
