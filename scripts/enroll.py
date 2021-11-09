import argparse
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from dca_plda.data import LabelledDataset
from dca_plda.utils_for_scripts import np_to_torch, load_model, setup_torch_and_cuda_vars
from dca_plda.utils import  save_checkpoint, load_configs, get_class_to_cluster_map_from_config
from dca_plda.scores import *

default_config = {
    'training': {
        'loss': 'cross_entropy',
        'ptar': 0.01}}

parser = argparse.ArgumentParser(description="Script for enrolling new languages")
parser.add_argument('--cuda',      help='Enables cuda.', action='store_true')
parser.add_argument('--config',    help='Config file.', default=None)
parser.add_argument('--clusters',  help='Map from class to cluster. Only used if the model is hierarchical.', default=None)
parser.add_argument('model',       help='Path to the model to be updated.')
parser.add_argument('embeddings',  help='Path to the npz file with training embeddings.')
parser.add_argument('metadata',    help='Metadata for enrollment samples.')
parser.add_argument('out_dir',     help='Dir for output model.')

opt = parser.parse_args()

##### Set the device and data type
device = setup_torch_and_cuda_vars(opt.cuda)

###### Load the model
model = load_model(opt.model, device)
print("Loaded model from %s"%opt.model)

print("Note: To enroll new classes the input model should be trained with freeze_params including ('cluster_stats','enrollment').")

###### Load the data
cluster_ids = get_class_to_cluster_map_from_config(model.config)
dataset = LabelledDataset(opt.embeddings, opt.metadata, cluster_ids=cluster_ids)

if model.enrollment_classes is None:
    print("This script can only be used when the model contains enrollment_classes.")

config = load_configs(opt.config, default_config, None, "%s/config"%(opt.out_dir))

# Set the config to only update the enrollment components and to add the new classes
config.training.init_params["init_components"]    = "enrollment"
config.training.init_params["enroll_new_classes"] = True

embeddings, metadata, metamaps = dataset.get_data_and_meta()
trn_loss = model.init_params_with_data(embeddings, metadata, metamaps, config.training, device=device)
save_checkpoint(model, opt.out_dir)


    
