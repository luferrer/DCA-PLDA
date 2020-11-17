import argparse
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from data import SpeakerDataset
from utils import np_to_torch, evaluate, load_model, compute_sideinfo
from IPython import embed
from scores import *

parser = argparse.ArgumentParser()
parser.add_argument('--cuda',     help='Enables cuda.', action='store_true')
parser.add_argument('model',      help='Path to the model to be used for evaluation.')
parser.add_argument('embeddings', help='Path to the npz file with development embeddings.')
parser.add_argument('out_file',   help='Output file for the computed vectors.')

opt = parser.parse_args()

##### Set the device and data type
cudnn.benchmark = True
if opt.cuda:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda:0")
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = torch.device("cpu")

###### Load the data
dataset = SpeakerDataset(opt.embeddings, device=device)

###### Load the model
model = load_model(opt.model, device)
print("Loaded model from %s"%opt.model)
    
###### Compute the sideinfo vectors
ids, si = compute_sideinfo(model, dataset)

f = h5py.File(opt.out_file, 'w')

f.create_dataset('ids', data=np.string_(ids))
f.create_dataset('data', data=np.array(si))
            
f.close()    




