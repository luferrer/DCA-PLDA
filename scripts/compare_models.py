import argparse
import torch
import numpy as np
from IPython import embed

parser = argparse.ArgumentParser(description="Print the L2 distance for each parameter between two models.")
parser.add_argument('model1',  help='Model 1 to compare.')
parser.add_argument('model2',  help='Model 2 to compare.')

opt = parser.parse_args()

###### Load the models
model1 = torch.load(opt.model1, map_location=torch.device('cpu'))['model']
model2 = torch.load(opt.model2, map_location=torch.device('cpu'))['model']

for p in model1.keys():

    m1 = model1[p]
    m2 = model2[p]

    print("%-30s  %f"%(p, np.linalg.norm(m1-m2)))

