import ast
import numpy as np
from pathlib import Path
import json
import time
import math
import argparse
import random
import torch

from network import TransformerModel
from data_utils import get_batch, batchify, get_accuracy
import arguments 


args = arguments.get_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
map_set_number = 0 
time_to_stop = 1000 # 400

print('map_set_number = ', map_set_number)
root = 'resources'
filename = f'human_room_nav_sparky_{str(map_set_number)}_till_{str(time_to_stop)}.csv'
# filename = f'human_room_nav_sparky_{str(map_set_number)}_till_{str(time_to_stop)}.csv'

raw_data = open(Path(root, filename), 'r').readlines()
d2 = [line.rstrip().split(',') for line in raw_data] 
d3 = [[int(i) for i in j] for j in d2]
d4 = [torch.tensor(i).type(torch.int64) for i in d3]


model = TransformerModel(args.ntokens, args.emsize, args.nhead, args.nhid, 
    args.nlayers, args.dropout).to(device)
model.load_state_dict(torch.load(f'best-model-parameters-{map_set_number}.pt'))

# args.batch_size = 2
for i in range(len(d4)):
    human_data = torch.tensor(d4[i])
    data_source = batchify(human_data, args.eval_batch_size, device)
    # print(f'Acc of transformer model for human data {i}: {get_accuracy(data_source, model, args)}') 
    acc = get_accuracy(data_source, model, args)
    print('{:.4f}'.format(acc))
