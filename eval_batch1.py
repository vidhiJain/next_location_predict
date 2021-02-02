import ast
import numpy as np
from pathlib import Path
import json
import time
import math
import argparse
import random
import torch
import networkx as nx
import matplotlib.pyplot as plt

from network import TransformerModel
from data_utils import get_batch, batchify, get_accuracy
import arguments 


args = arguments.get_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = 'resources'
map_set_number = 0
# filename = 'human_room_nav.csv'
# filename = 'human_room_nav_sparky_0_till_400.csv'
filename = f'human_room_nav_{str(map_set_number)}.csv'
G = nx.read_edgelist(f'resources/sparky_map_{str(map_set_number)}.edgelist')

raw_data = open(Path(root, filename), 'r').readlines()
d2 = [line.rstrip().split(',') for line in raw_data] 
d3 = [[int(i) for i in j] for j in d2]
d4 = [torch.tensor(i).type(torch.int64) for i in d3]


model = TransformerModel(args.ntokens, args.emsize, args.nhead, args.nhid, 
    args.nlayers, args.dropout).to(device)
model.load_state_dict(torch.load('best-model-parameters.pt'))

# for subject_num in range(len(d4)):  # args.eval_batch_size:
    # human_data = torch.tensor(d4[subject_num])
    # data_source = batchify(human_data, 1, device)
    # print(f'Acc of transformer model for human data {i}: {get_accuracy(data_source, model, args)}') 
correct_list = {}
total_list = {}
# args.eval_batch_size
for j in range(len(d4)):
    print(f'\nsubject {j}')
    human_data = batchify(torch.tensor(d4[j]), 1)
    model.eval()
    avg_acc = 0
    total_correct = 0
    with torch.no_grad():
        for i in range(0, human_data.size(0)-1, args.bptt):
            inp, tgt = get_batch(human_data, i, args.bptt)
            

            inp, tgt = inp.to(device), tgt.to(device)
            out = model(inp)
            # import ipdb; ipdb.set_trace()
            pred_probs = out.view(-1, args.ntokens)
            _, pred = pred_probs.max(1)
            correct = (pred == tgt) #.sum().item()
            # print(inp)
            # print(correct)
            
            total = tgt.size(0)
            acc = correct.sum().item()*100./total
            
            for k in range(inp.shape[0]):
                key = G.degree[str(inp[k].cpu().detach().numpy()[0])]
                if correct_list.get(key, None) is None:
                    correct_list[key] = 0
                    total_list[key] = 0
                correct_list[key] += int(correct[k].item())
                total_list[key] += 1
            # avg_acc += 1/(i+1) * (acc - avg_acc)
            total_correct += correct.sum().item()
    print(total_correct, len(d4[j]))
    print('Accuracy', total_correct / len(d4[j]))
    print('correct_list', correct_list)
    print('total_list', total_list)
import ipdb; ipdb.set_trace()

X = list(correct_list.keys())
X = np.array(X)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X, list(correct_list.values()), color='b', width=0.25)
ax.bar(X+0.25, list(total_list.values()), color='r', width=0.25)
plt.ylabel('Counts')
plt.xlabel('Degree')
plt.title('Next Location Accuracy per degree of the area')
plt.xticks(X)
plt.yticks(np.arange(0, 120, 10))
plt.savefig(f'accuracy_per_degree_{str(map_set_number)}')

# print(avg_acc)