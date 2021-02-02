import ast
import numpy as np
from pathlib import Path
import json
import time
import math
import argparse
import random
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.autograd import Variable

from network import TransformerModel
from data_utils import get_batch, batchify, get_accuracy
import arguments 
seed = 42
random.seed(seed)
torch.manual_seed(seed)

args = arguments.get_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")


# HYPERPARAM
batch_size = args.batch_size
bptt = args.bptt
eval_batch_size = args.eval_batch_size

ntokens = args.ntokens  # 26 #len(TEXT.vocab.stoi) # the size of vocabulary # number of rooms
emsize = args.emsize  # 26 # 200 # embedding dimension
nhid = args.nhid  # 8 # 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = args.nlayers  # 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = args.nhead  # 2 # the number of heads in the multiheadattention models
dropout = args.dropout  # 0.2 # the dropout value
lr = args.lr  # 5 # learning rate
epochs = args.epochs 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DATA

map_set_number = 1
# time_to_stop = 400
file_list = [x for x in glob(f'faux_human_room_nav/*_yellow-first_sparky{str(map_set_number)}.csv')]
# file_list = [f'resources/human_room_nav_sparky_{str(map_set_number)}_till_{str(time_to_stop)}.csv']
print(file_list)

# file_list = ['room_trajs_beep_False_opportunistic', 'room_trajs_beep_True_opportunistic',  'room_trajs_beep_True_yellow-first',  'room_trajs_beep_False_yellow-first']
# file_list = ['room_trajs_beep_True_yellow-first', 'room_trajs_beep_False_yellow-first']  #,  'room_trajs_beep_True_yellow-first',  'room_trajs_beep_False_yellow-first']

root2 = '../sparky_data'
room_segments_file = 'TestbedMap.json'

with open(Path(root2, room_segments_file), 'r') as g:
    room_segments = json.load(g)
id_index_rooms = {room_segments['areas'][i]['id']: i for i in range(len(room_segments['areas']))}
index_name_rooms = {value:room_segments['areas'][value]['name'] for _, value in id_index_rooms.items()}

d2 = []
for filename in file_list:
    raw_data = open(filename, 'r').readlines()
    d2 += [line.rstrip().split(',') + ['0'] for line in raw_data] # 0 added after end of every episode
d3 = [[int(i) for i in j] for j in d2]
d4 = [torch.tensor(i).type(torch.int64) for i in d3]

def train_val_test_split(data_list, train_part=0.6, val_part=0.1, shuffle=True):
    n = len(data_list)
    if shuffle:
        random.shuffle(data_list)
    train = torch.cat(data_list[: int(n*(train_part))])
    val = torch.cat(data_list[int(n*(train_part)) : int(n*(train_part + val_part))])
    test = torch.cat(data_list[int(n*(train_part + val_part)) :])
    return train, val, test

train, val, test = train_val_test_split(d4, train_part=0.8, val_part=0.1)
train_data = batchify(train, batch_size)
val_data = batchify(val, eval_batch_size)
test_data = batchify(test, eval_batch_size)

# MODEL & OPTIM PARAMS



model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def train(epoch):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, train_data.size(0)-1, bptt)):

        inputs, targets = get_batch(train_data, i, bptt)
        inputs = inputs.to(device)
        targets = targets.to(device)
        model.zero_grad()

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt,  scheduler.get_lr()[0],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()



def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    # ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            inputs, targets = get_batch(data_source, i, bptt)
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = eval_model(inputs)
            output_flat = output.view(-1, ntokens)
            total_loss += len(inputs) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


def learning_with_validation():
    best_val_loss = float("inf")
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(epoch)
        val_loss = evaluate(model, val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()
        torch.save(best_model.state_dict(), f'best-model-parameters-{map_set_number}.pt') 
        
    return best_model


def main():
    best_model = learning_with_validation()
    test_loss = evaluate(best_model, test_data)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))   
    
    # all data with 20 as batchsize
    d5 = torch.cat(d4)
    all_data = batchify(d5, 20)
    all_data_loss = evaluate(best_model, all_data)
    print('| End of training | all data loss {:5.2f} | all data ppl {:8.2f}'.format(
    all_data_loss, math.exp(all_data_loss)))  

    # best_model = model
    # best_model.load_state_dict(torch.load('best-model-parameters.pt'))
    # best_model.eval()
    import ipdb; ipdb.set_trace()
    print('train acc : {:.3f}'.format(get_accuracy(train_data, best_model, args)))
    print('val acc : {:.3f}'.format(get_accuracy(val_data, best_model, args)))
    print('test acc : {:.3f}'.format(get_accuracy(test_data, best_model, args)))
    # breakpoint()

    print('done')

main()