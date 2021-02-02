import pandas as pd
import json
from pathlib import Path
import networkx as nx
import torch
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import copy
import math
import pandas as pd

root = '../sparky_data'
dp = 'decision_points'
messages = 'Message_bus_data'
events = 'Events'
fov = 'FOV_events'

filename = 'luo0603_0.txt'
G = nx.read_edgelist('resources/sparky_map_0.edgelist')
room_segments_file = 'TestbedMap.json'
triage_events = pd.read_csv(Path(root, events, filename.split('.')[0]+'_triage.csv'))
fov_victim_events = pd.read_csv(Path(root, fov, filename.split('.')[0]+'_victim_fov.csv'))


with open(Path(root, room_segments_file), 'r') as g:
    room_segments = json.load(g)

with open('id_index_rooms.json', 'r') as f:
    id_index_rooms = json.load(f)

with open('index_name_rooms.json', 'r') as f:
    index_name_rooms = json.load(f)
# perturbed_map_file = 'resources/map_set_0.npy'

data = []
with open(Path(root, messages, filename), 'r') as f:
    for d in f.readlines():
        data.append(json.loads(d))
# breakpoint()
data = sorted(data, key=lambda i: float(i['header']['timestamp']))


# id_index_rooms = {room_segments['areas'][i]['id']: i for i in range(len(room_segments['areas']))}
# index_name_rooms = {value:room_segments['areas'][value]['name'] for _, value in id_index_rooms.items()}

# To check if player is in this room
def get_room_for_player(x, y):
    for segment in room_segments['areas']:
        if x >= round(segment['x1']) and x <= round(segment['x2']):
            if y >= round(segment['y1']) and y <= round(segment['y2']):
                return segment
    return None


# process edges
def get_edge_list():
    edgelist = []
    for conn in room_segments['connections']:
        n1 = id_index_rooms[conn['area_1']]
        n2 = id_index_rooms[conn['area_2']]
        edgelist.append((n1, n2, conn))
        edgelist.append((n2, n1, conn))
    return edgelist

def mean_pos(room_bb):
    x = (room_bb['x2'] + room_bb['x1'])/2
    y = (room_bb['y2'] + room_bb['y1'])/2
    return (x, y)


def get_width_height(room_bb):
    w = (room_bb['x2'] - room_bb['x1'])
    h = (room_bb['y2'] - room_bb['y1'])
    return (w, h)
    

def distance_to_room(agent_pos, room_bb):
    center_x, center_y = mean_pos(room_bb)
    width, height = get_width_height(room_bb)
    dx = max(abs(agent_pos['x'] - center_x) - width / 2, 0);
    dy = max(abs(agent_pos['z'] - center_y) - height / 2, 0);
    return dx * dx + dy * dy


noise = 1e-4
prev_idx = 0
degree_coefficient = 2
prob_entry_required_for_nav = 0.1
prob_returning_to_visited = 0.1
prob_entry_with_green_before_yellow = 0.1
visited = []
record_when_room_none = []
sequence_of_rooms_visited = []
record = []
rooms_with_green_seen = []
# triage_events = triage_events.set_index('Timestamp')
triage_events_counter = 0
fov_victim_events_counter = 0


for entry in data:
    if entry['header']['message_type']=='observation':
        x = entry['data']['x']
        z = entry['data']['z']
        if float(entry['header']['timestamp']) > 400:
            break
        if float(entry['header']['timestamp']) > fov_victim_events.iloc[fov_victim_events_counter]['timestamp']:
            fov_victim_events_counter += 1
        
        # Get current room
        segment = get_room_for_player(x, z)
        if segment == None:
            record_when_room_none.append(entry)
            continue
        cur_area = segment['id']
        cur_idx = id_index_rooms[cur_area]
        
        if cur_idx == 0: 
            continue
        print('Player in ', segment['name'])
        # breakpoint()

        # Fetch edge connected rooms (and their connections?)
        possible_idx = []

        # updating the probabilities
        likelihood = np.zeros(len(room_segments['areas'])) #id_index_rooms
        

        for edge in G.edges:
            if int(edge[0]) == cur_idx:
                # possible_idx.append(edge[1])
                # weigh based on node degree
                likelihood[int(edge[1])] =  G.degree[str(edge[1])]
                likelihood[int(edge[1])] *= 1./(distance_to_room(entry['data'], room_segments['areas'][int(edge[1])]) + noise)

                # check if untriaged victims here uptil triage counter only to avoid cheating


            elif int(edge[1]) == cur_idx:
                # possible_idx.append(edge[0])
                # weigh based on node degree
                likelihood[int(edge[0])] = G.degree[str(edge[0])]
                likelihood[int(edge[0])] *= 1./(distance_to_room(entry['data'], room_segments['areas'][int(edge[0])]) + noise)

            else:
                continue

        # TODO: untriaged victims in the area increase
        

        for i in range(fov_victim_events_counter):
            if fov_victim_events.iloc[i]['type'] == 'Green':
                room = get_room_for_player(fov_victim_events.iloc[i]['x'], fov_victim_events.iloc[i]['z'])
                if room is None:
                    breakpoint()
                likelihood[id_index_rooms[room['id']]] *= prob_entry_with_green_before_yellow
                # do this after yellow victims triaged
                rooms_with_green_seen.append(room)
            if fov_victim_events.iloc[i]['type'] == 'Unknown':
                room = get_room_for_player(fov_victim_events.iloc[i]['x'], fov_victim_events.iloc[i]['z'])
                if room is None:
                    breakpoint()
                likelihood[id_index_rooms[room['id']]] *= prob_entry_required_for_nav

        # previous visited area without victims, decrease depending on node-degree
        for node in visited:
            if node['type']=='room':
                likelihood[id_index_rooms[node['id']]] *= prob_returning_to_visited
        
        
        prob_vector = likelihood/np.sum(likelihood)

        print(prob_vector)
        prob_places = []
        for i, prob in enumerate(prob_vector):
            if prob:
                print('Likely to go to ', index_name_rooms[str(i)], prob)
                prob_places.append((index_name_rooms[str(i)], prob))
        probmax = np.argmax(prob_vector)
        # Adding to the current index
        if prev_idx != cur_idx:
            visited.append(segment)
            sequence_of_rooms_visited.append( {segment['name']: prob_places})
            record.append((segment['id'],probmax))
        prev_idx = cur_idx


def accuracy(record):
    correct = 0
    total = len(record)
    predicted_room_index = id_index_rooms['achl']
    for count, entry in enumerate(record):
        key, val = entry
        if count == 0:
            # next_room_index = id_index_rooms[key]  
            predicted_room_index = val
            continue
         
        next_room_index = id_index_rooms[key] 
        if predicted_room_index == next_room_index:
            correct += 1
        predicted_room_index = val
    print('acc: ', correct/total)
    return correct/total


pprint(sequence_of_rooms_visited)
accuracy(record)
breakpoint()
print('done')
    # class HumanDataLoader(torch.utils.data):
    #     """
    #     inputs : current room index, victims triaged in current room, last two room indices
    #     target : next room entered index
    #     """
    #     def __init__(self):
    #         pass

    #     def __len__(self):
    #         return 

    #     def __getitem__(self, index):
    #         return inputs, target