import pandas as pd
import json
import numpy as np
from pathlib import Path
from pprint import pprint
from glob import glob

import arguments

args = arguments.get_args()
map_set_num = 0
time_to_stop = 1000 #

root = '../sparky_data'
messages = 'Message_bus_data'
events = 'Events'

room_segments_file = 'TestbedMap.json'
with open(Path(root, room_segments_file), 'r') as g:
    room_segments = json.load(g)
name2index = {room_segments['areas'][i]['name']: i for i in range(len(room_segments['areas']))}

# participant_list = ['g0614', 'qi0619', 'seth0619', 'yan0605', 'zi0619', ]
# participant_list = glob(Path(root, events, f'*_{str(map_set_num)}_location.csv').as_posix())
participant_list = glob(Path(root, messages, f'*_{str(map_set_num)}.txt').as_posix())

data_record = []

with open('id_index_rooms.json', 'r') as f:
    id_index_rooms = json.load(f)

def get_room_for_player(x, y):
    for segment in room_segments['areas']:
        if x >= round(segment['x1']) and x <= round(segment['x2']):
            if y >= round(segment['y1']) and y <= round(segment['y2']):
                return segment
    return None

record_when_room_none = []
participant_record = []

for filename in participant_list:
    data = []
    prev_idx = 0
    print(filename)
    with open(filename, 'r') as f:
        for d in f.readlines():
            data.append(json.loads(d))
    data = sorted(data, key=lambda i: float(i['header']['timestamp']))
    prev_timestamp = float(data[0]['header']['timestamp'])
    record = []

    for entry in data:
        if entry['header']['message_type']=='observation':
            x = entry['data']['x']
            z = entry['data']['z']

            if entry['header']['timestamp'] is not None and float(entry['header']['timestamp']) >= prev_timestamp:
                prev_timestamp = float(entry['header']['timestamp']) 
            else:
                breakpoint()
            
            if float(entry['header']['timestamp']) > time_to_stop:
                break

            # Get current room
            segment = get_room_for_player(x, z)
            if segment == None:
                record_when_room_none.append(entry)
                continue
            cur_area = segment['id']
            cur_idx = id_index_rooms[cur_area]
            
            if cur_idx == 0: 
                continue
            print('Timestamp: ', entry['header']['timestamp'], ', Player in ', segment['name'])
            # breakpoint()
            # Adding to the current index
            if prev_idx != cur_idx:
                record.append(cur_idx)
                
            prev_idx = cur_idx

    participant_record.append(record)
# def nextElem(arr, target): 
#     # https://www.geeksforgeeks.org/first-strictly-greater-element-in-a-sorted-array-in-java/ 
#     start = 0; 
#     end = len(arr) - 1; 
  
#     ans = -1; 
#     while (start <= end): 
#         mid = (start + end) // 2; 
  
#         # Move to right side if target is 
#         # greater. 
#         if (arr[mid] <= target): 
#             start = mid + 1; 
  
#         # Move left side. 
#         else: 
#             ans = mid; 
#             end = mid - 1; 
  
#     return ans

# for filename in participant_list:
#     # filename = participant_id + '_' + str(map_set_num) + '_location.csv'

#     df = pd.read_csv(filename)
#     room_seq = df['Enter'].values
#     timeindex = nextElem(df['Timestamp'].values, time_to_stop)
#     # breakpoint()
#     room_seq_short = room_seq[:timeindex]
#     debug_ = [room for room in room_seq]
#     pprint(debug_)
#     int_seq = [name2index[room] for room in room_seq_short]
#     data_record.append(int_seq)



# roomseqlen = [len(record) for record in participant_record]
# max_len = max(roomseqlen)
# data_np = np.zeros((len(participant_record), max_len), dtype=np.uint8)
# print(data_np.shape)
# for i in range(len(participant_record)):
#     for j in range(len(participant_record[i])):
#         data_np[i][j] = participant_record[i][j]

# pd.DataFrame(data_np).to_csv(f'resources/human_room_nav_sparky_{str(map_set_num)}_till_{str(time_to_stop)}.csv', header=None, index=False)

import csv 
with open(f'resources/human_room_nav_{str(map_set_num)}.csv', 'w') as f:
    wr = csv.writer(f)
    wr.writerows(participant_record)
    # for record in participant_record:
        # f.write(''.join(str(record)))
            # f.write('\n')
# breakpoint()
import ipdb; ipdb.set_trace()
print('Human room nav sequence saved')