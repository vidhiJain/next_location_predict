import numpy as np
import pickle as pkl
from pathlib import Path
import json
import pandas as pd
origin = {'x': -2155.5, 'y': 51.5, 'z': 152.5}

# root1 = '/serverdata/tejus/asist_agent/examples/'
# filename = 'trajs_beep_False_opportunistic.pkl'
root1 = '/serverdata/tejus/data/faux-human'
filelist = [
    'trajs_beep_False_opportunistic_sparky0.pkl',
    'trajs_beep_False_opportunistic_sparky1.pkl',
    'trajs_beep_False_opportunistic_sparky2.pkl',
    'trajs_beep_False_yellow-first_sparky0.pkl',
    'trajs_beep_False_yellow-first_sparky1.pkl',
    'trajs_beep_False_yellow-first_sparky2.pkl',
    'trajs_beep_True_opportunistic_sparky0.pkl',
    'trajs_beep_True_opportunistic_sparky1.pkl',
    'trajs_beep_True_opportunistic_sparky2.pkl',
    'trajs_beep_True_yellow-first_sparky0.pkl',
    'trajs_beep_True_yellow-first_sparky1.pkl',
    'trajs_beep_True_yellow-first_sparky2.pkl',
]
target_folder = 'faux_human_room_nav3'
root2 = '../sparky_data'
room_segments_file = 'TestbedMap.json'


with open(Path(root2, room_segments_file), 'r') as g:
    room_segments = json.load(g)
id_index_rooms = {room_segments['areas'][i]['id']: i for i in range(len(room_segments['areas']))}
index_name_rooms = {value:room_segments['areas'][value]['name'] for _, value in id_index_rooms.items()}

# To check if player is in this room
def get_room_for_player(x, y):
    for segment in room_segments['areas']:
        if x >= round(segment['x1']) and x <= round(segment['x2']):
            if y >= round(segment['y1']) and y <= round(segment['y2']):
                return segment
    return None


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


# df = pd.DataFrame()
def generate_room_seq(filename):
    with open(Path(root1, filename), 'rb') as f:
        data = pkl.load(f)
    
    count_unsegmented_rooms = 0
    prev_idx = 1

    for traj in data:
        record_every_room_change = []
        for i in range(1400):
            x = traj['position'][i][0] + origin['x']
            z = traj['position'][i][1] + origin['z']
            segment = get_room_for_player(x, z)
            
            if segment is None:
                count_unsegmented_rooms += 1
                continue
                # breakpoint()
            
            cur_idx = segment['id']
            if prev_idx != cur_idx:
                prev_idx = cur_idx
                # print('Player is in ', segment['name'])
                record_every_room_change.append(str(id_index_rooms[segment['id']]))

        with open(Path(target_folder,'room_'+filename.split('.')[0]), 'a') as f:
            f.write(','.join(record_every_room_change))
            f.write('\n')
            
        print(record_every_room_change, count_unsegmented_rooms)


def main():
    for filename in filelist:
        generate_room_seq(filename)
        # print(record_every_room_change, count_unsegmented_rooms)


main()
breakpoint()
print('done~')
