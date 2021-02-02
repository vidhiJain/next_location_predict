import networkx as nx
import json
from pathlib import Path
import matplotlib.pyplot as plt
import copy 
from itertools import count

root = '../sparky_data'
room_segments_file = 'TestbedMap.json'

with open(Path(root, room_segments_file), 'r') as g:
    room_segments = json.load(g)
breakpoint()
# add room segments for upper and lower center hallway
room_segments['areas'].append({'id':'acha', 'name': 'Center Hallway', 'type': 'hallway', 'x1': -2139.571, 'y1': 174.3, 'x2': -2124.0, 'y2': 177.7})
room_segments['areas'].append({'id':'achb', 'name': 'Center Hallway', 'type': 'hallway', 'x1': -2125.0, 'y1': 174.3, 'x2': -2108.3, 'y2': 177.7})
room_segments['connections'].append({'id': 'cucl', 'name': 'Center Hallway upper and lower half', 'type': 'extension', 'x': -2125.0, 'y': 174.3, 'x2': -2124.0, 'y2': 177.7, 'area_1': 'acha', 'area_2': 'achb'})

id_index_rooms = {room_segments['areas'][i]['id']: i for i in range(len(room_segments['areas']))}
index_name_rooms = {value:room_segments['areas'][value]['name'] for _, value in id_index_rooms.items()}

# To check if player is in this room
def get_room_for_player(x, y):
    for segment in room_segments['areas']:
        if x > segment['x1'] and x < segment['x2']:
            if y > segment['y1'] and y < segment['y2']:
                return segment
    return None

with open('id_index_rooms.json', 'w') as f:
    json.dump(id_index_rooms, f)
with open('index_name_rooms.json', 'w') as f:
    json.dump(index_name_rooms, f)

# # process edges
# def get_edge_list():
#     edgelist = []
#     for conn in room_segments['connections']:
#         n1 = id_index_rooms[conn['area_1']]
#         n2 = id_index_rooms[conn['area_2']]
#         edgelist.append((n1, n2, conn))
#         edgelist.append((n2, n1, conn))
#     return edgelist


def mean_pos(i):
    x = (room_segments['areas'][i]['x2'] + room_segments['areas'][i]['x1'])/2
    y = (room_segments['areas'][i]['y2'] + room_segments['areas'][i]['y1'])/2
    return (y, x)

def mean_pos_for_graph(graph_node_attr):
    x = (graph_node_attr['x2'] + graph_node_attr['x1'])/2
    y = (graph_node_attr['y2'] + graph_node_attr['y1'])/2
    return (y, x)




def remove_edges(graph, a1,a2):
    if id_index_rooms[a1] < id_index_rooms[a2]:
        graph.remove_edge(id_index_rooms[a1], id_index_rooms[a2])
    else:
        graph.remove_edge(id_index_rooms[a2], id_index_rooms[a1])
    return graph


def add_edges(graph, a1, a2):
    if id_index_rooms[a1] < id_index_rooms[a2]:
        graph.add_edge(id_index_rooms[a1], id_index_rooms[a2])
    else:
        graph.add_edge(id_index_rooms[a2], id_index_rooms[a1])
    return graph


G = nx.Graph()
G.add_nodes_from([
    (i, room_segments['areas'][i]) for i in range(len(room_segments['areas']))
])   
G.add_edges_from([
    (id_index_rooms[conn['area_1']], id_index_rooms[conn['area_2']], conn) for conn in room_segments['connections']
])

# update based on perturbations
# wall_signs = find_locations(perturbed_map, index=44)
# blockages = find_locations(perturbed_map, index=22)
# hole_in_walls = find_locations(perturbed_map, index=11)
# Visualize
# pos = {room_segments['areas'][i]['id']: mean_pos(i) for i in range(len(room_segments['areas']))}
# # labels = {i: room_segments['areas'][i]['name'] for i in range(len(room_segments['areas']))}
# labels = {room_segments['areas'][i]['id']: room_segments['areas'][i]['name'] for i in range(len(room_segments['areas']))}

plt.clf()
pos = {x: mean_pos_for_graph(G.nodes[x]) for x in G.nodes}
labels = {x: G.nodes[x]['name'] for x in G.nodes}
nx.draw(G, pos, labels=labels, font_size=8, node_shape='s')
plt.savefig('sparky_base_map')
nx.write_edgelist(G, "sparky_base_map.edgelist")

G0 = copy.deepcopy(G)
G0.remove_edge(id_index_rooms['arha'], id_index_rooms['arhb'])
G0.remove_edge(id_index_rooms['ar215'], id_index_rooms['ar220'])
G0.remove_edge(id_index_rooms['alhb'], id_index_rooms['ar203'])
G0.add_edge(id_index_rooms['ar203'], id_index_rooms['ar208'])
G0.add_edge(id_index_rooms['ar209'], id_index_rooms['ar216'])
G0.add_edge(id_index_rooms['ar201'], id_index_rooms['ar203'])
G0.add_edge(id_index_rooms['ar211'], id_index_rooms['ar213'])
pos = {x: mean_pos_for_graph(G0.nodes[x]) for x in G0.nodes}
labels = {x: G0.nodes[x]['name'] for x in G0.nodes}
types = set(nx.get_node_attributes(G0, 'name').values())
mapping = dict(zip(sorted(types), count()))
colors = [mapping[G0.nodes[x]['name']] for x in G0.nodes]
plt.clf()
nx.draw(G0, pos, labels=labels, font_size=8, node_shape='s', node_color=colors, cmap='YlGn')
plt.savefig('sparky_map_0')
nx.write_edgelist(G0, "sparky_map_0.edgelist")

G1 = copy.deepcopy(G)
G1.remove_edge(id_index_rooms['alha'], id_index_rooms['ar201'])
G1.remove_edge(id_index_rooms['arhb'], id_index_rooms['ar218'])
G1.add_edge(id_index_rooms['ar203'], id_index_rooms['ar208'])
G1.add_edge(id_index_rooms['ar201'], id_index_rooms['ar203'])
G1.add_edge(id_index_rooms['ar211'], id_index_rooms['ar213'])
pos = {x: mean_pos_for_graph(G1.nodes[x]) for x in G1.nodes}
labels = {x: G1.nodes[x]['name'] for x in G1.nodes}
types = set(nx.get_node_attributes(G1, 'name').values())
mapping = dict(zip(sorted(types), count()))
colors = [mapping[G1.nodes[x]['name']] for x in G1.nodes]
plt.clf()
nx.draw(G1, pos, labels=labels, font_size=8, node_shape='s', node_color=colors, cmap='OrRd')
plt.savefig('sparky_map_1')
nx.write_edgelist(G1, "sparky_map_1.edgelist")

G2 = copy.deepcopy(G)
G2.remove_edge(id_index_rooms['ar207'], id_index_rooms['ar210'])
G2.remove_edge(id_index_rooms['ar215'], id_index_rooms['ar220'])
G2.add_edge(id_index_rooms['ar201'], id_index_rooms['ar203'])
G2.add_edge(id_index_rooms['ar203'], id_index_rooms['ar208'])
G2.add_edge(id_index_rooms['ar209'], id_index_rooms['ar216'])
G2.remove_node(id_index_rooms['ach'])
G2.remove_edge(id_index_rooms['acha'], id_index_rooms['achb'])
G2.add_edge(id_index_rooms['acha'], id_index_rooms['ajc'])
G2.add_edge(id_index_rooms['acha'], id_index_rooms['ar208'])
G2.add_edge(id_index_rooms['acha'], id_index_rooms['ar211'])
G2.add_edge(id_index_rooms['acha'], id_index_rooms['ar209'])
G2.add_edge(id_index_rooms['achb'], id_index_rooms['ar208'])
G2.add_edge(id_index_rooms['achb'], id_index_rooms['ar210'])
G2.add_edge(id_index_rooms['achb'], id_index_rooms['ar215'])

pos = {x: mean_pos_for_graph(G2.nodes[x]) for x in G2.nodes}
labels = {x: G2.nodes[x]['name'] for x in G2.nodes}
types = set(nx.get_node_attributes(G2, 'name').values())
mapping = dict(zip(sorted(types), count()))
colors = [mapping[G2.nodes[x]['name']] for x in G2.nodes]
plt.clf()
nx.draw(G2, pos, labels=labels, font_size=8, node_shape='s', node_color=colors, cmap='RdPu')
plt.savefig('sparky_map_2')
nx.write_edgelist(G2, "sparky_map_2.edgelist")
