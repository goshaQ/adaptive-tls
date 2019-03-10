import os
import sys
import ast
import argparse
import time

import numpy as np

import constants
import network_utils

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import sumolib
import traci

parser = argparse.ArgumentParser(description="Test program")
parser.add_argument('config_path', type=str,
                    help='The configuration file to be used')

args = parser.parse_args()

cmd = ['sumo-gui', '--start', '--quit-on-end',  '-c', args.config_path]
# cmd = ['sumo-gui', '--start', '-c', args.config_path]

tt = traci.trafficlight  # ?
tj = traci.junction  # ?
tl = traci.lane  # ?
traci.start(cmd)

trafficlight_id_list = tt.getIDList()

some_id = 'cluster_290051904_49145925'
curr_program_idx = int(tt.getProgram(some_id))
curr_phase_idx = int(tt.getPhase(some_id))

curr_phase = tt.getCompleteRedYellowGreenDefinition(some_id)[curr_program_idx].getPhases()[curr_phase_idx]
curr_phase = '{' + ', '.join([': '.join(["'" + val.replace(' ', '') + "'" for val in entry.split(':')])
                              for entry in str(curr_phase).split('\n')[:-1]]) + '}'
curr_phase = ast.literal_eval(curr_phase)  # WTF?

# ToDo: Think about how to find all lanes on which movement is allowed

# ToDo: Finish; Generate dict() of arrangement of lanes for each trafficlight junction

# ToDo: There should be a way to sort lanes so that opposite lanes are distinguishable

#########################
path = '/home/gosha/Загрузки/network/moco.net.xml'  # To be replaced
net = sumolib.net.readNet(path)
#########################

mesh = np.zeros((constants.MESH_SIZE, constants.MESH_SIZE))

# for trafficlight_id, arranged in network_utils.get_arranged_junctions(net).items():
#     print(trafficlight_id)
#     network_utils.get_arranged_lanes(net, arranged)
#     break

problematic_trafficlights = [
    'cluster_290051912_298136030_648538909',
    'cluster_2511020102_2511020103_290051922_298135886',
    'cluster_2511020106_49297289',
]

skeletons = {}
for trafficlight in net.getTrafficLights():
    if trafficlight.getID() in problematic_trafficlights: continue
    
    trafficlight_skeleton = network_utils.extract_tl_skeleton(net, trafficlight)
    skeletons[trafficlight.getID()] = trafficlight_skeleton

step = 0
while step < 100:
    traci.simulationStep()
    step += 1
# time.sleep(10)

#########################
from simulation import Simulation
from simulation.observer import Observer

conn = traci.getConnection()  # conn.trafficlight, etc.
sim = Simulation(conn)

observer = Observer(sim, skeletons, constants)
observer.get_state('cluster_290051904_49145925')

#########################

# some_id = '2511020105'
# some_node = net.getNode(some_id)
# junction_type = network_utils.get_junction_type(some_node)
# print(junction_type)

# ToDo: Think about graph traversal; probably there is a need to colorize the edges and vertexes; REMOVED

# print(lane_id_list)
# print(tl.getControlledLinks(some_id))
# print(len(set(tl.getControlledLanes(some_id))))
# print(tl.getProgram(some_id))
# print(tl.getControlledLanes(some_id)[0])
# print(jn.getShape(some_id))

# some_id = '116069075#0.376_0'
#
# step = 0
# while step < 10000:
#     traci.simulationStep()
#     time.sleep(0.05)
#     # if step == 100:
#     #     print(ln.getLength(some_id))
#     #     print(ln.getLastStepVehicleIDs(some_id))
#     #     break
#     step += 1

traci.close(wait=False)
