import os
import sys

from processing import network_utils
from simulation.collaborator import Collaborator

from ray.rllib.env.multi_agent_env import MultiAgentEnv

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import tools.sumolib as sumolib
import tools.traci as traci

_PROBLEMATIC_TRAFFICLIGHTS = [
    'cluster_290051912_298136030_648538909',
    'cluster_2511020102_2511020103_290051922_298135886',
]


class Env(MultiAgentEnv):
    def __init__(self, net_file, config_file, use_gui=True, single_agent=True):
        r"""SUMO Environment for Adaptive Traffic Light Control.

        Arguments:
            net_file (str): SUMO .net.xml file.
            config_file (str): SUMO .config file.
            use_gui (bool): whether to run SUMO simulation with GUI visualisation.
            single_agent (bool): whether the environment is single or multi agent.
        """

        self.trafficlight_skeletons = {}
        self.trafficlight_ids = []
        self.action_space = None
        self.observation_space = None
        self.collaborator = None

        net = sumolib.net.readNet(net_file)

        # Preprocess the network definition and produce internal representation of each trafficlight
        for trafficlight in net.getTrafficLights():
            id_ = trafficlight.getID()
            if id_ in _PROBLEMATIC_TRAFFICLIGHTS: continue

            self.trafficlight_skeletons[id_] = network_utils.extract_tl_skeleton(net, trafficlight)
            self.trafficlight_ids.append(id_)

        # Define the command to run SUMO simulation
        if use_gui:
            sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            sumo_binary = sumolib.checkBinary('sumo')
        self._sumo_cmd = [sumo_binary, '--start', '--quit-on-end', '-c', config_file]

    def step(self, action_dict):
        pass

    def reset(self):
        traci.start(self._sumo_cmd)

        # Reinitialize the collaborator since the connection changed
        self.collaborator = Collaborator(traci.getConnection(), self.trafficlight_skeletons)
        return self.collaborator.get_observations()
