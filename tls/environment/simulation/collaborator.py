import numpy as np

from .observer import Observer
from .trafficlight import Trafficlight

from environment.constants import (
    MESH_SIZE,
    YELLOW_TIME,
    SIMULATION_STEP,
)


class Collaborator:
    r"""Responsible for interaction with SUMO simulation using TraCI API.

    Arguments:
        connection (Connection): connection to a TraCI-Server.
        trafficlight_skeletons (Dict[str, Union[str, dict]]): internal representation of the trafficlight areas.
    """
    def __init__(self, connection, trafficlight_skeletons, additional):
        self.connection = connection

        # Initialize `Trafficlight` objects for each observed trafficlight
        self.trafficlights = dict()
        self.observers = dict()
        for trafficlight_id, skeleton in trafficlight_skeletons.items():
            if trafficlight_id not in additional: continue  # Ensure that reward can be computed

            self.trafficlights[trafficlight_id] = \
                Trafficlight(connection, trafficlight_id, additional.get(trafficlight_id, None))
            self.observers[trafficlight_id] = Observer(connection, skeleton)

        self.simulation_time = 0

    @property
    def observation_space_shape(self):
        for observer in self.observers.values():
            return observer.current_observation.shape

    @property
    def action_space_shape(self):
        return self.available_actions,

    @property
    def available_actions(self):
        max_ = 0
        for trafficlight in self.trafficlights.values():
            num_actions = len(trafficlight.complete_phases)
            if max_ < num_actions:
                max_ = num_actions
        return max_

    def step(self, actions):
        r"""Applies actions of traffic light controller(s) and makes simulation step.

        Note:
            This method assumes that all traffic lights have the same yellow phase duration.
            Probably, later this should changed so that traffic lights with different lengths
            of the yellow signal will be supported.

        Arguments:
            actions: dictionary with action for each agent.
        """
        self._apply_actions(actions, prepare=True)

        self.simulation_time += YELLOW_TIME
        self.connection.simulationStep(step=self.simulation_time)

        self._apply_actions(actions)

        self.simulation_time += SIMULATION_STEP - YELLOW_TIME
        self.connection.simulationStep(step=self.simulation_time)

        observations = self.compute_observations()
        rewards = self.compute_rewards()
        done = {'__all__': self.connection.simulation.getMinExpectedNumber() == 0}
        info = {}

        return observations, rewards, done, info

    def _apply_actions(self, actions, prepare=False):
        r"""Changes the next state of every trafficlight in accordance
        with the action provided by the corresponding agent.

        Arguments:
            actions: dictionary with action for each agent.
            prepare: whether the state need to be actually changed (Default: False).
        """

        if prepare:
            for trafficlight_id, action in actions.items():
                self.trafficlights[trafficlight_id].set_next_phase(action)
        else:
            for trafficlight_id in actions:
                self.trafficlights[trafficlight_id].update_phase()

    def compute_observations(self):
        r"""Collects observations from each intersection.

        :return: dictionary with current observation for each intersection
        """

        observations = {}
        for trafficlight_id, observer in self.observers.items():
            agent_actions = len(self.trafficlights[trafficlight_id].complete_phases)

            action_mask = np.zeros(self.available_actions)
            action_mask[:agent_actions] = 1

            observations[trafficlight_id] = {
                'obs': observer.get_observation(),
                'action_mask': action_mask
            }

            if trafficlight_id == 'cluster_290051912_298136030_648538909':
                observer.print_current_observation()
        return observations

    def compute_rewards(self):
        r"""Collects rewards from each intersection.

        :return: dictionary with reward for the last action for each intersection
        """

        rewards = {}
        for trafficlight_id, trafficlight in self.trafficlights.items():
            try:
                rewards[trafficlight_id] = trafficlight.get_throughput()
            except ValueError:
                rewards[trafficlight_id] = 0
        return rewards

    def close(self):
        self.connection.close()
