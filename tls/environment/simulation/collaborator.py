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
            self.trafficlights[trafficlight_id] = \
                Trafficlight(connection, trafficlight_id, additional.get(trafficlight_id, None))
            self.observers[trafficlight_id] = Observer(connection, skeleton)

        self.simulation_time = 0

    @staticmethod
    def get_observation_space_shape():
        # ToDo: Replace with `get_current_shape` call
        return MESH_SIZE, MESH_SIZE, 1

    @staticmethod
    def get_action_space_shape():
        # ToDo: Replace; Should return the maximum number of actions and masks
        # For reference github.com/ray-project/ray/blob/master/python/ray/rllib/examples/parametric_action_cartpole.py
        return 3

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
            observations[trafficlight_id] = observer.get_observation()
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
