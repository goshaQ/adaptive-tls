from .observer import Observer
from .trafficlight import Trafficlight

from constants import (
    YELLOW_TIME,
    SIMULATION_STEP,
)


class Collaborator:
    r"""Responsible for interaction with SUMO simulation using TraCI API.

    Arguments:
        connection (Connection): connection to a TraCI-Server.
        trafficlight_skeletons (Dict[str, Union[str, dict]]): internal representation of the trafficlight areas.
    """
    def __init__(self, connection, trafficlight_skeletons):
        self.connection = connection
        self.observer = Observer(connection, trafficlight_skeletons)

        # Initialize `Trafficlight` objects for each observed trafficlight
        self.trafficlights = dict()
        for trafficlight_id in trafficlight_skeletons:
            self.trafficlights[trafficlight_id] = Trafficlight(connection, trafficlight_id)

        self.simulation_time = 0

    def step(self):
        r"""Applies actions of traffic light controller(s) and makes simulation step.

        Note:
            This method assumes that all traffic lights have the same yellow phase duration.
            Probably, later this should changed so that traffic lights with different lengths
            of the yellow signal will be supported.
        """
        # action = self.agent_action_stub()
        # self.trafficlights['cluster_298135838_49135231'].set_next_phase(action)

        self.simulation_time += YELLOW_TIME
        self.connection.simulationStep(step=self.simulation_time)

        self.trafficlights['cluster_298135838_49135231'].update_phase()

        self.simulation_time += SIMULATION_STEP - YELLOW_TIME
        self.connection.simulationStep(step=self.simulation_time)

    def get_observations(self):
        r"""Collects observations from each intersection.

        :return: dictionary with current observation for each intersection
        """

        observations = {}
        for trafficlight_id in self.trafficlights:
            observations[trafficlight_id] = self.observer.get_observation(trafficlight_id, display=False)
        return observations
