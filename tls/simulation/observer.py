import numpy as np


class Observer:
    r"""Produces a snapshot of the current state of given trafficlight at any step of simulation.

    Arguments:
        simulation (Simulation): the traffic simulation.
        skeletons (dict{id: skeleton}): internal representation of trafficlights within the road network.
    """

    c = None  # Shortcomings of Python's import system

    def __init__(self, simulation, skeletons, constants):
        global c
        c = constants

        self.simulation = simulation
        self.skeletons = skeletons

    @staticmethod
    def _get_junction_shape( junction):
        # _, lane_id = junction[c.Position.BOTTOM]['elements']['lanes'][0][0]  # FixMe: Won't work if not intersection
        # lane_width = self.simulation.lane.getWidth(lane_id)  # Assume that all lanes have the same width
        lengths = {position: len(junction[position]['elements']['lanes']) for position in c.Position}
        width = max(lengths[position] for position in c.Position.horizontal())
        height = max(lengths[position] for position in c.Position.vertical())
        return width, height

    def get_state(self, trafficlight_id):
        r"""Produces a tensor obtained by staking matrices each of which represent current state
        of the specified trafficlight from different perspective.

        :param trafficlight_id: trafficlight ID
        :return:
        """

        trafficlight_skeleton = self.skeletons[trafficlight_id]
        mesh = np.zeros((c.MESH_SIZE, c.MESH_SIZE))

        # ToDO: Recursion vs. Queue for processing nested junctions?
        center = (15, 15)  # Revise

        shape = self._get_junction_shape(trafficlight_skeleton)
        bottom_left = np.subtract(center, (np.add(np.floor_divide(shape, 2), np.remainder(shape, 2))))

        for position in c.Position:
            direction, elements = trafficlight_skeleton[position].values()

            step = (0, 1)  # Revise
            for lanes in elements['lanes']:
                for offset, flow_direction, lane in lanes:
                    for vehicle in self.simulation.lane.getLastStepVehicleIDs(lane):
                        distance = self.simulation.vehicle.getLanePosition(vehicle)
                        distance = self.simulation.lane.getLength(lane) - distance if flow_direction else distance

                        idx = distance // c.MESH_PARTITIONING_STEP
                        print(idx, distance)

                bottom_left += step
            break
