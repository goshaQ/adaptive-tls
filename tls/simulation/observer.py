import numpy as np
from pprint import pprint


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
        return height, width

    def get_state(self, trafficlight_id, display=True):
        r"""Produces a tensor obtained by staking matrices each of which represent current state
        of the specified trafficlight from different perspective.

        :param trafficlight_id: trafficlight ID
        :return:
        """

        trafficlight_skeleton = self.skeletons[trafficlight_id]
        mesh = np.zeros((c.MESH_SIZE, c.MESH_SIZE))
        color = np.zeros((c.MESH_SIZE, c.MESH_SIZE))
        pprint(trafficlight_skeleton)

        # ToDO: Recursion vs. Queue for processing nested junctions?
        center = (15, 15)  # Revise
        rotate = (1, -1)

        shape = self._get_junction_shape(trafficlight_skeleton)
        bottom_left = np.add(center, np.multiply(rotate, np.add(np.floor_divide(shape, 2), np.remainder(shape, 2))))

        # Specify in which direction the cursor is moving
        steps = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        positions = c.Position

        for position, step in zip(positions, steps):
            direction, elements = trafficlight_skeleton[position].values()

            for lanes in elements['lanes']:
                for offset, flow_direction, lane in lanes:
                    for vehicle in self.simulation.lane.getLastStepVehicleIDs(lane):
                        # Get position of a car on the road as distance from the trafficlight
                        distance = self.simulation.vehicle.getLanePosition(vehicle)
                        if flow_direction:
                            distance = self.simulation.lane.getLength(lane) - distance

                        # Find idx of the cell relative to the cursor position
                        idx = np.int64((offset + distance) // c.MESH_PARTITIONING_STEP)

                        # Ensure that the car is within the observation area
                        if not 0 <= idx < c.MESH_SIZE:
                            continue

                        # Find idx of the cell on the grid
                        idx = np.add(bottom_left, np.multiply(np.flip(step, axis=0), idx))
                        mesh[tuple(idx)] = 1

                    # Save lanes to the color layer
                    from_ = np.int64(offset // c.MESH_PARTITIONING_STEP)
                    to_ = np.int64((offset + self.simulation.lane.getLength(lane)) // c.MESH_PARTITIONING_STEP)

                    # ToDo: Try to subtract 'to_' from 'bottom_left[1]'
                    if step.index(0):
                        from_, to_ = bottom_left[1] + min(from_, to_), bottom_left[1] + max(from_, to_)
                        color[bottom_left[0], from_:min(to_, c.MESH_SIZE - 1)] = 1
                    print(f'from {from_} to {to_}')
                    print(bottom_left)
                    print(offset, flow_direction, lane)
                bottom_left += step
            break

        np.set_printoptions(threshold=np.nan, linewidth=np.nan)
        print(color)

    def _print_mesh(self, mesh):
        pass

