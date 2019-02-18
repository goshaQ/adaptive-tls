import numpy as np
from pprint import pprint
from string import Template


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
    def _get_junction_polygon(junction):
        lengths = {position: len(junction[position]['elements']['lanes']) for position in c.Position}
        minh, maxh = sorted(lengths[position] for position in c.Position.horizontal())
        minw, maxw = sorted(lengths[position] for position in c.Position.vertical())

        rotate = {
            c.Position.LEFT: (1, -1), c.Position.RIGHT: (-1, 1),
            c.Position.TOP: (-1, -1), c.Position.BOTTOM: (1, 1),
        }

        center = ((c.MESH_SIZE - 1) // 2, (c.MESH_SIZE - 1) // 2)
        hoffset, voffset = np.int64((maxw % 2, 1))

        result = {}
        for position in c.Position:
            if position in c.Position.horizontal():
                result[position] = (
                    np.add(center, np.multiply(rotate[position], (lengths[position] // 2, hoffset + maxw // 2))))
                hoffset = 1
            else:
                result[position] = (
                    np.add(center, np.multiply(rotate[position], (voffset + maxh // 2, lengths[position] // 2))))
                voffset = maxh % 2
        return result

    @staticmethod
    def _clamp(*args, **kwargs):
        return [np.maximum(kwargs['min'], np.minimum(kwargs['max'], arg)) for arg in args]

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
        polygon = self._get_junction_polygon(trafficlight_skeleton)

        rotate = (-1, 1)
        b_offset = (0, 1)

        # Specify in which direction the cursor is moving
        steps = {
            c.Position.LEFT: (-1, 0), c.Position.RIGHT: (1, 0),
            c.Position.TOP: (0, 1), c.Position.BOTTOM: (0, -1),
        }

        for position in c.Position:
            step, cursor = steps[position], polygon[position]
            direction, elements = trafficlight_skeleton[position].values()

            # FixMe: Cursor doesn't move in the right direction
            for lanes in elements['lanes']:
                for offset, flow_direction, lane in lanes:
                    for vehicle in self.simulation.lane.getLastStepVehicleIDs(lane):
                        # Get position of a car on the road as distance from the trafficlight
                        distance = self.simulation.vehicle.getLanePosition(vehicle)
                        if flow_direction:
                            distance = self.simulation.lane.getLength(lane) - distance

                        # Find idx of the cell on the grid
                        idx = np.add(cursor, np.multiply(
                            np.int64((offset + distance) // c.MESH_PARTITIONING_STEP),
                            np.multiply(rotate, np.flip(step, axis=0))))

                        # Ensure that the car is within the observation area
                        if not np.array_equal(idx, self._clamp(idx, min=0, max=c.MESH_SIZE - 1)[0]):
                            continue
                        mesh[tuple(idx)] = 1

                    # Save lanes to the color layer
                    from_, to_ = self._clamp(
                        np.add(cursor, np.multiply(
                            np.int64(offset // c.MESH_PARTITIONING_STEP),
                            np.multiply(rotate, np.flip(step, axis=0)))),
                        np.add(cursor, np.multiply(
                            np.int64((offset + self.simulation.lane.getLength(lane)) // c.MESH_PARTITIONING_STEP),
                            np.multiply(rotate, np.flip(step, axis=0)))),
                        min=0, max=c.MESH_SIZE - 1)

                    tmp = np.column_stack((from_, to_))
                    idx = (tmp[0], slice(*np.add(b_offset, np.sort(tmp[1]))))\
                        if position in c.Position.horizontal() else (slice(*np.add(b_offset, np.sort(tmp[0]))), tmp[1])

                    color[idx] = 1 if flow_direction else -1
                cursor += step

        # Save junction to the color layer
        # tmp = np.column_stack(list(polygon.values()))
        # tmp = np.column_stack((np.amax(tmp, axis=1), np.amin(tmp, axis=1)))
        #
        # idx = (slice(*np.add(np.sort(tmp[0]), b_offset[::-1])), slice(*np.add(np.sort(tmp[1]), b_offset[::-1])))
        # color[idx] = 2

        # Save car positions to the color layer
        color[np.where(mesh == 1)] = 9
        self._print_mesh(color)

    @staticmethod
    def _print_mesh(mesh):
        _TURQUOISE = Template('\x1b[0;36;40m$str\x1b[0m')
        _WHITE = Template('\x1b[0;30;46m$str\x1b[0m')
        _GRAY = Template('\x1b[0;30;47m$str\x1b[0m')
        _PURPLE = Template('\x1b[0;30;45m$str\x1b[0m')

        for row in mesh:
            sep = ''

            row_string = ''
            for d in row:
                d_string = sep

                if not sep:
                    sep = ' '

                if d == 9:
                    d_string += '1.'

                    row_string += _PURPLE.substitute(str=d_string)
                else:
                    d_string += str(abs(d)).rstrip('0')

                    if d == 1:
                        row_string += _TURQUOISE.substitute(str=d_string)
                    elif d == -1:
                        row_string += _WHITE.substitute(str=d_string)
                    # elif d == 2:
                    #     row_string += _GRAY.substitute(str=d_string)
                    else:
                        row_string += d_string
            print(f'[{row_string}]')
