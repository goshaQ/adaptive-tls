import numpy as np
from environment import constants as c

from string import Template
from collections import deque


class Observer:
    r"""Produces a snapshot of the current state of assigned trafficlight at any step of simulation.

    Arguments:
        connection (Connection): connection to a TraCI-Server.
        trafficlight_skeleton (dict{id: skeleton}): internal representation of trafficlights within the road network.
    """

    # Center of the regulated intersection
    _CENTER = ((c.MESH_SIZE - 1) // 2,) * 2

    # Some auxiliary multipliers to rotate segments
    _ROTATE = {
        c.Position.LEFT: (1, -1),
        c.Position.RIGHT: (-1, 1),
        c.Position.TOP: (-1, -1),
        c.Position.BOTTOM: (1, 1),
    }

    # Directions in which the cursor is moving
    _STEPS = {
        c.Position.LEFT: (-1, 0),
        c.Position.RIGHT: (1, 0),
        c.Position.TOP: (0, 1),
        c.Position.BOTTOM: (0, -1),
    }

    def __init__(self, connection, trafficlight_skeleton):
        self.connection = connection
        self.current_observation = None

        self.trafficlight_id = trafficlight_skeleton['id']
        self.trafficlight_skeleton = trafficlight_skeleton

        self._buffered_shapes = {}
        self._init_color_layer()

    def _init_color_layer(self):
        self.color_layer = np.zeros((c.MESH_SIZE,)*2)
        self.topology = list()

        queued_junctions = deque()
        queued_junctions.append((self.trafficlight_skeleton, (0, 0)))

        while queued_junctions:
            self.topology.append(queued_junctions[-1])

            junction, relative_offset = queued_junctions.pop()
            junction_id = junction['id']

            polygon = self._get_junction_polygon(junction, relative_offset)

            for position, segment in junction['segments'].items():
                if segment['lanes'][0] is None: continue  # Skip if already added to the layer

                max_lane_length = 0
                step, cursor = Observer._STEPS[position], polygon[position].copy()
                for lanes in segment['lanes']:
                    for offset, flow_direction, lane in lanes:
                        # Find the start and end coordinate of the lane
                        from_, to_ = (
                            np.add(cursor, np.multiply(
                                np.int64(offset // c.MESH_PARTITIONING_STEP),
                                np.multiply((-1, 1), np.flip(step, axis=0)))),
                            np.add(cursor, np.multiply(
                                np.int64(
                                    (offset + self.connection.lane.getLength(lane)) // c.MESH_PARTITIONING_STEP),
                                np.multiply((-1, 1), np.flip(step, axis=0)))),
                        )

                        # Ensure that the lane is within observed area
                        from_clamp, to_clamp = self._clamp(from_, to_, min=0, max=c.MESH_SIZE - 1)
                        if not (np.array_equal(from_, from_clamp) or np.array_equal(to_, to_clamp)):
                            continue

                        # Add to the layer the appropriate color
                        tmp = np.column_stack((from_clamp, to_clamp))
                        idx = ((tmp[0], slice(*np.add((0, 1), np.sort(tmp[1]))))
                               if position in c.Position.horizontal() else
                               (slice(*np.add((0, 1), np.sort(tmp[0]))), tmp[1]))

                        if junction_id == self.trafficlight_id:
                            self.color_layer[tuple(idx)] = 1 if flow_direction else -1
                        else:
                            self.color_layer[tuple(idx)] = 3
                    lane_length = lanes[-1][0] + self.connection.lane.getLength(lanes[-1][2])
                    if max_lane_length < lane_length:
                        max_lane_length = lane_length

                    cursor += step

                neigh_junction = segment.get('junction', None)
                if neigh_junction is not None:
                    neigh_junction['segments'][c.Position.invert(position)] = \
                        {'lanes': [None] * len(segment['lanes'])}

                    junction_shape = self._get_junction_shape(junction)['shape']
                    neigh_junction_shape = self._get_junction_shape(neigh_junction)['shape']
                    lane_shape = (
                        (0, max_lane_length)
                        if position in c.Position.horizontal()
                        else (max_lane_length, 0)
                    )

                    center_offset = np.add(relative_offset, np.multiply(
                        (step[::-1]
                         if position in c.Position.horizontal()
                         else np.negative(step[::-1])),
                        np.int64(np.sum(
                            (np.floor(np.divide(junction_shape, 2)),
                             np.floor(np.divide(lane_shape, c.MESH_PARTITIONING_STEP)),
                             np.floor(np.divide(neigh_junction_shape, 2)))))))
                    if position == c.Position.BOTTOM:  # FIXME: Because the center of the junction is shifted...
                        center_offset = np.subtract(center_offset, (1, 0))

                    # TODO: We can check, whether the junction should be processed
                    queued_junctions.append((neigh_junction, center_offset))

            # The flag specifies whether we use horizontal or vertical segment to find the fill area
            flag = all(position in polygon for position in c.Position.horizontal())
            tmp = np.column_stack([val for key, val in polygon.items()
                                   if key in (c.Position.horizontal() if flag else c.Position.vertical())])

            offset = np.stack(((0, 1), (1, 0)) if flag else ((1, 0), (0, 1)))
            idx = [slice(*self._clamp(range_, min=0, max=c.MESH_SIZE)) for range_ in np.add(np.sort(tmp), offset)]
            self.color_layer[tuple(idx)] = 2

    @staticmethod
    def _get_junction_shape(junction):
        r"""Calculates shape of a given junction

        :param junction: a junction.
        :return: shape of the junction
        """

        lengths = {}
        height = width = 0

        for position, segment in junction['segments'].items():
            lengths[position] = len(segment['lanes'])
            if position in c.Position.horizontal():
                height = max(height, lengths[position])
            else:
                width = max(width, lengths[position])
        return {'lengths': lengths, 'shape': (height, width)}

    @staticmethod
    def _get_junction_polygon(junction, offset=None):
        r"""Creates a polygon that defines the boundaries of the junction. The polygon is defined in the
        following format: the key specifies the position of the starting point and the values specifies the
        coordinates of the starting point.

             0. <t>  0.  0.
             0.  0.  0. <r>
            <l>  0.  0.  0.
             0.  0. <b>  0.

        :param junction: a junction.
        :param offset: the offset from the center.
        :return: boundaries of the junction
        """

        shape = Observer._get_junction_shape(junction)
        lengths = shape['lengths']
        (height, width) = shape['shape']

        center = Observer._CENTER
        if offset is not None:
            center = np.add(center, offset)

        result = {}
        for position in lengths:
            if position in c.Position.horizontal():
                if position == c.Position.LEFT:
                    offset = (0, width % 2)
                else:
                    offset = (height % 2 - 1, 1)
                coord = (lengths[position] // 2, width // 2)
            else:
                if position == c.Position.TOP:
                    offset = (height % 2, width % 2 - 1)
                else:
                    offset = (1, 0)
                coord = (height // 2, lengths[position] // 2)

            result[position] = np.add(center, np.multiply(Observer._ROTATE[position], np.add(offset, coord)))
        return result

    @staticmethod
    def _clamp(*args, **kwargs):
        args = [np.maximum(kwargs['min'], np.minimum(kwargs['max'], arg)) for arg in args]
        return args if len(args) > 1 else args[0]

    def get_observation(self):
        r"""Produces a tensor obtained by staking matrices each of which represent current state
        of the specified trafficlight from different perspective.

        :return: none.
        """

        self.current_observation = np.zeros((c.MESH_SIZE, c.MESH_SIZE, 1))

        for junction, relative_offset in self.topology:
            polygon = self._get_junction_polygon(junction, relative_offset)

            for position, segment in junction['segments'].items():
                step, cursor = Observer._STEPS[position], polygon[position]
                for lanes in segment['lanes']:
                    if lanes is None: break

                    for offset, flow_direction, lane in lanes:
                        for vehicle in self.connection.lane.getLastStepVehicleIDs(lane):
                            # Get position of a car on the road as distance from the trafficlight
                            distance = self.connection.vehicle.getLanePosition(vehicle)
                            if flow_direction:
                                distance = self.connection.lane.getLength(lane) - distance

                            # Find idx of the cell on the grid
                            idx = np.add(cursor, np.multiply(
                                np.int64(np.round((offset + distance) / c.MESH_PARTITIONING_STEP)),
                                np.multiply((-1, 1), np.flip(step, axis=0))))

                            # Ensure that the car is within the observation area
                            if not np.array_equal(idx, self._clamp(idx, min=0, max=c.MESH_SIZE - 1)):
                                continue

                            self.current_observation[tuple(idx)][0] = 1
                    cursor += step
        return self.current_observation

    def print_current_observation(self):
        _TURQUOISE = Template('\x1b[0;30;46m$str\x1b[0m')
        _WHITE_TURQUOISE_FONT = Template('\x1b[0;36;40m$str\x1b[0m')
        _WHITE_GRAY_FONT = Template('\x1b[0;37;40m$str\x1b[0m')
        _GRAY = Template('\x1b[0;37;47m$str\x1b[0m')
        _PURPLE = Template('\x1b[0;30;45m$str\x1b[0m')

        mesh = self.color_layer.copy()
        mesh[np.where(self.current_observation == 1)] = 9

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
                    d_string += '0.'

                    if d == 1:
                        row_string += _WHITE_TURQUOISE_FONT.substitute(str=d_string)
                    elif d == -1:
                        row_string += _TURQUOISE.substitute(str=d_string)
                    elif d == 2:
                        row_string += _GRAY.substitute(str=d_string)
                    elif d == 3:
                        row_string += _WHITE_GRAY_FONT.substitute(str=d_string)
                    else:
                        row_string += d_string
            print(f'[{row_string}]')
