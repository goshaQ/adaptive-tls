import numpy as np
from pprint import pprint
from string import Template
from collections import deque


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

        self._buffered_shapes = {}

        # Specify center of the regulated intersection
        self._CENTER = ((c.MESH_SIZE - 1) // 2,)*2

        # Some auxiliary multipliers to rotate segments
        self._ROTATE = {
            c.Position.LEFT: (1, -1),
            c.Position.RIGHT: (-1, 1),
            c.Position.TOP: (-1, -1),
            c.Position.BOTTOM: (1, 1),
        }

    # ToDo: Think about changes in the format of the junction; Segments
    def _get_junction_shape(self, junction):
        junction_id = junction['id']

        if junction_id not in self._buffered_shapes:
            lengths = {}
            height = width = 0

            for position, segment in junction['segments'].items():
                lengths[position] = len(segment['lanes'])
                if position in c.Position.horizontal():
                    height = max(height, lengths[position])
                else:
                    width = max(width, lengths[position])
            self._buffered_shapes[junction_id] = {'lengths': lengths, 'shape': (height, width)}
        return self._buffered_shapes[junction_id]

    @staticmethod
    def _clamp(*args, **kwargs):
        args = [np.maximum(kwargs['min'], np.minimum(kwargs['max'], arg)) for arg in args]
        return args if len(args) > 1 else args[0]

    def _get_junction_polygon(self, junction_id, offset=None):
        shape = self._buffered_shapes[junction_id]
        lengths = shape['lengths']
        (height, width) = shape['shape']

        # Specify how points of the polygon are rotated
        rotate = {
            c.Position.LEFT: (1, -1), c.Position.RIGHT: (-1, 1),
            c.Position.TOP: (-1, -1), c.Position.BOTTOM: (1, 1),
        }

        center = ((c.MESH_SIZE - 1) // 2, (c.MESH_SIZE - 1) // 2)
        if offset is not None:
            center = np.add(center, offset)
        hoffset, voffset = np.int64((width % 2, 1))

        result = {}
        for position in lengths:
            if position in c.Position.horizontal():
                result[position] = (
                    np.add(center, np.multiply(rotate[position], (lengths[position] // 2, hoffset + width // 2))))
                hoffset = 1
            else:
                result[position] = (
                    np.add(center, np.multiply(rotate[position], (voffset + height // 2, lengths[position] // 2))))
                voffset = height % 2
        return result

    def _get_junction_polygon2(self, junction_id, offset=None):
        shape = self._buffered_shapes[junction_id]
        lengths = shape['lengths']
        (height, width) = shape['shape']

        center = self._CENTER
        if offset is not None:
            center = np.add(center, offset)

        if junction_id == '648538922':
            height, width = width, height

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

            result[position] = np.add(center, np.multiply(self._ROTATE[position], np.add(offset, coord)))
        return result

    def get_state(self, trafficlight_id, display=True):
        r"""Produces a tensor obtained by staking matrices each of which represent current state
        of the specified trafficlight from different perspective.

        :param trafficlight_id: trafficlight ID
        :return:
        """

        trafficlight_skeleton = self.skeletons[trafficlight_id]
        if trafficlight_id not in self._buffered_shapes:  # ToDo: Revise?
            self._buffered_shapes[trafficlight_id] = self._get_junction_shape(trafficlight_skeleton)

        mesh = np.zeros((c.MESH_SIZE, c.MESH_SIZE))
        color = np.zeros((c.MESH_SIZE, c.MESH_SIZE))
        pprint(trafficlight_skeleton)

        rotate = (-1, 1)
        b_offset = (0, 1)

        steps = {  # Specify in which direction the cursor is moving
            c.Position.LEFT: (-1, 0), c.Position.RIGHT: (1, 0),
            c.Position.TOP: (0, 1), c.Position.BOTTOM: (0, -1),
        }

        # ToDO: Recursion vs. Queue for processing nested junctions?
        queued_junctions = deque()
        queued_junctions.append((trafficlight_skeleton, ((0, 0), None)))

        while queued_junctions:
            junction, complementary = queued_junctions.pop()
            junction_id = junction['id']

            center_offset, relative = complementary
            polygon = self._get_junction_polygon2(junction_id, center_offset)

            for position, segment in junction['segments'].items():
                if position is c.Position.invert(relative): continue

                max_lane_length = 0
                step, cursor = steps[position], polygon[position]
                for lanes in segment['lanes']:
                    if lanes is None: continue

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
                            if not np.array_equal(idx, self._clamp(idx, min=0, max=c.MESH_SIZE - 1)):
                                continue
                            mesh[tuple(idx)] = 1

                        # ToDo: There is no need to create color layer every time
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
                        idx = ((tmp[0], slice(*np.add(b_offset, np.sort(tmp[1]))))
                               if position in c.Position.horizontal() else
                               (slice(*np.add(b_offset, np.sort(tmp[0]))), tmp[1]))
                        color[idx] = 1 if flow_direction else -1
                    color[(15, 15)] = 2
                    cursor += step

                    # There might be a problem if lanes are merged
                    lane_length = lanes[-1][0] + self.simulation.lane.getLength(lanes[-1][2])
                    if max_lane_length < lane_length:
                        max_lane_length = lane_length

                print(position, max_lane_length)
                neigh_junction = segment.get('junction', None)
                if neigh_junction is not None:
                    neigh_junction['segments'][c.Position.invert(position)] = {'lanes': [None]*len(segment['lanes'])}

                    junction_shape = self._get_junction_shape(junction)['shape']
                    neigh_junction_shape = self._get_junction_shape(neigh_junction)['shape']

                    print('b', center_offset)
                    center_offset1 = np.add(center_offset, np.multiply(
                        (np.negative(step[::-1]) if position in c.Position.vertical() else step[::-1]),  # ToDo: Revise
                        np.int64(np.sum(
                            (np.floor(np.divide(junction_shape, 2)),
                             np.floor(np.divide((max_lane_length, 0) if position in c.Position.vertical() else (0, max_lane_length), c.MESH_PARTITIONING_STEP)),
                             np.floor(np.divide(neigh_junction_shape, 2)))))))
                    print('position', position, 'step', step, 'offset', center_offset1)
                    print(self._get_junction_polygon2(neigh_junction['id']))

                    # ToDo: If offset is outside of the observed area, don't add to the queue
                    queued_junctions.append((neigh_junction, (center_offset1, position)))
                    # Add to the queue...

        ############################################################################
        #
        # for position in c.Position:
        #     step, cursor = steps[position], polygon[position]
        #     direction, elements = trafficlight_skeleton[position].values()
        #
        #     for lanes in elements['lanes']:
        #         for offset, flow_direction, lane in lanes:
        #             for vehicle in self.simulation.lane.getLastStepVehicleIDs(lane):
        #                 # Get position of a car on the road as distance from the trafficlight
        #                 distance = self.simulation.vehicle.getLanePosition(vehicle)
        #                 if flow_direction:
        #                     distance = self.simulation.lane.getLength(lane) - distance
        #
        #                 # Find idx of the cell on the grid
        #                 idx = np.add(cursor, np.multiply(
        #                     np.int64((offset + distance) // c.MESH_PARTITIONING_STEP),
        #                     np.multiply(rotate, np.flip(step, axis=0))))
        #
        #                 # Ensure that the car is within the observation area
        #                 if not np.array_equal(idx, self._clamp(idx, min=0, max=c.MESH_SIZE - 1)):
        #                     continue
        #                 mesh[tuple(idx)] = 1
        #
        #             # Save lanes to the color layer
        #             from_, to_ = self._clamp(
        #                 np.add(cursor, np.multiply(
        #                     np.int64(offset // c.MESH_PARTITIONING_STEP),
        #                     np.multiply(rotate, np.flip(step, axis=0)))),
        #                 np.add(cursor, np.multiply(
        #                     np.int64((offset + self.simulation.lane.getLength(lane)) // c.MESH_PARTITIONING_STEP),
        #                     np.multiply(rotate, np.flip(step, axis=0)))),
        #                 min=0, max=c.MESH_SIZE - 1)
        #
        #             tmp = np.column_stack((from_, to_))
        #             idx = (tmp[0], slice(*np.add(b_offset, np.sort(tmp[1]))))\
        #                 if position in c.Position.horizontal() else (slice(*np.add(b_offset, np.sort(tmp[0]))), tmp[1])
        #
        #             color[idx] = 1 if flow_direction else -1
        #         cursor += step
        #
        #     # ToDo: Consider additionally the shape of parent junction
        #     # ToDo: Instead store the relative offset
        #     # ToDo: Store in the queue Tuple[Junction, Tuple[Offset, Relative]]
        #     junction = elements.get('junction', None)
        #     if junction is not None:
        #         # offset = np.int64(np.ceil(np.divide(junction['offset'], c.MESH_PARTITIONING_STEP)))
        #         # offset = np.add(np.multiply(step[::-1], cursor),
        #         #                 np.multiply(np.negative(step[::-1]), offset))
        #         #
        #         # tmp = np.column_stack(list(polygon.values()))
        #         # tmp = np.subtract((np.amax(tmp, axis=1), np.amin(tmp, axis=1)))
        #
        #         _, shape = self._get_junction_shape(trafficlight_skeleton)
        #         offset = np.floor_divide(shape, 2) + 1
        #         offset += np.int64(np.ceil(np.divide(junction['offset'], c.MESH_PARTITIONING_STEP)))
        #         _, shape = self._get_junction_shape(junction)
        #         offset += np.floor_divide(shape, 2)
        #
        #         print('KKK', offset)
        #         # offset = np.multiply(offset, np.add(np.negative(step[::-1]), (height // 2, width // 2)))
        #         print(f'cursor: {cursor}')
        #         print(f'{position} offset: {offset}')
        #         polygon1 = self._get_junction_polygon(junction, offset)
        #         print(f'polygon1: {polygon1}')
        #
        #         tmp = np.column_stack(list(polygon.values()))
        #         print(tmp)
        #         tmp = np.column_stack((np.amax(tmp, axis=1), np.amin(tmp, axis=1)))
        #         print(tmp)
        #
        #     # if offset is not None:
        #     #     offset = np.add(offset, (height // 2, 0))
        #     #     center = np.subtract(center, offset)
        #     #     print('dd', center)
        #
        #         for point in polygon1.values():
        #             print(f'point1: {point}')
        #             # np.multiply(steps[c.Position.LEFT], np.add(offset, np.multiply(step[::-1], cursor)))
        #             print(np.add(np.multiply(np.negative(step[::-1]), cursor), offset))
        #
        #             if not np.array_equal(point, self._clamp(point, min=0, max=c.MESH_SIZE - 1)):
        #                 continue
        #
        #             # point = np.add(point, [-13, 0])
        #             color[tuple(point)] = 2
        #
        #             print(f'point2: {point}')
        #
        #         print(idx)
        #         print(cursor)
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
                    d_string += '0.'

                    if d == 1:
                        row_string += _TURQUOISE.substitute(str=d_string)
                    elif d == -1:
                        row_string += _WHITE.substitute(str=d_string)
                    elif d == 2:
                        row_string += _GRAY.substitute(str=d_string)
                    else:
                        row_string += d_string
            print(f'[{row_string}]')
