from environment import constants as c

from queue import deque
from collections import defaultdict


def get_positioned_junction(net, junction):
    r"""
    Note: This method is pure heuristic, so there is no guarantee that it will return
    always properly positioned junction.
    """

    ext_bottom_left = ext_bottom_right = ext_top_left = ext_top_right = None
    junction_shape = junction.getShape()
    junction_center = junction.getCoord()
    for point in junction_shape:
        if point[1] < junction_center[1]:  # Below the center
            if ext_bottom_left is None and ext_bottom_right is None:
                ext_bottom_left = ext_bottom_right = point
            elif ext_bottom_left[0] > point[0]:  # New extreme bottom left
                ext_bottom_left = point
            elif ext_bottom_right[0] < point[0]:  # New extreme bottom right
                ext_bottom_right = point
        else:  # Above the center
            if ext_top_left is None and ext_top_right is None:
                ext_top_left = ext_top_right = point
            elif ext_top_left[0] > point[0]:  # New extreme top left
                ext_top_left = point
            elif ext_top_right[0] < point[0]:  # New extreme top right
                ext_top_right = point

    slope = []
    for point in [ext_bottom_left, ext_bottom_right, ext_top_left, ext_top_right]:
        slope.append((point[1] - junction_center[1]) / (point[0] - junction_center[0]))

    tmp = defaultdict(list)
    for edge in junction.getOutgoing() + junction.getIncoming():
        # Select the second point from the border of the junction, mb improve later
        idx = 1 if edge in junction.getOutgoing() else -2

        first_point = edge.getShape()[idx]
        diff_x, diff_y = first_point[0] - junction_center[0], first_point[1] - junction_center[1]

        if slope[0] * diff_x < diff_y < slope[2] * diff_x:  # Find edges on the left side
            tmp[c.Position.LEFT].insert(1, edge.getID())
        elif slope[1] * diff_x > diff_y < slope[0] * diff_x:  # Find edges on the bottom side
            tmp[c.Position.BOTTOM].insert(1, edge.getID())
        elif slope[3] * diff_x > diff_y > slope[1] * diff_x:  # Find edges on the right side
            tmp[c.Position.RIGHT].insert(0, edge.getID())
        elif slope[2] * diff_x < diff_y > slope[3] * diff_x:  # Find edges on the top side
            tmp[c.Position.TOP].insert(0, edge.getID())

    # Add None to the list if degenerative case encountered to indicate
    # whether outgoing or incoming edge is present
    for direction, edge_id_list in tmp.items():
        if len(edge_id_list) == 1:  # The list contain either outgoing or incoming edge
            edge = net.getEdge(edge_id_list[0])

            idx = 0 if edge in junction.getOutgoing() else 1
            edge_id_list.insert(idx, None)

            tmp[direction] = edge_id_list
    return tmp


def get_junction_type(node):
    """
    Returns the type of the junction. To properly locate placement of the neighboring edge we need to determine
    the type of the corresponding junction. There are several types of junctions: (1) connection of two edges
    that form the roadway, (2) exit on the dedicated roadway to turn right, (3) Regulated intersection of two roadways,
    (4) Unregulated intersection  of two roadways.

    :param node:
    :return:
    """

    def is_connection(_node=node):
        if (len(_node.getIncoming()) == len(_node.getOutgoing()) == 1
                and _node.getIncoming()[0].getToNode() == _node.getOutgoing()[0].getFromNode()
                and _node.getType() == 'priority'):
            return True

    def is_channelized_right_turn(_node=node):
        def get_incoming_or_outgoing(x, condition):
            return x.getIncoming() if condition else x.getOutgoing()

        def get_to_or_from_node(x, condition):
            return x.getFromNode() if condition else x.getToNode()

        is_ending = len(node.getIncoming()) > len(node.getOutgoing())
        try:
            main = None
            for e in get_incoming_or_outgoing(node, is_ending):
                to_node = get_to_or_from_node(e, is_ending)
                if is_regulated_intersection(to_node):  # Replace?
                    main = e
                    break
            if main is None: return False

            complementary = None

            preceding = next(iter(get_incoming_or_outgoing(main, not is_ending)))
            preceding_to = get_incoming_or_outgoing(preceding, is_ending)
            for e in preceding_to:
                if e != main:
                    complementary = e
                    break

            complementary_to = get_to_or_from_node(complementary, is_ending)
            complementary_list = [complementary]
            while is_connection(complementary_to):
                complementary_to = next(iter(get_incoming_or_outgoing(complementary_to, is_ending)))
                # FIXME: Instead an edge need to be appended to the list
                complementary_list.append(complementary_to)

                complementary_to = get_to_or_from_node(complementary_to, is_ending)

            for e in get_incoming_or_outgoing(complementary_to, not is_ending):
                from_node = get_to_or_from_node(e, not is_ending)  # Find the node we expect to be traffic light
                if from_node == get_to_or_from_node(main, is_ending):  # Check whether our expectation is met
                    args['is_ending'] = is_ending
                    args['complementary'] = complementary_list
                    return True
        except AttributeError:
            pass  # Assumption was wrong; pretend we didn't do it

    def is_unknown(_node=node):
        if _node.getType() not in ['traffic_light', 'priority', 'dead_end']:
            return True

    def is_dead_end(_node=node):
        if _node.getType() == 'dead_end':
            return True

    def is_regulated_intersection(_node=node):
        # ToDo: Revise?
        if _node.getType() == 'traffic_light':
            return True

    def is_unregulated_intersection(_node=node):
        # ToDo: Revise?
        if _node.getType() == 'priority':
            if any(_c.getDirection() in ['r', 'l'] for _c in _node.getConnections()):
                return True

    args = {}
    if is_unknown():
        return c.Junction.UNKNOWN, args
    elif is_dead_end():
        return c.Junction.DEAD_END, args
    elif is_connection():
        return c.Junction.CONNECTION, args
    elif is_channelized_right_turn():
        return c.Junction.CHANNELIZED_RIGHT_TURN, args
    elif is_regulated_intersection():
        return c.Junction.REGULATED_INTERSECTION, args
    elif is_unregulated_intersection():
        return c.Junction.UNREGULATED_INTERSECTION, args
    else:  # The type of junction is not recognized, just for now
        return c.Junction.UNKNOWN, args


def extract_tl_skeleton(net, trafficlight):
    def get_adjacent_node(_edge, _flag):
        return _edge.getFromNode() if _flag else _edge.getToNode()

    def get_adjacent_edges(_node, _flag):
        return _node.getIncoming() if _flag else _node.getOutgoing()

    tl_skeleton = {
        'id': trafficlight.getID(),
    }

    queued_skeletons = deque()
    queued_skeletons.append(tl_skeleton)

    while queued_skeletons:
        skeleton = queued_skeletons.pop()
        node = net.getNode(skeleton['id'])

        voffset, hoffset = skeleton.get('offset', (0, 0))
        positioned_edges = get_positioned_junction(net, node)

        upstream_changed = False

        if 'segments' not in skeleton:
            skeleton['segments'] = {}
        else:
            # TODO: Is there a better approach?
            position, seq = next(iter(skeleton['segments'].items()))
            while not any(edge in positioned_edges[position] for edge in seq):
                keys = list(positioned_edges.keys())
                values = list(positioned_edges.values())

                for key, val in zip(keys[1:]+keys[:1], values):
                    positioned_edges[key] = val
                upstream_changed = not upstream_changed
            skeleton['segments'][position] = None

        for position, edges in positioned_edges.items():
            if not edges or position in skeleton['segments']: continue

            segment = dict()

            upstream = position in c.Position.upper_corner() and not upstream_changed
            horizontal = position in c.Position.horizontal()

            for edge_id in edges:
                if edge_id is not None:
                    edge = net.getEdge(edge_id)

                    length = (hoffset if horizontal else voffset) + edge.getLength()
                    seq = [edge_id]

                    while length < c.MESH_PARTITIONING_STEP * c.MESH_SIZE / 2:
                        adj_node = get_adjacent_node(edge, upstream)
                        node_type, _ = get_junction_type(adj_node)

                        # TODO: Handle c.Junction.CHANNELIZED_RIGHT_TURN
                        if node_type is c.Junction.CONNECTION:
                            edge = get_adjacent_edges(adj_node, upstream)[0]
                            length += edge.getLength()
                            seq.append(edge.getID())
                        elif node_type is c.Junction.UNREGULATED_INTERSECTION\
                                and 'junction' not in segment:
                            junction = {
                                'id': adj_node.getID(),
                                'offset': (voffset, length) if horizontal else (length, hoffset),

                                # Indicate the segment that should be avoided
                                'segments': {
                                    c.Position.invert(position): seq,
                                },
                            }

                            segment['junction'] = junction
                            queued_skeletons.append(junction)
                            break
                        else:  # Someone else's responsibility
                            break

                    segment['upstream' if upstream else 'downstream'] = seq
                upstream = not upstream

            segment = _process_elements(net, segment)
            skeleton['segments'][position] = segment
    return tl_skeleton


def _process_elements(net, elements, squeeze=True):
    def _get_connections(_edge, _neigh_edge, _flag):
        return (_edge.getIncoming() if _flag else _edge.getOutgoing()).get(_neigh_edge, [])

    def _extract_lanes_from_edges(_edges, _flow_direction):
        _edges = [net.getEdge(edge_id) for edge_id in _edges]

        shift = 0
        _offset = 0
        prev_edge = None

        extracted_lanes = []
        for edge in _edges:
            if prev_edge is not None:
                connections = _get_connections(prev_edge, edge, _flow_direction)
                connections = [(_c.getFromLane().getIndex(), _c.getToLane().getIndex()) for _c in connections]

                diff = 0
                for connection in connections:
                    from_lane, to_lane = map(bool, connection)
                    diff += to_lane - from_lane if _flow_direction else from_lane - to_lane

                shift += diff
                _offset += prev_edge.getLength()
            prev_edge = edge

            for lane in edge.getLanes():
                _idx = shift + lane.getIndex()

                if not 0 <= _idx < len(extracted_lanes):
                    _idx = max(0, _idx)
                    extracted_lanes.insert(_idx, [])
                extracted_lanes[_idx].append([_offset, _flow_direction, lane.getID()])
        return extracted_lanes

    upstream_lanes = _extract_lanes_from_edges(elements.pop('upstream', []), True)
    downstream_lanes = _extract_lanes_from_edges(elements.pop('downstream', []), False)

    if squeeze:
        l1_count = 0  # Maximum number of lanes that we can merge
        l1_offset = 0  # Offset that these lanes have
        for lanes in reversed(downstream_lanes):
            offset, _, _ = lanes[0]

            if offset != 0:
                l1_count += 1
                l1_offset = offset
            else:
                break

        l2_count = 0  # Number of lanes that we can actually merge
        for lanes in reversed(upstream_lanes):
            offset = lanes[-1][0] + net.getLane(lanes[-1][2]).getLength()

            if offset <= l1_offset:
                l2_count += 1
            else:
                break

        merged = [*upstream_lanes]

        idx_offset = len(merged) - (l1_count if l2_count >= l1_count else l2_count)  # Revise
        for idx, lanes in enumerate(reversed(downstream_lanes)):
            idx += idx_offset

            if not 0 <= idx < len(merged):
                merged.insert(idx, [])
            merged[idx].extend(lanes)
    else:
        merged = upstream_lanes + downstream_lanes

    elements['lanes'] = merged
    return elements
