from enum import Enum

# Number N of cells in the grid NxN
MESH_SIZE = 32

# Length L of the cell in the grid; m
MESH_PARTITIONING_STEP = 6


class Direction:
    BIDIRECTIONAL = 0
    INCOMING = 1
    OUTGOING = 2


class Junction(Enum):
    CONNECTION = 0b001
    CHANNELIZED_RIGHT_TURN = 0b010
    REGULATED_INTERSECTION = 0b011
    UNREGULATED_INTERSECTION = 0b100
    UNKNOWN = 0b000


class Position(Enum):
    LEFT = 'left'
    TOP = 'top'
    RIGHT = 'right'
    BOTTOM = 'bottom'

    @classmethod
    def horizontal(cls):
        return cls.LEFT, cls.RIGHT

    @classmethod
    def vertical(cls):
        return cls.TOP, cls.BOTTOM

    @classmethod
    def upper_corner(cls):
        return cls.TOP, cls.RIGHT

    @classmethod
    def lower_corner(cls):
        return cls.LEFT, cls.BOTTOM
