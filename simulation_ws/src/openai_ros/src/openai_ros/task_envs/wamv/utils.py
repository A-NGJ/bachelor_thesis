from enum import Enum


class Actions(Enum):
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    LEFT45 = 4
    RIGHT45 = 5
    LEFT30 = 6
    RIGHT30 = 7


class ColorBoundaries:
    GREEN = ([30, 50, 0], [50, 80, 110])
    RED = ([0, 0, 0], [110, 50, 255])
