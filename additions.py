from typing import Optional

from numpy import sqrt, arange


class Dot:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __sub__(self, other: "Dot"):
        return sqrt((self.x - other.x) ** 2 + (self.x - other.y) ** 2)

    def __eq__(self, other: list):
        return other[0] == self.x and other[1] == self.y


def make_grid(start, stop, step):
    """

    :param start:
    :param stop: not included
    :param step:
    :return:
    """
    x = []
    y = []
    for i in arange(start, stop, step):
        for j in arange(start, stop, step):
            x.append(i)
            y.append(j)
    return x, y
