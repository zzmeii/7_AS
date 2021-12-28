from typing import List

import numpy as np
from numpy import pi


def normal_desp_3d(x, y, disp=(1, 2), expect=(0, 0), cor=0):
    return 1 / (2 * pi * disp[0] * disp[1] * np.sqrt(1 - cor ** 2)) * np.exp(-1 / (2 * (1 - cor ** 2)) * (
            (x - expect[0]) ** 2 / disp[0] ** 2 - cor * (2 * (x - expect[0]) * (y - expect[1])) / disp[0] * disp[
        1] + (y - expect[1]) ** 2) / disp[1] ** 2)


def normal_desp(x):
    """
    Нормальное распределение
    Ядро для Нарадая-Ватсона
    :param x:
    :return:
    """
    return (2 * np.pi) ** -0.5 * np.exp(-x ** 2 / 2)


def naraday_whatson(nw_c, nw_u, nw_u_arr, nw_x_arr):
    """
    Нарадай-Ватсон
    Отдельно считается числитель (numerator) и знаменатель (denominator)
    :param nw_c: Критерий гладкости, != 0
    :param nw_u: Точка для которой рассчитывается апроксимация
    :param nw_u_arr: Весь массив по x
    :param nw_x_arr: Весь массив по y
    :return: Значение по y
    """
    numerator = sum([nw_x_arr[i] * normal_desp((nw_u - nw_u_arr[i]) / nw_c) for i in range(len(nw_x_arr))])
    denominator = sum([normal_desp((nw_u - nw_u_arr[i]) / nw_c) for i in range(len(nw_x_arr))])
    return numerator / denominator


def nep_aprox(data: List, c, work_range: range):
    """
    Используя функцию выше, создаёт аппроксимированный массив y
    :param data:
    :param c:
    :param work_range:
    :return:
    """
    res = [naraday_whatson(c, data[0][i], data[0], data[1]) for i in work_range]
    return res


def err_ex(data, real_func):
    """
    Среднеквадратичное отклонение
    :param data:
    :param real_func:
    :return:
    """
    return np.mean([(data[i] - real_func[i]) ** 2 for i in range(len(real_func))])
