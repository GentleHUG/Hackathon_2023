import cmath as cm
import math as m
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jvp
from math import *


'''Постоянные в СИ'''
R = 2e-3
a = 5e-4
lmbd = 720e-9
k = 2 * cm.pi / lmbd
E0 = 1
z = 1
N = 151
focus_size = 2


teta_max = a
tetas = np.linspace(-teta_max, teta_max, N, endpoint=True)


'''Геометрия источников'''
# Квадрат 2*2
coords_4 = [(1 / 2, 1 / 2),   (-1 / 2, 1 / 2),
            (-1 / 2, -1 / 2), (1 / 2, -1 / 2)]

# Гексагон
coords_7 = [(0, 0), (1, 0), (1 / 2, sqrt(3) / 2), (-1 / 2, sqrt(3) / 2), (-1, 0), (-1 / 2, -sqrt(3) / 2),
            (1 / 2, -sqrt(3) / 2)]

# Квадрат 3*3
coords_9 = [(-1, 1),  (0, 1),  (1, 1),
            (-1, 0),  (0, 0),  (1, 0),
            (-1, -1), (0, -1), (1, -1)]

# Додекагон 12 отрезков на стороне)))
coords_13 = [(0, 0),
             (1, 0),  (sqrt(3)/2, 1/2), (1/2, sqrt(3)/2),
             (0, 1), (-1/2, sqrt(3)/2), (-sqrt(3)/2, 1/2),
             (-1, 0), (-sqrt(3) / 2, -1 / 2), (-1 / 2, -sqrt(3) / 2),
             (0, -1), (1 / 2, -sqrt(3) / 2), (sqrt(3) / 2, -1 / 2)]



selectedcoords = coords_13





def f_constants(teta_x: float, teta_y: float, E=E0):
    if teta_x == 0 and teta_y == 0:
        return pi * a ** 2 * E / (lmbd * z)
    sq = m.sqrt(teta_x ** 2 + teta_y ** 2)
    j = jvp(1, k * a * sq, 0)
    return a * E * j / (z * sq)


def certain_e(teta_x: float, teta_y: float, phase: float):
    return cm.exp(complex('j') * (-(k * R) * (teta_x + teta_y) + phase))


def i_n_sources(coord_arr, phase_arr, teta_x: float, teta_y: float):
    exps = np.array(
        [certain_e(coords[0] * teta_x, coords[1] * teta_y, phase) for coords, phase in zip(coord_arr, phase_arr)])
    res_exp = np.sum(exps)
    return f_constants(teta_x, teta_y) ** 2 * (res_exp.real ** 2 + res_exp.imag ** 2) / f_constants(0, 0) ** 2


def num_py_arr_n(coords, tetas, phase_arr=None):
    if phase_arr is None:
        phase_arr = [0, 0, 0, 0]
    return np.array([[i_n_sources(coords, phase_arr, x, y) for y in tetas] for x in tetas])


def show_graf_n(tetas, coords, fi_arr, title=''):
    # print(fi_arr / 2 / pi)
    i_2d_arr = num_py_arr_n(coords, tetas, fi_arr)
    fig, ax = plt.subplots()
    cmap = plt.colormaps["plasma"]
    cmap = cmap.with_extremes(bad=cmap(0))
    ax.set_title(title)
    cf = ax.pcolormesh(tetas / teta_max, tetas / teta_max, i_2d_arr, cmap=cmap)
    # for i in range(len(fi_arr)):
    #     plt.arrow(0, 0, cos(fi_arr[i]) / 2, sin(fi_arr[i]) / 2, width=0.002, color="white")
    fig.colorbar(cf, ax=ax)
    plt.savefig('imag' + title[16:])


def find_better_phase_n(coords, real_phase_arr, plus_arr, tetas, size_of_point=focus_size, steps=100):
    # Фукция ищёт поправку к plus_arr, при постоянном phase_arr
    plus_phase = np.zeros(len(real_phase_arr))
    deltas = np.linspace(0, 2 * pi, steps)
    new_tetas = tetas[len(tetas) // 2 - size_of_point: len(tetas) // 2 + size_of_point]

    def i_in_focus(delta, i=0):
        # Средняя интенсивность в квадрате 2*size_of_points+1
        pluser = real_phase_arr + plus_arr + plus_phase
        pluser[i] += delta
        return np.array([([i_n_sources(coords, pluser, x, y) for y in new_tetas]) for x in new_tetas]).mean()

    for i in range(len(real_phase_arr)):
        better_delta = max(deltas, key=lambda x: i_in_focus(x, i))
        # print(better_delta / pi)
        plus_phase[i] = better_delta

    return plus_phase


fi_0 = np.random.random(len(selectedcoords)) * 2 * pi


def do_and_print_one_correction(coords, fi_0, tetas):
    plus_arr = np.zeros(len(fi_0))
    print(fi_0 + plus_arr, plus_arr, sep='\n')
    show_graf_n(tetas, coords, fi_0 + plus_arr, f'Airy pattern of {len(coords)}-channel laser')

    plus_arr += find_better_phase_n(coords, fi_0, plus_arr, tetas)
    print(fi_0 + plus_arr, plus_arr, sep='\n')
    show_graf_n(tetas, coords, fi_0 + plus_arr, f'Airy pattern of {len(coords)}-channel laser corrected x1')

    plus_arr += find_better_phase_n(coords, fi_0, plus_arr, tetas)
    print(fi_0 + plus_arr, plus_arr, sep='\n')
    show_graf_n(tetas, coords, fi_0 + plus_arr, f'Airy pattern of {len(coords)}-channel laser corrected x2')

do_and_print_one_correction(selectedcoords, fi_0, tetas)
