"""Весь код писался в GoogleColaboratory, поэтому здесь только код, без проекта и тд"""


import cmath as cm
import math as m
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jvp
from math import*

'''Constants in SI'''
R = 1e-3
a = 5e-4
lmbd = 720e-9
k = 2 * cm.pi / lmbd
E0 = 1
z = 1
N = 201

teta_max = 1 * R
tetas = np.linspace(-teta_max, teta_max, N, endpoint=True)



def f_constans(teta_x: float, teta_y: float, E=E0):
  if teta_x == 0 and teta_y == 0:
    return pi * a ** 2 * E / (lmbd * z)
  sq = m.sqrt(teta_x ** 2 + teta_y ** 2)
  j = jvp(1, k * a * sq, 0)
  return a * E * j  / (z * sq)

def certain_e(teta_x: float, teta_y: float, phase: float):
  return cm.exp(complex('j') * ((k * R / 2) * (teta_x + teta_y + phase)))

def i_4_sourses(phase_arr, teta_x: float, teta_y: float):
  exps = np.array([certain_e(-teta_x, -teta_y, phase_arr[0]), certain_e(teta_x, -teta_y, phase_arr[1]), certain_e(teta_x, teta_y, phase_arr[2]), certain_e(-teta_x, teta_y, phase_arr[3])])
  res_exp = sum(exps)
  return f_constans(teta_x, teta_y) ** 2 * (res_exp.real ** 2 + res_exp.imag ** 2)

def num_py_arr(tetas, phase_arr=None):
  if phase_arr is None:
    phase_arr = [0,0,0,0]
  return np.array([[i_4_sourses(phase_arr, x, y) for y in tetas] for x in tetas])

def show_graf(tetas, fi_arr, title='Airy pattern'):
  # print(fi_arr / 2 / pi)
  I_2d_arr = num_py_arr(tetas, fi_arr)
  fig, ax = plt.subplots()
  ax.set_title(title)
  cf = ax.pcolormesh(tetas/R, tetas/R, I_2d_arr)
  # for i in range(4):
  #       plt.arrow(0, 0, cos(fi_arr[i]) / 2, sin(fi_arr[i]) / 2, width = 0.02, color="black")
  fig.colorbar(cf, ax=ax)
  plt.show()




def find_better_phase(real_phase_arr, plus_arr, tetas, size_of_point=5, steps=100):
  # Фукция ищёт поправку к plus_arr, при постоянном phase_arr
  plus_phase = np.zeros(len(real_phase_arr))
  deltas = np.linspace(0, 2 * pi, steps)
  new_tetas = tetas[len(tetas) // 2 - size_of_point : len(tetas) // 2 + size_of_point]

  def i_in_focus(delta, i=0):
    #Средняя интенсивность в квадрате 2*size_of_points+1
    pluser = real_phase_arr + plus_arr + plus_phase
    pluser[i] += delta
    return np.array([([i_4_sourses(pluser, x, y) for y in new_tetas]) for x in new_tetas]).mean()

  for i in range(len(real_phase_arr)):
    better_delta = max(deltas, key=lambda x: i_in_focus(x, i))
    # print(better_delta / pi)
    plus_phase[i] = better_delta 

  return(plus_phase)

fi_0 = np.random.random(4)
plused_phase = np.zeros(len(fi_0))
show_graf(tetas, fi_0 + plused_phase)

for num in range(4):
  print(plused_phase)
  plused_phase += find_better_phase(fi_0, plused_phase, tetas)
  show_graf(tetas, fi_0 + plused_phase, f'Airy pattern corrected {num + 1}th')