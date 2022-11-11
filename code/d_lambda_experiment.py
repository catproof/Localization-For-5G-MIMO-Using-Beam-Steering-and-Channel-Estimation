import os
import string
from threading import Thread
import numpy as np
import numpy.random
import math
import random
from matplotlib import pyplot as plt

from util import norm_2, print_debug, random_MS_pose_within_circle
from main_for_DCS_SOMP import fiveG_positioning_experiement

class ExpPair():

  def __init__(self, d, lambda_n):
    self.d: np.double = d
    self.lambda_n: np.double = lambda_n
    self.result: ResultPair = None;

  def to_nd_array(self):
    return np.array([
      self.d, 
      self.lambda_n, 
      self.result.posRx_error_mean if self.result is not None else None,
      self.result.posRx_error_confidence_interval if self.result is not None else None,
      self.result.alpha_error_mean if self.result is not None else None,
      self.result.alpha_error_confidence_interval if self.result is not None else None,
      ], dtype=np.double)
  
  def __str__(self): return self.to_nd_array().__str__()

class ResultPair():

  def __init__(self, posRx_error_mean, posRx_error_confidence_interval, alpha_error_mean, alpha_error_confidence_interval):
    self.posRx_error_mean = posRx_error_mean
    self.posRx_error_confidence_interval = posRx_error_confidence_interval
    self.alpha_error_mean = alpha_error_mean
    self.alpha_error_confidence_interval = alpha_error_confidence_interval

c = 299.792458 # light speed in m/us

def init_experiemnts(d_relative_range: np.ndarray, fiveG_freqencies: np.ndarray):
  # for each d and lambda_n in arrays, forms a experiement input instances, and return the list of all instances
  d_range_size = np.shape(d_relative_range)[0]
  lambda_n_size = np.shape(fiveG_freqencies)[0]
  exp_list = np.zeros([d_range_size * lambda_n_size], dtype=ExpPair)
  for n_i in range(lambda_n_size):
    current_lambda = c / (fiveG_freqencies[n_i] * 1000) # convert frequency to MHz and get wavelength in meter
    d_range = d_relative_range * current_lambda
    for d_i in range(d_range_size):
      exp_list[n_i * d_range_size + d_i] = ExpPair(d_range[d_i], current_lambda)
  return exp_list

d_relative_range = np.array([0.25, 0.333, 0.5, 0.75, 1.0, 1.5, 2.0]) # which will become d in meter during init_experiments 
fiveG_freqencies = np.array([24, 32, 37, 41, 46, 51, 66, 81]) # in GHz
iteration = 5 # run 5 times for each experiements

experiments: np.ndarray = init_experiemnts(d_relative_range, fiveG_freqencies)


def get_confidence_interval_and_mean(y):
  confidence_interval = 1.96 * np.std(y, axis=0)/math.sqrt(y.shape[0])
  mean = np.mean(y, axis=0)
  return confidence_interval, mean

for exp in experiments:
  acc_posRx_errors = np.zeros([iteration])
  acc_alpha_errors = np.zeros([iteration])
  threads = list()
  def run_iteration(i, posRx, d, lambda_n, iteration_name):
    error_of_posRx_estimation, error_of_alpha_estimation = fiveG_positioning_experiement(posRx, d, lambda_n, iteration_name)
    # error_of_posRx_estimation, error_of_alpha_estimation = random.random(), random.random() # fiveG_positioning_experiement(d, lambda_n, iteration_name)
    acc_alpha_errors[i] = error_of_alpha_estimation
    acc_posRx_errors[i] = error_of_posRx_estimation

  for i in range(0, iteration, 1): # ayncly runs all iteration
    x, y, alpha = random_MS_pose_within_circle(1, 5)
    posRx = np.array([x, y])
    iteration_name = "with d={}, lambda_n={}, posRx = {}, iteration {} ".format(exp.d, exp.lambda_n, posRx, i + 1)
    thread = Thread(target=run_iteration, args=(i, posRx, exp.d, exp.lambda_n, iteration_name))
    threads.append(thread)
    thread.start()
  
  for t in threads: # wait for all threads to be finished
    t.join()

  posRx_error_confidence_interval, posRx_error_mean = get_confidence_interval_and_mean(acc_posRx_errors)
  alpha_error_confidence_interval, alpha_error_mean = get_confidence_interval_and_mean(acc_alpha_errors)

  exp.result = ResultPair(
    posRx_error_mean, posRx_error_confidence_interval,
    alpha_error_mean, alpha_error_confidence_interval
  )

print("we got the following results to plot \n{}".format("\n".join([str(xi) for xi in experiments])) )

print("start plotting, for each lambda_n, plot d vs (poxRs error, alpha_error) ")
experiments_2d = np.reshape(experiments, [np.shape(fiveG_freqencies)[0], int(np.shape(experiments)[0]/ np.shape(fiveG_freqencies)[0])])
for d in range(np.shape(experiments_2d)[0]):
  experiments_for_this_lambda = experiments_2d[d]
  this_lambda_n_in_mm = experiments_for_this_lambda[0].lambda_n * 1000

  # first, plot the poxRx erro vs d plot
  fig, ax = plt.subplots()
  posRx_error_means = np.array([xi.result.posRx_error_mean for xi in experiments_for_this_lambda])
  posRx_confidence_intervals = np.array([xi.result.posRx_error_confidence_interval for xi in experiments_for_this_lambda])
  x_axis = np.array([xi.d * 1000 for xi in experiments_for_this_lambda])
  ax.plot(x_axis, posRx_error_means)
  ax.fill_between(x_axis, posRx_error_means - posRx_confidence_intervals, posRx_error_means + posRx_confidence_intervals, color='b', alpha=.1)
  plt.title("Localization Error of Estimated posRx vs $\mathit{d}$ \nwhen $\lambda_n$ = " + "{:.2f} mm".format(this_lambda_n_in_mm))
  plt.xlabel("$\mathit{d}$ (in mm)")
  plt.ylabel("Localization Error (in m)")
  plt.savefig(os.path.join(os.getcwd(), "posRx error vs d when λ = {:.2f}mm.jpg".format(this_lambda_n_in_mm)),pil_kwargs={'quality':95}, dpi=300)
  plt.show()
  # then, plot the alpha vs d plot
  fig, ax = plt.subplots()
  alpha_means = np.array([xi.result.alpha_error_mean for xi in experiments_for_this_lambda])
  alpha_confidence_intervals = np.array([xi.result.alpha_error_confidence_interval for xi in experiments_for_this_lambda])
  ax.plot(x_axis, alpha_means)
  ax.fill_between(x_axis, alpha_means - alpha_confidence_intervals, alpha_means + alpha_confidence_intervals, color='r', alpha=.1)
  plt.title("Orientation Error of Estimated Angle alpha in radian vs $\mathit{d}$ \nwhen $\lambda_n$ = " + "{:.2f} mm".format(this_lambda_n_in_mm))
  plt.xlabel("$\mathit{d}$ (in mm)")
  plt.ylabel("Orientation Error (in radian)")
  plt.savefig(os.path.join(os.getcwd(), "alpha error vs d when λ = {:.2f}mm.jpg".format(this_lambda_n_in_mm)),pil_kwargs={'quality':95}, dpi=300)
  plt.show()

  plt.close('all')

