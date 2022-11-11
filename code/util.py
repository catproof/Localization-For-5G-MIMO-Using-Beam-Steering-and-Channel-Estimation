
import string

import numpy as np

import math

import random

def print_debug(str : string):
  print("[DEBUG] " + str)


def norm_2(a: np.ndarray):
  # This function is written to try to make a matlab equalivent norm() operation in python
  # numpy by default uses Frobenius norm while matlab uses 2-norm
  # however, even though specifying numpy to use 2-norm by calling norm(A, 2), the result seems to still be different than matlab's norm()
  # which causing TOA result to be different from matlab's when fixing the SP array
  return np.linalg.norm(a, 2)

def get_Orientation_Error(angle1, angle2):
    return min(abs((angle1 - angle2) % (2 * math.pi)), abs((angle2 - angle1) % (2 * math.pi)))

def get_Distance_Error(MS_x, MS_y, BS_x, BS_y, estimation_x, estimation_y):
    return abs(math.hypot(MS_x - BS_x, MS_y - BS_y) - math.hypot(estimation_x - BS_x, estimation_y - BS_y))

def get_AoD(MS_x, MS_y, BS_x, BS_y):
    AoD = math.atan2(MS_y - BS_y, MS_x - BS_x)
    if AoD < 0 :
        return AoD + (2 * math.pi)
    return AoD

def random_MS_pose_within_square(max_x, max_y, min_x, min_y):
    return random.uniform(min_x, max_x), random.uniform(min_y, max_y), random.uniform(0, 2 * math.pi)

def random_MS_pose_within_circle(min_distance, max_distance):
    distance = random.uniform(min_distance, max_distance)
    AoD = random.uniform(0, 2 * math.pi)
    alpha = random.uniform(0, 2 * math.pi)
    MS_x = distance * math.cos(AoD)
    MS_y = distance * math.sin(AoD)
    return MS_x, MS_y, alpha

# debug code for norm()
# with norm(x, 2), the results below are:

# A = np.array([[-4, -3, -2],
#        [-1,  0,  1],
#        [ 2,  3,  4]])
# print_debug("norm of A  is {}".format(norm_2(A)))  # is 7.348469228349534

# B = np.arange(-4, 5, 1)
# print_debug("norm of B  is {}".format(norm_2(B)))  # is 7.745966692414834

# BB = np.array([np.arange(-4, 5, 1)])
# print_debug("norm of BB is {}".format(norm_2(BB))) # is 7.745966692414833

# B3 = np.array([[np.arange(-4,5,1)]])
# print_debug("norm of B3 is {}".format(norm_2(B3))) # will not work

# A1 = np.array([3, 3, 3])
# B1 = np.array([4, 4, 4])

# T1 = np.array([3, 3]).transpose()
# print_debug("T1 norm is {}".format(norm_2(T1))) # is 4.242640687119285
# T2 = np.array([-3, -3]).transpose()
# print_debug("T2 norm is {}".format(norm_2(T2))) # is 4.242640687119285

### vector elementary test, it shows that there is no concept of column vector in python
# v1 = np.array([3,3]).transpose()
# v2 = np.array([1,1])
# print_debug("v1 - v2[0] = {}".format(v1 - v2[0]))

# python complex matrix tranpose = .conjugate().transpose()
# c1 = np.array([[1+1j, 2-2j], [3-3j, 4+4j]])
# c2 = np.array([[5+5j, 6-6j], [7-7j, 8+8j]])
# print( "c1 direct transpose = {}, c1 conjugate transpose = {}".format(c1.transpose(), c1.conjugate().transpose()))

# complex matrix multiplication
# c1_trans = c1.conjugate().transpose()
# multi_res = c1_trans.dot(c2)
# print("c1' * c2 = {}".format(multi_res))

# complex matrix norm
# print("norm of complex matrix c1 = {}".format(norm_2(c1))) # is 7.23606797749979 same as matlab
