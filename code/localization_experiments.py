from beam_switching_localization import beam_switching_localization, get_beam_switching_angles
from beam_switching_and_CSI_localization_combined import beam_switching_and_CSI_localization, beam_switching_and_CSI_localization_v2
import numpy as np
import math
from util import random_MS_pose_within_circle, get_AoD, get_Orientation_Error, get_Distance_Error
from main_for_DCS_SOMP import fiveG_positioning_experiement

Nt = 32            #number of antennas
c = 299792458      #speed of light in meter / second
f = 25000000000    #frequency of the signal in Hz
lambda_n = c/f     #wave length for subcarrier n
d = lambda_n/2     #spacing between antennas
beam_switching_angles = get_beam_switching_angles(Nt)

BS_x = 0    #x position of the base station in meters
BS_y = 0    #y position of the base station in meters


def get_confidence_interval_and_mean(y):
  confidence_interval = 1.96 * np.std(y, axis=0)/math.sqrt(y.shape[0])
  mean = np.mean(y, axis=0)
  return confidence_interval, mean


num_experiments = 500

estimation = None
ground_truth = None
estimation = np.load("estimation_2.npy")
ground_truth = np.load("ground_truth_2.npy")
num_experiments = ground_truth.shape[0]

positioning_errors = np.zeros(num_experiments)
AoD_errors = np.zeros(num_experiments)
distance_errors = np.zeros(num_experiments)
for i in range(num_experiments):
    MS_x = ground_truth[i][0]
    MS_y = ground_truth[i][1]
    MS_alpha = ground_truth[i][2]
    
    position_estimation = estimation[i][0], estimation[i][1]
    
    positioning_errors[i] = math.hypot(position_estimation[0] - MS_x, position_estimation[1] - MS_y)
    AoD_errors[i] = get_Orientation_Error(get_AoD(MS_x, MS_y, BS_x, BS_y),get_AoD(position_estimation[0], position_estimation[1],BS_x, BS_y))
    distance_errors[i] = get_Distance_Error(MS_x, MS_y, BS_x, BS_y, position_estimation[0], position_estimation[1])

print("\nCSI localization")
positiong_error = get_confidence_interval_and_mean(positioning_errors)
print("positioning error: " + str(positiong_error[1]) + " ± " + str(positiong_error[0]))
AoD_error = get_confidence_interval_and_mean(AoD_errors)
print("AoD error: " + str(AoD_error[1]) + " ± " + str(AoD_error[0]))
distance_error = get_confidence_interval_and_mean(distance_errors)
print("distance error: " + str(distance_error[1]) + " ± " + str(distance_error[0]))



positioning_errors = np.zeros(num_experiments)
AoD_errors = np.zeros(num_experiments)
distance_errors = np.zeros(num_experiments)
num_beams = 50
print("total possible number of beams: " + str(beam_switching_angles.shape[0]))
avg_num_beams = np.zeros(num_experiments)

#best params are 50 num beams and threshold of 6.0 when using 32 antennas

#evaluate beam switching localization error
for i in range(num_experiments):
    #MS_x, MS_y, MS_alpha = random_MS_pose_within_circle(3,10)
    MS_x = ground_truth[i][0]
    MS_y = ground_truth[i][1]
    MS_alpha = ground_truth[i][2]
    
    noise = 1.5
    #CSI_localization = MS_x + np.random.normal(0,noise), MS_y + np.random.normal(0,noise), 0
    CSI_localization = estimation[i][0], estimation[i][1], 0
    
    #position_estimation = beam_switching_and_CSI_localization(CSI_localization, MS_x, MS_y, BS_x, BS_y, Nt, d, lambda_n, beam_switching_angles, num_beams)
    threshold = 6
    result_x, result_y, num_used_beams = beam_switching_and_CSI_localization_v2(CSI_localization, MS_x, MS_y, BS_x, BS_y, Nt, d, lambda_n, beam_switching_angles, num_beams, threshold)
    
    position_estimation = result_x, result_y
    positioning_errors[i] = math.hypot(position_estimation[0] - MS_x, position_estimation[1] - MS_y)
    AoD_errors[i] = get_Orientation_Error(get_AoD(MS_x, MS_y, BS_x, BS_y),get_AoD(position_estimation[0], position_estimation[1],BS_x, BS_y))
    distance_errors[i] = get_Distance_Error(MS_x, MS_y, BS_x, BS_y, position_estimation[0], position_estimation[1])
    avg_num_beams[i] = num_used_beams
    

print("\nCSI localization and beam switching localization combined")
positiong_error = get_confidence_interval_and_mean(positioning_errors)
print("positioning error: " + str(positiong_error[1]) + " ± " + str(positiong_error[0]))
AoD_error = get_confidence_interval_and_mean(AoD_errors)
print("AoD error: " + str(AoD_error[1]) + " ± " + str(AoD_error[0]))
distance_error = get_confidence_interval_and_mean(distance_errors)
print("distance error: " + str(distance_error[1]) + " ± " + str(distance_error[0]))
avg_num_beams = get_confidence_interval_and_mean(avg_num_beams)
print("avg_num_beams: " + str(avg_num_beams[1]) + " ± " + str(avg_num_beams[0]))