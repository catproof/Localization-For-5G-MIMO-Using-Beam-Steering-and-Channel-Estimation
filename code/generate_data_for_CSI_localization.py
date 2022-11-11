import numpy as np
from util import random_MS_pose_within_circle
from main_for_DCS_SOMP import fiveG_positioning_experiement



Nt = 32            #number of antennas
c = 299792458      #speed of light in meter / second
f = 25000000000    #frequency of the signal in Hz
lambda_n = c/f     #wave length for subcarrier n
d = lambda_n/2     #spacing between antennas
print(lambda_n)
print(d)

print("d")
print("λ")

BS_x = 0    #x position of the base station in meters
BS_y = 0    #y position of the base station in meters
MS_x = 800    #x position of the mobile station in meters
MS_y = 0  #y position of the mobile station in meters

d = 0.006076874148648648
lambda_n = 0.008102498864864865

num_experiments = 500
ground_truth = np.zeros((num_experiments,3))
estimation  = np.zeros((num_experiments,2))
for i in range(num_experiments):
    print("at experiment: " + str(i))
    MS_x, MS_y, alpha = random_MS_pose_within_circle(1, 5)
    iteration_name = ""
    posRx = MS_x, MS_y
    _, _, posRx_hat, alpha_hat = fiveG_positioning_experiement(posRx, alpha, d, lambda_n, iteration_name)
    estimation[i][0] = posRx_hat[0] 
    estimation[i][1] = posRx_hat[1] 
    ground_truth[i][0] = MS_x 
    ground_truth[i][1] = MS_y
    ground_truth[i][2] = alpha 

print(estimation)
print(ground_truth)
np.save("estimation_2" , estimation)
np.save("ground_truth_2", ground_truth)

    