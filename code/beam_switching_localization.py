import math
from classical_beamforming import get_beamforming_vector, get_gain_at_theta
import numpy as np
from util import get_AoD

#Gets the received signal strength of the signal received at the mobile station.
#It assumes that the mobile station measures the RSS using a single antenna.
#Uses the Friis transmission equation to calculate RSS. Assumes the transmitted 
#power is 1.
#In the future, we might use the AoD and AoA to determine a recieve beam former, 
#and incorporate that into the RSS reading.
def get_MS_RSS(MS_x, MS_y, BS_x, BS_y, Nt, d, lambda_n, beamforming_weight_vector):
    AoD = get_AoD(MS_x, MS_y, BS_x, BS_y)
    transmit_gain = get_gain_at_theta(Nt,AoD,d,lambda_n,beamforming_weight_vector)
    receive_gain = 1
    distance = math.hypot(MS_x - BS_x, MS_y - BS_y)
    #for now, using a normal distribution ceneter at 1 to scale the distance
    noise = np.random.normal(1,0.4)
    #see beginning of section 5.A of the paper:
    #Position and Orientation Estimation Through Millimeter-Wave MIMO in 5G Systems
    #for a more realistic noise model for path loss
    return transmit_gain * receive_gain * (lambda_n/(4 * math.pi * distance * noise))**2
    
    
#Using the Friis transmission equation.
#Assume the receiver does not beamform, and RSS is measured by a signle
#antenna.
def get_MS_distance_from_RSS(RSS, Nt, lambda_n):
    max_transmit_gain = Nt
    max_receive_gain = 1 #assume the receiver measures RSS with one antenna
    return math.sqrt(max_transmit_gain) * math.sqrt(max_receive_gain) * lambda_n / (4 * math.pi * math.sqrt(RSS))

#Perform localization using beam steering and RSS
def beam_switching_localization(MS_x, MS_y, BS_x, BS_y, Nt, d, lambda_n, beam_steering_angles):
    max_RSS = -math.inf
    max_RSS_angle = None
    for beam_steering_angle in beam_steering_angles:
        beamforming_weight_vector = get_beamforming_vector(Nt,beam_steering_angle,d,lambda_n)
        RSS = get_MS_RSS(MS_x, MS_y, BS_x, BS_y, Nt, d, lambda_n, beamforming_weight_vector)
        if RSS > max_RSS:
            max_RSS = RSS
            max_RSS_angle = beam_steering_angle
    
    max_RSS_angle_1 = max_RSS_angle 
    max_RSS_angle_2 = (math.pi - max_RSS_angle) % (2 * math.pi) 
    distance = get_MS_distance_from_RSS(max_RSS, Nt, lambda_n)
    possible_location_1 = (distance * math.cos(max_RSS_angle_1), distance * math.sin(max_RSS_angle_1))
    possible_location_2 = (distance * math.cos(max_RSS_angle_2), distance * math.sin(max_RSS_angle_2))
    return possible_location_1, possible_location_2

def get_beam_switching_angles(Nt):
    return np.arange(0, 2 * np.pi, (1.78 / Nt))
    #return np.arange(-np.pi/2, np.pi/2, (1.78 / Nt))

''' 
Nt = 32            #number of antennas
c = 299792458      #speed of light in meter / second
f = 25000000000    #frequency of the signal in Hz
lambda_n = c/f     #wave length for subcarrier n
d = lambda_n/2     #spacing between antennas
beam_steering_angles = np.arange(0, (2 * np.pi), 0.01)

BS_x = 0    #x position of the base station in meters
BS_y = 0    #y position of the base station in meters
MS_x = 800    #x position of the mobile station in meters
MS_y = 0  #y position of the mobile station in meters

#print("The beam steering determined the mobile station is in one of these 2 positions:")
#print(beam_steering_localization(MS_x, MS_y, BS_x, BS_y, Nt, d, lambda_n, beam_steering_angles))

#print("The beam steering determined the mobile station is in one of these 2 positions:")
#print(beam_steering_localization(MS_x, MS_y, BS_x, BS_y, Nt, d, lambda_n, [get_AoD(MS_x, MS_y, BS_x, BS_y)]))

#np.set_printoptions(suppress=True)
#print(get_channel(0.5, 0.5, 3, d, lambda_n))
'''