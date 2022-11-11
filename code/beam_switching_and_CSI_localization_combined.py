import math
from beam_switching_localization import beam_switching_localization
from util import get_AoD
import numpy as np

#As per this link: https://jwcn-eurasipjournals.springeropen.com/track/pdf/10.1186/s13638-018-1209-z.pdf
#the beam index corresponds to a beamformer in the codebook. The codebook is a collection
#of beam forming vectors. I can refactor the code later to use a codebook
#Assumes beam_steering_angles is sorted, and in the range of 0 to 2 * math.pi.
def find_closest_beam_index(beam_steering_angles, estimated_angle):
    curr_closest_beam_index = None
    curr_smallest_diff = math.inf
    for curr_beam_index in range(len(beam_steering_angles)):
        curr_diff = abs(beam_steering_angles[curr_beam_index] - estimated_angle)
        if curr_diff < curr_smallest_diff:
            curr_smallest_diff = curr_diff
            curr_closest_beam_index = curr_beam_index
        else:
            break
    
    if curr_closest_beam_index == len(beam_steering_angles) - 1:
        first_beam_steering_angle = beam_steering_angles[0] + 2 * math.pi
        curr_diff = abs(first_beam_steering_angle - beam_steering_angles[-1]) 
        if curr_diff < curr_smallest_diff:
            curr_smallest_diff = curr_diff
            curr_closest_beam_index = 0
    elif curr_closest_beam_index == 0:
        last_beam_steering_angle = beam_steering_angles[-1] - 2 * math.pi
        curr_diff = abs(last_beam_steering_angle - beam_steering_angles[0]) 
        if curr_diff < curr_smallest_diff:
            curr_smallest_diff = curr_diff
            curr_closest_beam_index = len(beam_steering_angles) - 1 
    
    closest_beam_index = curr_closest_beam_index
    return closest_beam_index

#Picks a subset of all possible beam steering angles by choosing the 'num_beams'
#closest beam steering angles to 'closest_beam_index'.
#Assumes 'num_beams' is odd.
def refine_beam_steering_angles(beam_steering_angles, closest_beam_index, num_beams):
    beginning_index = int(closest_beam_index - (num_beams-1)/2)
    refined_beam_steering_angles = []
    for beam_steering_index in range(beginning_index,num_beams + beginning_index):
        curr_index = beam_steering_index % len(beam_steering_angles)
        refined_beam_steering_angles.append(beam_steering_angles[curr_index])
    return refined_beam_steering_angles


#An initial localization guess using channel estimation localization, followed by
#a refinement using beam steering.
#from the paper of interest:"Moreover, the proposed algorithm performs well even 
#for very low values of the received SNR, which is the typical case at mm-wave systems 
#before beamforming" we can assume that the position estimation is ran once, with an
#unspecified beamformer, then the beam steering is used to refine the position
def beam_switching_and_CSI_localization(CSI_localization, MS_x, MS_y, BS_x, BS_y, Nt, d, lambda_n, beam_steering_angles, num_beams):
    CSI_x, CSI_y, CSI_alpha = CSI_localization
    closest_beam_index = find_closest_beam_index(beam_steering_angles, CSI_alpha)
    refined_beam_steering_angles = refine_beam_steering_angles(beam_steering_angles, closest_beam_index, num_beams)
    beam_steer_results = beam_switching_localization(MS_x, MS_y, BS_x, BS_y, Nt, d, lambda_n, refined_beam_steering_angles)
    best_result = None
    closest_distance = math.inf
    curr_distance = None
    for result in beam_steer_results:
        curr_distance = math.hypot(CSI_x - result[0], CSI_y - result[1])
        if curr_distance < closest_distance:
            best_result = result
            closest_distance = curr_distance
            
    return best_result[0], best_result[1]


#same as beam_switching_and_CSI_localization, but does an initial small beam
#sweep based on the initial CSI localization, and then if the results between
#the beam switching localization and the CSI localization disagree too much,
#the beam switching algorithm is ran again but with a full beam sweep
def beam_switching_and_CSI_localization_v2(CSI_localization, MS_x, MS_y, BS_x, BS_y, Nt, d, lambda_n, beam_steering_angles, num_beams, threshold):
    CSI_x, CSI_y, CSI_alpha = CSI_localization
    result_x, result_y = beam_switching_and_CSI_localization(CSI_localization, MS_x, MS_y, BS_x, BS_y, Nt, d, lambda_n, beam_steering_angles, num_beams)
    
    estimated_distance_beam_switching = math.hypot(result_x - BS_x, result_y - BS_y) 
    estimated_distance_CSI = math.hypot(CSI_x - BS_x, CSI_y - BS_y)
    old_num_beams = 0
    if estimated_distance_beam_switching > estimated_distance_CSI * threshold:
        old_num_beams = num_beams
        num_beams = beam_steering_angles.shape[0]
        result_x, result_y = beam_switching_and_CSI_localization(CSI_localization, MS_x, MS_y, BS_x, BS_y, Nt, d, lambda_n, beam_steering_angles, num_beams)

    return result_x, result_y, num_beams + old_num_beams

'''
Nt = 32            #number of antennas
c = 299792458      #speed of light in meter / second
f = 25000000000    #frequency of the signal in Hz
lambda_n = c/f     #wave length for subcarrier n
d = lambda_n/2     #spacing between antennas
beam_steering_angle = 0 * math.pi/10
beam_steering_angles = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi]
#beam_steering_angles = [math.pi/2]
beam_steering_angles = np.arange(0, (2 * np.pi), 0.01)

BS_x = 0    #x position of the base station in meters
BS_y = 0    #y position of the base station in meters
MS_x = 800    #x position of the mobile station in meters
MS_y = 0  #y position of the mobile station in meters

print("The beam steering combined with CSI localization yielded:")
CSI_localization = MS_x, MS_y, get_AoD(MS_x, MS_y, BS_x, BS_y)
num_beams = 5
print(beam_steering_and_CSI_localization(CSI_localization, MS_x, MS_y, BS_x, BS_y, Nt, d, lambda_n, beam_steering_angles, num_beams))
'''