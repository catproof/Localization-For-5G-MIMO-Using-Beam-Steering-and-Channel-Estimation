import numpy as np
import matplotlib.pyplot as plt
import math

#code is following equations primarily from:
#Advanced Antenna Systems for 5G Network Deployments Bridging the Gap Between Theory and Practice

#gets a response vector for a ULA
def get_response_vector(N,phi,d,lambda_n):
    antennas = np.arange(N)
    response_vector = np.exp(1j*antennas*2*math.pi*d*np.sin(phi)/lambda_n)/math.sqrt(N)
    return response_vector

#classical beamforming using a ULA
def get_beamforming_vector(N,theta,d,lambda_n):
    antennas = np.arange(N)
    beamforming_vector = np.exp(-1j*antennas*2*math.pi*d*np.sin(theta)/lambda_n)
    return beamforming_vector


#random beamforming vector using a ULA
def get_random_beamforming_vector(N,d,lambda_n):
    return np.exp(1j * np.random.rand(N, 1)[:,0] * 2 * np.pi)


#gets the gain at the specified angle
def get_gain_at_theta(Nt,theta,d,lambda_n,beamforming_weight_vector):
    array_response_vector = get_response_vector(Nt,theta,d,lambda_n)
    return np.abs(np.matmul(array_response_vector, beamforming_weight_vector))**2 #equation 4.57 in AAS textbook


#calculates the channel matrix (except for the gain and path loss constant)
def get_channel(AoA, AoD, N, d, lambda_n):
    array_response_vector_t = get_response_vector(N,AoD,d,lambda_n)
    array_response_vector_r = get_response_vector(N,AoA,d,lambda_n)
    return np.outer(array_response_vector_r, np.conjugate(array_response_vector_t))


def get_beamforming_pattern(angles_of_departure, Nt, beam_steering_angle, d, lambda_n):
    gains = np.zeros((angles_of_departure.shape[0]))
    beamforming_weight_vector = get_beamforming_vector(Nt,beam_steering_angle,d,lambda_n)
    #beamforming_weight_vector = get_random_beamforming_vector(Nt,d,lambda_n)
    for theta_index in range(angles_of_departure.shape[0]):
        gains[theta_index] = get_gain_at_theta(Nt,angles_of_departure[theta_index],d,lambda_n,beamforming_weight_vector)
    return gains

#plotting polar graph for the beam pattern
def plot_polar_graph(x,y,args="", axis='on'):
    plt.axes(projection = 'polar')
    plt.axis(axis)
    plt.polar(x,y,args)
    plt.show()
    
#plotting polar graph for the beam pattern
def plot_polar_graph_with_3_patterns(x,y1,y2,y3,args="", axis='on'):
    plt.axes(projection = 'polar')
    plt.axis(axis)
    plt.polar(x,y1,"b")
    plt.axis(axis)
    plt.polar(x,y2,"r")
    plt.axis(axis)
    plt.polar(x,y3,"g")
    plt.savefig('test.png', format='png', dpi=600)
    plt.show()
    

def plot_4_polar_graphs(xs,ys,rticks=[],axis='on'):
    rlabel_position = -66.5
    
    fig, axs = plt.subplots(2,2,subplot_kw={'projection': 'polar'},figsize=(5,5))
    axs = axs.flatten()
    (ax1, ax2, ax3, ax4) = axs
    
    ax1.plot(xs[0], ys[0], color="r",  linewidth=1)
    ax1.set_title("θ = 0")
    ax1.set_rticks(rticks)
    ax1.set_rmax(rticks[-1])
    ax1.set_rlabel_position(rlabel_position)
    
    ax2.plot(xs[1], ys[1], color="r",  linewidth=1)
    ax2.set_title("θ = π/6")
    ax2.set_rticks(rticks)
    ax2.set_rmax(rticks[-1])
    ax2.set_rlabel_position(rlabel_position)
    
    ax3.plot(xs[2], ys[2], color="r",  linewidth=1)
    ax3.set_title("θ = π/3")
    ax3.set_rticks(rticks)
    ax3.set_rmax(rticks[-1])
    ax3.set_rlabel_position(rlabel_position + 12)
    
    ax4.plot(xs[3], ys[3], color="r",  linewidth=1)
    ax4.set_title("θ = π/2")
    ax4.set_rticks(rticks)
    ax4.set_rmax(rticks[-1])
    ax4.set_rlabel_position(rlabel_position + 18.5)
    
    fig.tight_layout()
    plt.savefig('test.png', format='png', dpi=300)
    plt.show()
    
    
#plotting line graph for the beam pattern
def plot_line_graph(x,y):
    plt.plot(x, y)
    plt.title('Antenna Gain Vs Angle of Departure')
    plt.xlabel('Angle of Departure')
    plt.ylabel('Antenna Gain')
    plt.show()
    
    
def get_max_angles_and_gain(x,y):
    max_gain = -math.inf
    maximum_angles = []
    for i in range(x.shape[0]):
        if y[i] > max_gain:
            max_gain = y[i]
            maximum_angles = []
            maximum_angles.append(x[i])
        elif y[i] == max_gain:
            maximum_angles.append(x[i])
    return maximum_angles, max_gain

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

beam_steering_angle = 0
angles_of_departure = np.arange(0, (2 * np.pi), 0.01)
gains = get_beamforming_pattern(angles_of_departure, 5, beam_steering_angle, d, lambda_n)

plot_line_graph(angles_of_departure, gains)
plot_polar_graph(angles_of_departure, gains, axis="on")

angles_of_departure = np.arange(0, (2 * np.pi), 0.01)
Nt = 5
rticks = [1,2,3]
gains1 = get_beamforming_pattern(angles_of_departure, Nt, 0.35, d, lambda_n)
gains2 = get_beamforming_pattern(angles_of_departure, Nt, 0.59, d, lambda_n)
gains3 = get_beamforming_pattern(angles_of_departure, Nt, 0.83, d, lambda_n)
gains4 = get_beamforming_pattern(angles_of_departure, Nt, math.pi/2, d, lambda_n)
#plot_4_polar_graphs([angles_of_departure,angles_of_departure,angles_of_departure,angles_of_departure], [gains1,gains2,gains3,gains4], rticks, axis="on")
plot_polar_graph_with_3_patterns(angles_of_departure,gains1,gains2,gains3,args="", axis='off')
'''