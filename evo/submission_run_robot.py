import opensim as osim
from osim.http.client import Client
from osim.env import RunEnv
import numpy as np
import copy
import pickle

# IMPLEMENTATION OF YOUR CONTROLLER

T=4

alpha=0.05
alpha_0=0.05

def output(a,T,t):
    # Output of a 4th degree Fourier Series of sin.
    # INPUT: the harmonics weights, time period T, and the time t.
        y=0
        
        for i in range(4):
            y += a[i] * np.sin( (i+1) * 2*np.pi*t/T + a[i+4] )
        return y

def input(w,t):
	
    #This function generates the input to the model.
    #INPUT: weights, time step
    #Returns: input arrat
    global T
    inputs=[]
    
    #The model input has dimension 18. Eventhough, I only use 9 functions, since the last 9 are just the same but with a -
    #When one of the inputs is <0 it is =0 for the model.
    """
    inputs=[-output(w[0],T,t),output(w[1],T,t),-output(w[2],T,t), output(w[3],T,t),output(w[4],T,t),output(w[5],T,t),-output(w[6],T,t),-output(w[7],T,t),output(w[8],T,t),
    output(w[0],T,t),-output(w[1],T,t),output(w[2],T,t), -output(w[3],T,t), -output(w[4],T,t), -output(w[5],T,t), output(w[6],T,t), output(w[7],T,t), -output(w[8],T,t),]
    """
    inputs = [-output(w[0],T,t),output(w[1],T,t),-output(w[2],T,t), 
    output(w[3],T,t),output(w[4],T,t),output(w[5],T,t),
    -output(w[6],T,t),-output(w[7],T,t),output(w[8],T,t),
    # consider the second legs with a phase
    output(w[0],T,t+T/2),-output(w[1],T,t+T/2),output(w[2],T,t+T/2), 
    -output(w[3],T,t+T/2), -output(w[4],T,t+T/2), -output(w[5],T,t+T/2),
     output(w[6],T,t+T/2), output(w[7],T,t+T/2), -output(w[8],T,t+T/2),]

    return inputs

w_best = ([np.array([ 0.23100136, -0.20097108, -0.45689656, -0.24844133, -0.11952992,-0.18181525,  0.29405677, -0.1308828 ]), 
	     np.array([ 0.22361338, -0.84184931,  0.04807946, -0.32798469, -0.55832818,0.15862826, -0.7754661 , -0.46233336]), 
	     np.array([-0.4314024 ,  0.08888579, -0.13188666, -0.64162866,  0.11962261,0.48288929,  1.23094164, -0.36075015]), 
	     np.array([ 0.31086194,  0.19280792,  0.38097105, -0.45227432,  0.09288655,0.13043282,  0.32774874,  0.13832417]), 
	     np.array([-0.43868369, -0.3173145 ,  0.47171041,  0.38198377, -0.12851197,-0.14754433, -0.25027703, -0.31168021]), 
	     np.array([-0.28856407, -0.2194842 ,  0.34120592, -0.08137398,  0.37970297,0.34144053,  0.04027321, -0.39641663]), 
	     np.array([-0.51242177, -0.1771355 ,  0.17712722,  0.09732865,  0.21182721,0.89530259,  0.29641186,  1.09922087]), 
	     np.array([-0.7544521 , -0.13492536,  0.51341343, -0.40059504, -0.8089869 ,-0.32452642, -0.04910592,  0.80597189]), 
	     np.array([ 0.5293721 ,  0.44560003,  0.48018323, -0.07223837,  0.57207472,1.20298059, -0.10919251,  0.30443165])])


# my_controller = ... (for example the one trained in keras_rl)

with open('crowdai_key.txt', 'r') as f:
    
    key = f.read()

# Settings
remote_base = "http://grader.crowdai.org:1729"
crowdai_token = key

client = Client(remote_base)

# Create environment
observation = client.env_create(crowdai_token)

# Run a single step
#score = 0
j = 0
while True:

	j += 1 

	i = j * 0.01
	[observation, reward, done, info] = client.env_step(input(w_best,i), True)
	#score += reward
	#print(observation)
	if done:
	    observation = client.env_reset()
	    j = 0
	    if not observation:
	        break
#print(score)
client.submit()