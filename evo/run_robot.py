from osim.env import RunEnv
import numpy as np
import copy
import pickle

env = RunEnv(visualize=False)
observation = env.reset(difficulty = 0)
sin=np.sin
file_Name = "w_best"

array=np.array

T=4


alpha=0.05
alpha_0=0.05
#TODO: we should exploit the Fourier property for which higher harmonics weights 
#tend to decays as 1/x^n for smooth and continous functions

#I initialize to 0 the weights list, 4 weights for each muscle (I compose the periodic function with 4 elements of a Fourier Series)
#I define weights only for 9 periodic functions, as I assume that the legs move symmetrically in time.
#Now I use 8 weights: 4 amplitudes for harmonics and 4 phases

w=[]

for i in range(9):
    w.append(np.array([0.,0.,0.,0.,0.,0.,0.,0.]))



def output(a,T,t):
    # Output of a 4th degree Fourier Series of sin.
    # INPUT: the harmonics weights, time period T, and the time t.
        y=0
        
        for i in range(4):
            y += a[i] * sin( (i+1) * 2*np.pi*t/T + a[i+4] )
        return y


def evolve(w):
    #This functions evolves randomly w generating a direction, sampling from gaussians distribution.
    #It operates directly on w so it doesn't return anything.

    for i in range(9):
        w[i]+=np.random.randn(8)*alpha
    """
    for i in range(9):
        delta=[]
        for j in range(4):
            delta.append(np.random.randn()*0.5/((j+1)**2))
        delta=np.asarray(delta)
        w[i]+=delta
    """


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





"""inputs=[]
for i in range(400):
    inputs.append([-np.sin(i*np.pi/200)*0.2, np.sin(i*np.pi/200)*0.2,-np.sin(i*np.pi/200)*0.2,
        np.sin(i*np.pi/200)*0.2,np.sin(i*np.pi/200)*0.2,np.sin(i*np.pi/200)*0.2,
        -np.sin(i*np.pi/200)*0.2*1.2,-np.sin(i*np.pi/200)*0.2,np.sin(i*np.pi/200)*0.2,np.sin(i*np.pi/200)*0.2,-np.sin(i*np.pi/200)*0.2,np.sin(i*np.pi/200)*0.2,-np.sin(i*np.pi/200)*0.2,-np.sin(i*np.pi/200)*0.2,-np.sin(i*np.pi/200)*0.2,np.sin(i*np.pi/200)*0.2*1.2,np.sin(i*np.pi/200)*0.2,-0.2*np.sin(i*np.pi/200)])
"""


############# MAIN ################

#Initialize the data structures that will be identical to w.
#best_w will be the best performing weights.
#w_try will be the current try

#I initially used pickle to save and load the best weights but for now I won't use it

try:
    fileObject = open(file_Name,'r')
    w_best = pickle.load(fileObject)
    fileObject.close()
    print("Best loaded!")
except:
    #w_best=copy.deepcopy(w)
     
    w_best=([array([ 0.23100136, -0.20097108, -0.45689656, -0.24844133, -0.11952992,
       -0.18181525,  0.29405677, -0.1308828 ]), array([ 0.22361338, -0.84184931,  0.04807946, -0.32798469, -0.55832818,
        0.15862826, -0.7754661 , -0.46233336]), array([-0.4314024 ,  0.08888579, -0.13188666, -0.64162866,  0.11962261,
        0.48288929,  1.23094164, -0.36075015]), array([ 0.31086194,  0.19280792,  0.38097105, -0.45227432,  0.09288655,
        0.13043282,  0.32774874,  0.13832417]), array([-0.43868369, -0.3173145 ,  0.47171041,  0.38198377, -0.12851197,
       -0.14754433, -0.25027703, -0.31168021]), array([-0.28856407, -0.2194842 ,  0.34120592, -0.08137398,  0.37970297,
        0.34144053,  0.04027321, -0.39641663]), array([-0.51242177, -0.1771355 ,  0.17712722,  0.09732865,  0.21182721,
        0.89530259,  0.29641186,  1.09922087]), array([-0.7544521 , -0.13492536,  0.51341343, -0.40059504, -0.8089869 ,
       -0.32452642, -0.04910592,  0.80597189]), array([ 0.5293721 ,  0.44560003,  0.48018323, -0.07223837,  0.57207472,
        1.20298059, -0.10919251,  0.30443165])])
    
    print("Initializing new best")

w_try=copy.deepcopy(w)
best_reward=0.
runs=500
unev_runs=0

print("Baseline, run with w_best")
observation = env.reset(difficulty = 0)
total_reward = 0.0
for i in range(300):
    i*=0.01
    # make a step given by the controller and record the state and the reward
    observation, reward, done, info = env.step(input(w_best,i))
    total_reward += reward
    if done:
        break
best_reward=total_reward

# Your reward is
print("Total reward %f" % total_reward)


for run in range(runs):

    #if it doens't get better for more than 10 iterations, increase alpha to allow bigger changes
    #Increase alpha then set unev_runs back to 0
    if unev_runs>30:
        print("Augmenting alpha")
        alpha+=alpha_0
        unev_runs=0


    unev_runs+=1
    print("Run {}/{}".format(run,runs))
    observation = env.reset(difficulty = 0)

    #I copy the best performing w and I try to evolve it
    w_try=copy.deepcopy(w_best)
    evolve(w_try)

    total_reward = 0.0
    for i in range(300):
        # make a step given by the controller and record the state and the reward
        i*=0.01 #Every step is 0.01 s

        observation, reward, done, info = env.step(input(w_try,i))
        total_reward += reward
        if done:
            print("done")
            break

    if total_reward>best_reward:
        #If the total reward is the best one, I store w_try as w_best, dump it with pickle and save the reward
        print("Found a better one!")
        unev_runs=0
        alpha=alpha_0
        w_best=copy.deepcopy(w_try)
        print(w_best)

        
        fileObject = open(file_Name,'wb')
        #pickle.dump(w_best,fileObject)
        fileObject.close()
        best_reward=total_reward

        

    # Your reward is
    print("Total reward %f" % total_reward)



#Final run with video and best weights. The raw_input waits for the user to type something. (if it's afk)
print("Run with best weights")
_=raw_input("ready? ")
env = RunEnv(visualize=True)
observation = env.reset(difficulty = 0)


total_reward = 0.0
for i in range(300):
    # make a step given by the controller and record the state and the reward
    i*=0.01 #Every step is 0.01 s

    observation, reward, done, info = env.step(input(w_best,i))
    total_reward += reward
    if done:
        break

# Your reward is
print("Total reward %f" % total_reward)

print("best weights")
print(w_best)
