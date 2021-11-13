import pickle
import numpy as np
with open("CartPole-v0_dataset.pkl", 'rb') as f:
    data = pickle.load(f)
    

state= np.zeros((len(data),4))

for i  in range(len(data)):
    state[i,:] = z=data[i][0]