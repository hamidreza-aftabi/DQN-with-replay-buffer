import gym
import math
import random
from itertools import count
import torch
from eval_policy import eval_policy, device
from model import MyModel
from replay_buffer import ReplayBuffer
import torch.nn as nn
import numpy as np
import copy



BATCH_SIZE = 256
GAMMA = 0.99
EPS_EXPLORATION = 0.2
TARGET_UPDATE = 3
NUM_EPISODES = 6000
TEST_INTERVAL = 25
LEARNING_RATE = 10e-4
RENDER_INTERVAL = 20
ENV_NAME = 'CartPole-v0'
PRINT_INTERVAL = 10
BUFFER = False

env = gym.make(ENV_NAME)
state_shape = len(env.reset())
n_actions = env.action_space.n

loss_func = nn.MSELoss()

model = MyModel(state_shape, n_actions)
target = MyModel(state_shape, n_actions)
target.load_state_dict(model.state_dict())
target.eval()


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
memory = ReplayBuffer()

def choose_action(state, i_episode, test_mode=False):
    # TODO implement an epsilon-greedy strategy
    
    epsilon = max(0.03, 0.3 - 0.01*(i_episode/200)) 

    ee_tradeoff = np.random.random()

    if ee_tradeoff <= epsilon:
        action = torch.tensor(env.action_space.sample())  
    else:
        action = torch.argmax(model(torch.tensor(state).float()))
    
    return action.view(-1,1)
    raise NotImplementedError()



def optimize_model(state, action, next_state, reward, done):
    
    # TODO given a tuple (s_t, a_t, s_{t+1}, r_t, done_t) update your model weights
    with torch.no_grad():
        q_next= target(next_state).float()
       
    out, inds = torch.max(q_next,dim=1)
    status  = torch.logical_not(done)+0
    y = (reward + GAMMA * status * out).view(-1,1)
    q = model(state)
    Q = torch.zeros(batch,1)
    
    for i in range(batch):
        Q[i] = q[i,action[i].type(torch.LongTensor)] 
        
    loss = loss_func(y,Q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

        

steps_done = 0
best_score = -float("inf")
render=False

for i_episode in range(1, NUM_EPISODES+1):
    episode_total_reward = 0
    state = env.reset()
    
    for t in count():
        action = choose_action(state,i_episode)
        next_state, reward, done, _ = env.step(action.cpu().numpy()[0][0])
        steps_done += 1
        episode_total_reward += reward
            
        if BUFFER == True: 
            memory.push(state, action, next_state, reward, done)

            if i_episode > BATCH_SIZE:
                
                batch = BATCH_SIZE
                
                state_buf, action_buf, next_state_buf, reward_buf, done_buf= memory.sample(batch)
                optimize_model(state_buf, action_buf, next_state_buf, reward_buf, done_buf)
                
        else:
            
            batch = 1
            
            state1 = torch.tensor(state).view(1,-1).float()
            action1 = action.view(1,-1).float()
            next_state1 = torch.tensor(next_state).view(1,-1).float()
            reward1 = torch.tensor(reward).view(1,-1).float()
            done1 = torch.tensor(done).view(1,-1)
            
            optimize_model(state1, action1 , next_state1 , reward1, done1)
          
            
        state = next_state
 
                
        if render:
            env.render(mode='human')

        if done:
            if i_episode % PRINT_INTERVAL == 0:
                print('[Episode {:4d}/{}] [Steps {:4d}] [reward {:.1f}]'.format(i_episode, NUM_EPISODES, t, episode_total_reward))
            break
        


    if i_episode % TARGET_UPDATE == 0:
        target.load_state_dict(model.state_dict())

    if i_episode % TEST_INTERVAL == 0:
        print('-'*10)
        score = eval_policy(policy=model, env=ENV_NAME, render=render)
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), "best_model_{}.pt".format(ENV_NAME))
            print('saving model.')
        print("[TEST Episode {}] [Average Reward {}]".format(i_episode, score))
        print('-'*10)


