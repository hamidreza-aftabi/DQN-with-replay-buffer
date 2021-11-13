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


BATCH_SIZE = 1
GAMMA = 0.99
EPS_EXPLORATION = 0.15
TARGET_UPDATE = 10
NUM_EPISODES = 4000
TEST_INTERVAL = 25
LEARNING_RATE = 10e-4
RENDER_INTERVAL = 20
ENV_NAME = 'CartPole-v0'
PRINT_INTERVAL = 5

env = gym.make(ENV_NAME)
state_shape = len(env.reset())
n_actions = env.action_space.n

model = MyModel(state_shape, n_actions).to(device)
target = MyModel(state_shape, n_actions).to(device)
target.load_state_dict(model.state_dict())
target.eval()

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
#memory = ReplayBuffer()

def choose_action(state, test_mode=False):
    # TODO implement an epsilon-greedy strategy
    ee_tradeoff = np.random.random()

    if ee_tradeoff <= EPS_EXPLORATION:
        action = torch.tensor(env.action_space.sample())  
    else:
        action = torch.argmax(model(torch.tensor(state).float()))
    
    return action.view(-1,1)
    raise NotImplementedError()


Q = torch.zeros(BATCH_SIZE).view(-1,1)
y = torch.zeros(BATCH_SIZE).view(-1,1)
def optimize_model(state, action, next_state, reward, done, steps_done):
    
    # TODO given a tuple (s_t, a_t, s_{t+1}, r_t, done_t) update your model weights
    loss_func = nn.MSELoss()
    
    if done:
        y[steps_done-1] = torch.tensor(reward).float()
    else:
        y[steps_done-1] = (reward + GAMMA *  max(model(torch.tensor(next_state).float()))).detach()
        
    Q[steps_done-1] = model(torch.tensor(state).float())[action]    
        
    if steps_done == BATCH_SIZE:
        loss = loss_func(y,Q)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        steps_done = 0
    
    return steps_done 

        

def train_reinforcement_learning(render=False):
    steps_done = 0
    best_score = -float("inf")
    target = model

    for i_episode in range(1, NUM_EPISODES+1):
        episode_total_reward = 0
        state = env.reset()
        for t in count():
            action = choose_action(state)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0][0])
            steps_done += 1
            episode_total_reward += reward

            steps_done = optimize_model(state, action, next_state, reward, done, steps_done)

            state = next_state

            if render:
                env.render(mode='human')

            if done:
                if i_episode % PRINT_INTERVAL == 0:
                    print('[Episode {:4d}/{}] [Steps {:4d}] [reward {:.1f}]'
                        .format(i_episode, NUM_EPISODES, t, episode_total_reward))
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


if __name__ == "__main__":
    train_reinforcement_learning()
