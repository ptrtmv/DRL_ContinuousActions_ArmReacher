'''
Created on Jun 7, 2019

@author: ptrtmv
'''

import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    
    def __init__(self,brain):
        self.brain = brain
        
    def act(self,state,noise=None):
        return self.brain.react(state,noise)
    
    def step(self, state, action, reward, next_state, done):    
        self.brain.experience(state, action, reward, next_state, done)

    def reset(self):
        self.brain.reset()

class Brain():
    
    def __init__(self, stateSize, actionSize, 
                 gamma = 0.99,
                 actorLearningRate = 1e-4,   
                 criticLearningRate = 1e-4,    
                 actorSoftHardUpdatePace = 1e-3,  
                 criticSoftHardUpdatePace = 1e-3,        
                 dnnUpdatePace = 4,  
                 bufferSize = int(1e5),
                 batchSize = 64, 
                 batchEpochs = 1,
                 weightDecay = 1e-5,
                 seed = None):
        '''
        Initialization of the brain object
        
        :param stateSize: 
                The dimension of the state space; number of features for the deep Q-network
        :param actionSize: 
                Number of possible actions; size of output layer       
        :param gamma: 
                RL discount factor for future rewards (Bellman's return) 
        :param xxxLearningRate: 
                The learning rate for the gradient descent in the DQN; 
                corresponds more or less to the parameter alpha in
                RL controlling the how much the most recent episodes
                contribute to the update of the Q-Table
        :param xxxSoftHardUpdatePace: 
                If xxxSoftHardUpdatePace < 1: a soft update is performed at each 
                local network update
                If xxxSoftHardUpdatePace >= 1: the target network is replaced by 
                the local network after targetdnnUpdatePace steps
        :param dnnUpdatePace:
                Determines after how many state-action steps the local network 
                should  be updated. 
        :param bufferSize: 
                Size of the memory buffer containing the experiences < s, a, r, s’ >     
        
        '''            
        
        self.bufferSize = bufferSize
        self.batchSize = batchSize
        self.batchEpochs = batchEpochs
        
        self.actorSoftHardUpdatePace = actorSoftHardUpdatePace
        self.criticSoftHardUpdatePace = criticSoftHardUpdatePace
        self.dnnUpdatePace = dnnUpdatePace
        self.numberExperiences = 0
        
        self.actorLearningRate = actorLearningRate
        self.criticLearningRate = criticLearningRate
        self.gamma = gamma
        
                
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.seed = seed
        
        if seed == None:
            self.seed = np.random.randint(0,999) 
        
        # Actor Network (w/ Target Network)
        self.actorLocal = Actor(stateSize, actionSize, seed).to(device)
        self.actorTarget = Actor(stateSize, actionSize, seed).to(device)
        self.actorOptimizer = optim.Adam(self.actorLocal.parameters(), lr=actorLearningRate)

        # Critic Network (w/ Target Network)
        self.criticLocal = Critic(stateSize, actionSize, seed).to(device)
        self.criticTarget = Critic(stateSize, actionSize, seed).to(device)
        self.criticOptimizer = optim.Adam(self.criticLocal.parameters(), lr=criticLearningRate, 
                                           weight_decay=weightDecay)

        # Noise process
        self.noise = OUNoise(self.actionSize, self.seed)
 
        self.memory = Memory(bufferSize, batchSize, seed)
    
    def react(self,state,noise=None):
        '''
        Get the action for a given state
        :param state: react to agiven state
        :param sigma: a parameter governing the scale of noise in the Ornstein-Uhlenbeck process. 
                    For sigma=None there is no noise 
                    For sigma=0 there the Ornstein-Uhlenbeck process remains without noise
        '''
        
        #state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        state = torch.from_numpy(state).float().to(device)
        self.actorLocal.eval()
        with torch.no_grad():
            actions = self.actorLocal(state).cpu().data.numpy()
        self.actorLocal.train()
        # check if the Ornstein-Uhlenbeck process is active
        if noise != None:
            actions += self.noise.sample(noise)       
            #actions += sigma * np.array([random.random() for _ in range(len(actions))])
        return np.clip(actions, -1, 1)
    
    
    def experience(self,state, action, reward, next_state, done):
        self.numberExperiences += 1        
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        if self.numberExperiences %  self.dnnUpdatePace == 0:
            self.learn()
        
    
    
    def learn(self):    
        if len(self.memory) <= self.batchSize*self.batchEpochs:
            return
        
        for epoch in range(self.batchEpochs):
            self._learn() 
        
        if self.criticSoftHardUpdatePace < 1:
            self.softTargetUpdate(self.criticTarget,self.criticLocal,self.criticSoftHardUpdatePace)
        elif self.numberExperiences %  ( self.dnnUpdatePace * self.criticSoftHardUpdatePace) == 0: 
            self.targetUpdate(self.criticTarget,self.criticLocal)

            
        if self.actorSoftHardUpdatePace < 1 and \
           self.numberExperiences %  self.dnnUpdatePace == 0: 
            self.softTargetUpdate(self.actorTarget,self.actorLocal,self.actorSoftHardUpdatePace)
        elif self.numberExperiences %  ( self.dnnUpdatePace * self.actorSoftHardUpdatePace) == 0: 
            self.targetUpdate(self.actorTarget,self.actorLocal)
    
    def _learn(self):
        
        states, actions, rewards, nextStates, dones = self.memory.torchSample()
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        nextActions = self.actorTarget(nextStates)
        nextQTargets = self.criticTarget(nextStates, nextActions)
        # Compute Q targets for current states (y_i)
        targetsQ = rewards + (self.gamma * nextQTargets.detach() * (1 - dones))
        # Compute critic loss
        expectedQ = self.criticLocal(states, actions)
        criticLoss = F.mse_loss(expectedQ, targetsQ)
        # Minimize the loss
        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.criticLocal.parameters(), 1)
        self.criticOptimizer.step()
 
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        nextActions = self.actorLocal(states)
        actorLoss = -self.criticLocal(states, nextActions).mean()
        # Minimize the loss
        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        self.actorOptimizer.step()
    
    def softTargetUpdate(self,targetDnn,localDnn,updateRatio):
        for targetParam, localParam in zip(targetDnn.parameters(), localDnn.parameters()):
            targetParam.data.copy_(updateRatio*localParam.data + (1.0-updateRatio)*targetParam.data)

     
    def targetUpdate(self,targetDnn,localDnn):
        
        for targetParam, localParam in zip(targetDnn.parameters(), localDnn.parameters()):
            targetParam.data.copy_(localParam.data)

    
    def reset(self):
        """
        Reset the Ornstein-Uhlenbeck process
        """        
        self.noise.reset()    
        
 
class Memory():
    '''
    Memory buffer containing the experiences < s, a, r, s’ >
    '''    

    def __init__(self,bufferSize, batchSize, seed):
        '''  
        Initialize Memory object
        
        :param bufferSize (int): 
                maximum size of buffer
        :param batchSize (int): 
                size of each training batch
        :param seed (int): 
                random seed
        '''  
        self.memory = deque(maxlen=bufferSize)  
        self.batchSize = batchSize
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
            
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batchSize)        
            
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        nextStates = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None])
  
        return (states, actions, rewards, nextStates, dones)    
    
    def torchSample(self):
        states, actions, rewards, nextStates, dones = self.sample()
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        nextStates = torch.from_numpy(nextStates).float().to(device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(device)
        return (states, actions, rewards, nextStates, dones) 
        
            
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
            
 
class OUNoise:
    """Ornstein-Uhlenbeck process."""
 
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
 
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
 
    def sample(self,sigma=None):
        """Update internal state and return it as a noise sample."""
        if sigma == None:
            sigma = self.sigma
            
        x = self.state
        dx = self.theta * (self.mu - x) + sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state            
            


























