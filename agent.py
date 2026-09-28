import torch
import torch.nn as nn
import torch.optim as optim
from dqn import DQN
import random
class Agent:
    lr = 1e-3
    batch_size = 32
    output = 9
    gamma = 0.99

    def __init__(self):
        self.online_net = DQN()
        self.online_net.train()

        self.target_net = DQN()
        self.update_target_net()

        for param in self.target_net.parameters(): param.requires_grad = False

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=self.lr)



    
    def act(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q = self.online_net(state_t)[0]

        q = q.clone()
        for i in range(9):
            if state[i] != 0:
                q[i] = -1e9

        return q.argmax().item()


    def act_e_greedyy(self,state,epsilion=0.01):
        if random.random() < epsilion:
            legal = [i for i in range(9) if state[i] == 0]
            return random.choice(legal)

        else:
            return self.act(state)




    def _sample_batch(self,buffer):
        s, a , r , ns , t = zip(*random.sample(buffer, self.batch_size))
        f,l = torch.FloatTensor, torch.LongTensor

        return f(s), l(a), f(r), f(ns), f(t)


    def train_iter(self,buffer):
        state, action, reward, next_state , terminal = self._sample_batch(buffer)

        q_value = self.online_net(state)[range(self.batch_size), action]

        with torch.no_grad():
            next_state_action = self.online_net(next_state).max(1)[1]
            next_qv = self.target_net(next_state)[range(self.batch_size), next_state_action]
            target_qv = reward - self.gamma * (1 - terminal) * next_qv

        loss = (q_value - target_qv).pow(2).mean()

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        return loss.item()


    def update_target_net(self):
            self.target_net.load_state_dict(self.online_net.state_dict())
    