# dqn-self-play-tic-tac-to
# Tic-Tac-Toe DQN

A reinforcement learning project where I trained a DQN to play Tic-Tac-Toe through self-play.

## Objective

The goal was to see whether a DQN could learn how to play Tic-Tac-Toe by playing against itself, without being given a strategy or search algorithm.

The agent learns entirely from the games it plays and the rewards it receives.

## Results

After training, I tested the model against a random player for 1,000 games.

### As X

* Wins: **500 / 500 (100%)**
* Draws: **0**
* Losses: **0**

### As O

* Wins: **418 / 500 (83.6%)**
* Draws: **79 / 500 (15.8%)**
* Losses: **3 (0.6%)**

### Combined

* Wins: **918 / 1000 (91.8%)**
* Draws: **79 / 1000 (7.9%)**
* Losses: **3 / 1000 (0.3%)**

I also played against the trained agent myself and was unable to beat it.

## What I Learned

This project was my first proper look at self-play. What I found most interesting was that the agent was never given a Tic-Tac-Toe strategy — it simply played against itself and learned from those games.

It also gave me more experience with DQN, experience replay, target networks, exploration, and handling invalid actions.

The next step is to move onto more complex games and eventually combine reinforcement learning with search.
