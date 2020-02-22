## DDQN with PyTorch for OpenAI Gym
Implementation of Double DQN reinforcement learning for OpenAI Gym environments with PyTorch.  
The related paper can be found here: [Hasselt, 2010](https://papers.nips.cc/paper/3964-double-q-learning)

# Double DQN
The standard DQN method has been shown to overestimate the true Q-value, because for the target an argmax over estimated Q-values is used. Therefore when some values are overestimated and some underestimated, the overestimated values have a higher probability to be selected.
Standart DQN target:  
Q*(s<sub>t</sub>)


