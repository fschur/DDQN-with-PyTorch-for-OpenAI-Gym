# DDQN with PyTorch for OpenAI Gym
Implementation of Double DQN reinforcement learning for OpenAI Gym environments with discrete action spaces. Performance is defined as the sample efficiency of the algorithm i.e. how good is the average reward after using x episodes of interaction in the environment for training.   
The related paper can be found here: [Hasselt, 2010](https://papers.nips.cc/paper/3964-double-q-learning)

## Double DQN
The standard DQN method has been shown to overestimate the true Q-value, because for the target an argmax over estimated Q-values is used. Therefore when some values are overestimated and some underestimated, the overestimated values have a higher probability to be selected.

**Standard DQN target:**  
Q(s<sub>t</sub>, a<sub>t</sub>) = r<sub>t</sub> + Q(s<sub>t+1</sub>, argmax<sub>a</sub>Q(s<sub>t</sub>, a))  

By using two uncorralated Q-Networks we can prevent this overestimation. In order to save computation time we do gradient updates only for one of the Q-Networks and periodically update the parameters of the target Q-Network to match the parameter of the Q-Network that is updated.  

**The Double DQN target then becomes:**  
Q(s<sub>t</sub>, a<sub>t</sub>) = r<sub>t</sub> + Q<sub>&theta;</sub>(s<sub>t+1</sub>, argmax<sub>a</sub>Q<sub>target</sub>(s<sub>t</sub>, a))  

**And the loss function is given by:**  
(Q(s<sub>t</sub>, a<sub>t</sub>) - Q<sub>&theta;</sub>(s<sub>t</sub>, a<sub>t</sub>))^2  
