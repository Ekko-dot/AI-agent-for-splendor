# AI Method 3 - Deep Q-Learning - Yuexiang He

# Table of Contents
  * [Motivation](#motivation)
  * [Application](#application)
  * [Trade-offs](#trade-offs)     
     - [Advantages](#advantages)
     - [Disadvantages](#disadvantages)
  * [Challenges](#challenges)
  * [Future improvements](#future-improvements)



### Motivation  

Deep Q-Learning(DQN) is an extension of the Reinforced learning(RL) algorithm Q-Learning that integrates a deep neural network into the agent. Instead of the traditional Q-matrix, DQN uses a neural network that takes only the state as input, and the output is the Q-value of each action for the current state.

Using DQN to implement agents for splendor has the following advantages:

1. Splendor contains multiple resources (gems/cards/nobles), which leads to the complexity of its state space and action space, and DQN is able to fit this high-dimensional state space and action space effectively through deep neural networks.
2. DQN requires less predefined and complex game strategies and gameplay. This is because DQN can learn strategies to cope with different game phases on their own through interaction with the game environment, reducing the introduction of subjective biases.
3. DQN can enhance the learning process through a variety of techniques. For example, the experience replay technique can store past game experiences and sample them for multiple training sessions, thus breaking the correlation between samples. From these experiences, model can also learn different ways to cope with different game play situations. Reward-driven techniques can train the network by learning how to maximize the agent's score, which is consistent with splendor's win condition.
4.  Based on the above features and techniques, DQN has strong adaptability and generalization ability, i.e., it can gradually adjust its own strategy to meet the challenges in the face of different opponents.

[Back to top](#table-of-contents)

### Application  

1. Neural network structure

| Layers | Input     | Output     |
| ------ | --------- | ---------- |
| Layer1 | state_dim | 512        |
| Layer2 | 512       | 768        |
| Layer3 | 768       | 1024       |
| Layer4 | 1024      | action_dim |

Networks use MLPs with activation functions because MLPs are simple, universal and generalized, and also have sufficient expressive power. The large number of parameters provided by MLPs and the ability to fit non-linear functions give a theoretically make them well suited for splendor, which don't involve textual sequences and vision.

| Other components | Parameter |
| ---------------- | --------- |
| Dropout          | P=0.4     |
| Batch size       | 32        |
| Learning rate    | 0.0000001 |
| Mult step        | 20        |
| Episode          | 5000      |

The activation function uses leakyrelu, because comparing the results of leakyrelu and relud shows that leakyrelu is more effective. Dropout is used, p=0.4 Dropout is introduced to prevent overfitting. Optimizer chosen Adam.

2. Calculate Reward Function

Reward mechanisms can largely influence the strategy of model learning. the reward mechanisms are mainly about the following aspects:

- Encourage score for positive actions: The positive actions including getting gems, cards (tier1, 2, 3), and nobles. These cumulative achievements ultimately lead to winning score, so reasonable rewards will be given bonus points based on the number of rounds played.
-  Penalize resource wastage: an increase or no change in the number of gems but no increase in the score indicates that the resources may not be reasonably utilized and converted into scores. Adding a penalty for wasted resources can push the model focus on scheduling resources. It is also important to prevent it from falling into a cycle of taking gems and returning them without buying the cards.
- Penalize the increase in the number of rounds: With the increase in the number of rounds, the player's resources continue to accumulate, and the one-time-get score will increase, so the model is expected to win the game as quickly as possible.If the number of rounds exceeds a certain level need to make a penalty on the current score. In addition the same state change in different rounds will change the number of rewards and even the attributes of the rewards and penalties.

3. Experience Replay

Setting up experience replayers plays a very important role in DQN. On the one hand, experience replay breaks the correlation of samples in certain degree. if the model just selects consecutive states as samples during the learning process,what model will learn have strong relationship with time. Therefore, experience replay is used to record previous states as samples and enable the model to sample and learn from them, thus increasing the variance of the training process and contributing to more stable learning. On the other hand, the samples in the experience replayer covered historical data for a wide range of situations, and agents would perform additional learning by utilizing the historical data, which allows model learn strategy from history and improves data utilization.

4. Multi Step

Multi-step learning is an mechanism that extends standard one-step learning to multiple future time steps, then comparing the current state of this step of with the state after multiple steps. Multi-step learning is often used when a task requires planning multiple actions or when the consequences of an action are delayed in nature. For the implement, the accumulation of rewards from n-steps is considered before updating the strategy and value function. Since in splendor, the impact of the decision in this step may not be realized until after multiple steps, it was decided to introduce this strategy.

5. Epsilon greedy + Burn in 

Epsilon-greedy is a intuitive way to weigh the strategies of exploration and exploitation in the learning process. In this situation, exploration means taking an executable action at random.  Only by trying out the possibilities, the model can only learn the whole picture of its environment . Exploitation, on the other hand, uses the output of the network to select the action that is self-perceived to be the most available, i.e., the one with the largest q-value and legal. The strategy weighs the two choices through the parameter Epsilon. Specifically the probability of epsilon selects exploration and the probability of 1-Epsilon selects utilization. This value decays with STEP.

The Burn in mechanism is chosen because a certain amount of initial data is required before training a deep learning model. The Burn-in period is the phase in which the model performs a series of exploratory actions before any significant learning begins, with the aim of accumulating initial experience and filling the experience replay buffer. This collects data from many different states and actions, which contributes to the generalization ability during model training and reduces the risk of overfitting.

| Parameter     | Value        |
| ------------- | ------------ |
| Start Epsilon | 1.0          |
| End Epsilon   | 0.01         |
| Epsilon decay | 0.995        |
| Burn-in       | 50000 (step) |

6. DDQN + soft update

Double Deep Q Network(DDQN) is an improved version of DQN , which aims to solve the problem of overestimation that exists in DQN. In this part, the main reason for the introduction of DDQN is to counteract the rapid increase in the loss value due to gradient explosion. The results show that the introduction of DDQN improves this situation to a large extent. Specifically, DDQN mitigates this problem by using two networks: a current network for selecting the best action and a target network for evaluating the value of that action. This separation results in slower network updates and smoother gradient changes. The risk of overestimation is reduced because maximizing the action and evaluating the value are no longer done by the same network.

7. Confrontational training

Confrontational training means that another agent is introduced to simulate a real sparring situation instead of playing the game alone. The introduction of such a training mode can simulate a real matchmaking situation and reduces the fact that playing the game alone has too many actions to choose from (especially different combinations of gem collecting and gem returning actions), which possibly leads to the model being stuck in a dead loop of constantly collecting gems but not scoring at the beginning of the training period. This method uses two models of adversarial training, one against a random agent and one self-confrontational .

[Back to top](#table-of-contents)

### Trade-offs  
#### *Advantages*  

1. The introduction of DDQN and softupdate strategy largely overcomes the non-convergence of the model due to the increasing loss value brought about by the gradient explosion, and it can be seen from the image that the loss value is in a decreasing trend after the addition of DDQN and softupdate.
2. Setting the scoring function in a detailed way gives some guidance to the model. For example, when I set up the scoring function without setting the bonus points for the increase in the number of noble, according to the replay log, agent almost never gets a noble. However, with the addition of the noble scoring mechanism, the agent regularly gets celebrities when the number of rounds is sufficient.
3. The introduction of the confrontation training mechanism can avoid the agent from being trapped in a dead cycle. And the result of self-confrontation is better than that of confrontation with random agent. Meanwhile, after the introduction of the confrontation training mechanism, the game can easily come to the late stage since the resources can be consumed quickly , which ensures that the model learns the coping strategies in the late stage of the game.
4. The sufficient number of parameters in MLP provides the possibility to learn the complex resource scheduling of the splendor and to learn the corresponding strategies to deal with various complex situations. At the same time, the method of adjusting parameters and preventing overfitting in deep learning area can also be applied.

#### *Disadvantages*

1. The detailed setting of the scoring function is likely to be too subjective and restrictive, leading to negative effects for the model learning strategy.
2. The introduction of burn in requires more episodes, because during the period of burn in the model is in the stage of expanding and not actually generating strategies. Therefore, after the introduction of 50,000 steps of burn in, the episodes need to be increased from 1,000 to 3,000 in order for the model to actually train strategies.
3. The basic experience replayer just samples randomly, and does not focus on training low-quality samples, i.e., samples with high TD error, which results in the model being easily affected by low-quality samples, wasting training cycles and decreasing the learning effect. This drawback can be optimized by applying prioritized experience replay.

[Back to top](#table-of-contents)

### Challenges

1. Overcoming the gradient explosion problem

   During the training of DQN, I encountered the problem of gradient explosion and it was very serious. the Loss values increased exponentially in the early stage of training. It makes model difficult to learn useful information from the data. I overcame the gradient explosion problem by following methods:

   1. Data batch standardization: firstly, the encoded STATE needs to be normalized for mean and variance, which can significantly reduce the problem of covariate bias. Second, in the calculation of expected q value need to use the reward, current value (i.e., output), so it is necessary to normalize the reward matrix. Otherwise the batch of data in different distributions will lead to unstable calculation results.

   2. Tuning parameters: reducing the learning rate can reduce the magnitude of the weight update to avoid too large update within a step. Meanwhile, appropriately increasing the batch size usually provides smoother and more accurate gradient estimation, but in this experiment the batch size of 64 does not perform as well as 32, which may be due to the relatively low learning rate used in this experiment (0.00001 - 0.000001)

   3. Resizing the network: a network structure that is too large or too small may cause training to crash. A network that is too deep may cause the gradient to grow rapidly during propagation when the gradient is propagated for cumulative multiplication. However, a network that is too small has too few parameters to cope with generating complex game strategies.

   4. Introducing target model (DDQN): the introduction of target network can reduce the drastic fluctuation of Q-value updating, which can mitigate or prevent the gradient explosion during the training process.

      After performing the above steps of playing, the phenomenon of gradient explosion is finally eliminated more completely. In particular, the effect is greatly improved after the introduction of DDQN.

2. In splendor, score rewards are scarce and unbalanced distributed. Score growth tends to be low in the early stages of the game, while in the later stages it is faster and more critical. This leads to a lack of guidance in the early stages of model training if the model only uses player scores as rewards, which makes it impossible to train an effective strategy to transition to the later stages of the game. Therefore, the reward strategy designed for this experiment is shown in Figs:

3. 

4. |                                                              | Opening (0-12) | Inning (12-25) | Ending (26+) |
   | ------------------------------------------------------------ | -------------- | -------------- | ------------ |
   | Gain a tier 1 card                                           | 20             | 10             | 5            |
   | Gain a tier 2 card                                           | 10             | 25             | 15           |
   | Gain a tier 3 card                                           | 5              | 20             | 70           |
   | Number of gems in next turn - Number of gems in current turn | -n             | -2n            | -5n          |

   |                  | entire episode |
   | ---------------- | -------------- |
   | Get a noble      | 50             |
   | Earned 15 scores | 300            |
   | Steps over 40    | -100           |
   | Steps over 100   | -300           |
   | Earned n scores  | 15             |

In designing the reward strategy, it is necessary to analyze the game mechanism. First of all, the whole game can be divided into three periods: in the early part of the game, players are encouraged to acquire limited low-level cards to accumulate bounces, so the score of the first-tier cards is rewarded the highest in the first 12 rounds, and in the late part of the game, rounds should not be wasted to buy first-tier cards with no scores (except for the sake of the noble, but this part of the bonus scores is reflected in the rewards of the noble). In the mid-game, it is necessary to accumulate as many scores as possible because when a certain number of points have been accumulated by the end of the game it is possible to quickly accumulate points to end the game by acquiring Tier 3 cards and celebrities. Therefore the rewards of Level 2 cards are greatest in the mid-game. Tier 3 cards are the primary collection target towards the end of the game, and since obtaining four points in one turn should be more rewarding than obtaining four points in four turns (and thus opponents are likely to obtain more points in four turns), the rewards for tertiary cards are far greater.

The penalties for gem counts are designed to encourage aggressive scheduling of resources by the model, and since each turn becomes more valuable as the number of turns increases, every move to collect gems is risky since it does not result in a direct gain in points. The penalty for rounds is designed to motivate the model to win in a shorter period of time, and most agents can end the game in less than 40 rounds (humans will end in about 27 rounds). Rounds over a hundred end the game and give a huge penalty. Winning the game with 15 points ends the episode and gives a huge bonus!

[Back to top](#table-of-contents)

### Future improvements  

1. Using prioritized experience playback 

Prioritized Experience Replay (PER) is a method to improve the experience replay mechanism in deep reinforcement learning. PER prioritizes “more valuable” experiences to be used more frequently for training by introducing prioritization, which is mainly based on the Temporal Difference Error (TDE). In the currently implemented method, the experience replay are fed from the replay buffer to the playback buffer, and the experience replay are fed from the playback buffer to the playback buffer.

In the currently implemented methods, the experience replay randomly select experiences from the playback buffer for learning, which means that each experience has an equal chance of being resampled. This makes the training process not efficient, or could be more efficient. Therefore, the introduction of PER theoretically allows the model to focus on learning samples with high TD error, correcting the estimation of these state-action pairs and thus accelerating the overall learning. It also avoids repetitive sampling leading to overlearning of the model for only a few minority experiences.

2. Adjust network size and hyperparameters

The decrease in loss after 5000 episode iterations is not significant in terms of loss variation. There are again many original factors that lead to this result, among which the network size and hyperparameters cannot be ignored when they affect the model performance. An oversized network means more parameters for a stronger fit, but it is also prone to gradient explosion, model underfitting, and other problems that can cause the model to crash. A network that is too small may overfit over a large number of training rounds. For hyperparameters, in addition to learning rate, batch size, etc., which are common in deep learning, parameters such as discount factor (gamma), epsilon, empirical playback buffer size, multi step size, etc., can be optimized in DQN. In addition, the choice of optimizer, activation function, and loss function may also have an impact on the performance of the model.

[Back to top](#table-of-contents)
