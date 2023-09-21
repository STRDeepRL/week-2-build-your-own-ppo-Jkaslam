

---
# Assignment 2: Implement, Tune and Evaluate your Policy Gradient and Actor-Critic Algorithms

## Due Date
- **Due Date:** Thursday, September 21, 6:00 PM


**Q.1** From the command line outputs, can you report the values for the following parameters from the command line outputs? Additionally, please describe the role of each parameter in the training loop and explain how these values influence training in a sentence or two. This exercise can help you grasp the fundamentals of `Sample Efficiency` and understand the tradeoffs when scaling your training process in a parallel fashion.  

- **num_envs**: 4
This is the number of agents that are running in parallel. 
- **batch_size**: 512
The total number of trajectories in the buffer used during a training phase.  
- **num_minibatches**: 4
The number of minibatches to split the buffer into during SGD. 
- **minibatch_size**: 128
The number of trajectories in a minibatch used in SGD. 
- **total_timesteps**: 10000000
The total number of timesteps taken by all agents throughout all trajectories used
to train the agent in PPO. 
- **num_updates**: 19531
The total number of times the policy will be updated if the agent is trained through
all 10^7 training steps. 
- **num_steps**: 128
The length of a trajectory. 
- **update_epochs**: 4
The number of times to split the batch into minibatches and perform SGD. 

Num_envs can be useful if running a single simulation for a task is expensive and
can help speed up training in this case.  As the minibatch size increases the improvements
should get better because we are less likely to get minibatches which are not representative
of standard behavior. But it will take more time to generate larger batches.  

***Q.1*** As mentioned in [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/), PPO employs a streamlined paradigm known as the vectorized architecture. This architecture encompasses two phases within the training loop:

- **Rollout Phase**: During this phase, the agent samples actions for 'N' environments and continues to process them for a designated 'M' number of steps.

- **Learning Phase**: In this phase, fundamentally, the agent learns from the data collected during the rollout phase. This data, with a length of NM, includes 'next_obs' and 'done'.

Utilizing your baseline codebase tagged `v2.1`, please pinpoint the `Rollout Phase` and the `Learning Phase` within the codebase, indicating specific line numbers.

The rollout phase is lines 500-528. The learning phase is 550-717.  

## Task 3 & Task 4

I was unable to reach the baseline metrics for either task 3 or task 4. I spent a lot of time trying to tune hyperparameters. I'm interested to see if other people had success and how they figured it out! I've included a screenshot directory of some of the relevant metrics. 

In task 3 I was able to train multiple agents that fairly consistently stayed between 40 and 60 in episode length. But I could not get anything to consistently stay at around or under 40. I was able to train agents that satisfied the other necessary metrics. 

For task 4 I didn't have as much time to train as many models. But what I did notice was that my models tended to unlearn behavior. I managed to get a few models to hover around an episode length mean of 20 as per the metrics but it would not maintain this for 100k time steps. In fact, I let one of the agents train for significantly more time steps and the average episode length mean grew a lot. 

I was able to get one agent to maintain a policy reward mean in the 1.25 range for a large number of timesteps. But I was unable to consistently stay at 1.3+. I was also unable to get good explained variance but not good entropy results. 


