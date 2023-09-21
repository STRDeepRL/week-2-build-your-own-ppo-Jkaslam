

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

## Task 3 - Tuning the 🎲 **Exploration & Exploitation Strategies** using Algorithm-Specific Hyperparameters

Having implemented GAE in Task 2, re-run the training command provided below to start agent training. You're encouraged to adjust or introduce additional parameters as required.

```shell
python multigrid/scripts/train_ppo_cleanrl.py --env-id MultiGrid-CompetativeRedBlueDoor-v2-DTDE-Red-Single-with-Obsticle --num-envs 8 --num-steps 128 --learning-rate 3e-4 --total-timesteps 10000000 --exp-name baseline
```

### Deepening Your Understanding to Interpret Your Results
***Q.1*** Train a baseline agent using default or adjusted parameter values. Capture and present Tensorboard screenshots to report the following training metrics. Indicate the `Sample Effiicency`, the number of training timesteps and policy updates, required to achieve the Training Baseline Thresholds:

- **episodic_length**
- **episodic_return**
- **policy_updates**
- **entropy**
- **explained_variance**
- **value_loss**
- **policy_loss**
- **approx_kl**

**CleanRL Agent Training Baseline Thresholds for Your Reference**:
- `episodic_length` should converge to a solution within 40 time steps and maintain for at least 100k time steps at the end of training.
- `episodic_return` should converge to consistently achieve 2.0+ returns, enduring for a minimum of the last 100k time steps.
- `explained_variance` should stabilize at a level above 0.6 for at least the last 100k time steps.
- `entropy` should settle at values below 0.3 for a minimum of 100k final training steps.

### Hands-on Experiences on PPO-Specific Hyperparameters
***Q.2*** If your baseline agent struggles to achieve the Training Baseline Thresholds, or if there's potential for enhancment, now you are getting the chance to fine-tuning the following PPO-specific parameters discussed in class to improve the performance of your agent. You may want to run multiple versions of experinements, so remember to modify `--exp-name` to differentiate between agent configurations. For final submissions, pick the top 3 performing or representable results and present the training metrics via screenshots and specify the number of timesteps and policy updates needed to fulfill or surpass the Training Baseline Thresholds. (Including links to their videos will be ideal)

- **gamma**
- **gae-lambda**
- **clip-coef**
- **clip-vloss**
- **ent-coef**
- **vf-coef**
- **target-kl**

Additionally, consider tweaking the following generic Deep RL hyperparameters:

- **num_envs**
- **batch_size**
- **num_minibatches**
- **minibatch_size**
- **total_timesteps**
- **num_updates**
- **num_steps**
- **update_epochs**


**Tips:**
- Monitor and track your runs using Tensorboard with the following command:
  ```shell
  tensorboard --logdir submission/cleanRL/runs
  ```
    ```shell
  tensorboard --logdir submission/ray_results/1v1_death_match_baseline
  ```


## Task 4: Bring the Lessons Learned from CleanRL to RLlib to solve a 1v1, 🤖 🆚 🤖 Scenario 

As you get familiar with PPO by working through the CleanRL implementation, let's pivot back to RLlib. We'll harness our understanding of hyperparameter tuning to address a 1v1 competition with a pre-trained opponent.

### 🎮 Visualizing the Scenario:

### Starting Training:

### Q.1 Metrics to Report:

As the same as Task 2&3, document the following training metrics, showcasing them with screenshots. Also, detail the number of timesteps and policy updates that meet or exceed the Training Baseline Thresholds.

Here are the RLLib and Scenario specific metrics:

- **episode_len_mean**
- **ray/tune/policy_reward_mean/red_0**
- **ray/tune/policy_reward_mean/blue_0**
- **ray/tune/sampler_results/custom_metrics/red_0/door_open_done_mean**
- **ray/tune/sampler_results/custom_metrics/blue_0/door_open_done_mean**
- **ray/tune/sampler_results/custom_metrics/red_0/eliminated_opponents_done_mean**
- **ray/tune/sampler_results/custom_metrics/blue_0/eliminated_opponents_done_mean**
- **ray/tune/counters/num_agent_steps_trained**
- **ray/tune/counters/num_agent_steps_sampled**
- **ray/tune/counters/num_env_steps_sampled**
- **ray/tune/counters/num_env_steps_trained**
- **episodes_total**
- **episodes_this_iter**
- **red_0/learner_stats/cur_kl_coeff**
- **red_0/learner_stats/entropy**
- **red_0/learner_stats/grad_gnorm**
- **red_0/learner_stats/kl**
- **red_0/learner_stats/policy_loss**
- **red_0/learner_stats/total_loss**
- **red_0/learner_stats/vf_explained_var**
- **red_0/learner_stats/vf_loss**
- **red_0/num_grad_updates_lifetime**



Here are the PPO-specific parameters in RLLib:
- **gamma**
- **lambda_**
- **kl_coeff**
- **kl_target**
- **clip_param**
- **grad_clip**
- **vf_clip_param**
- **vf_loss_coeff**
- **entropy_coeff**

> **Note**: [submission/configs/algorithm_training_config.json](submission/configs/algorithm_training_config.json) is where the training script calling the algorithm specific parameters from. Default values of PG and PPO specific parameters are stored in there as baselines for you.


**RLlib Agent Training Baseline Thresholds for Your Reference**:
- `episode_len_mean` should converge to a solution within 20 time steps and maintain for at least 100k time steps at the end of training.
- `ray/tune/policy_reward_mean/red_0` should converge to consistently achieve 1.3+ returns, enduring for a minimum of the last 100k time steps.
- `explained_variance` should stabilize at a level above 0.4 for at least the last 100k time steps.
- `red_0/learner_stats/entropy` should settle at values below 0.3 for a minimum of 100k final training steps.

**RLlib Agent Behavior Analysis Thresholds**
The following Metrics are Behavior-specific metrics. It depends on how your agent emerges into certain specific behaviors to achieve the RL objective to maximize the discounted sum of rewards from time step t to the end of the game. So, how to achieve the maximum return depends on the training environment's world dynamic and the agent's reward structures. So, the "Player Archetypes" of your agent can be varied. 

Our training scenario can be interpreted as a Zero-Sum game. Therefore, if your agent learned to solve a particular scenario by unlocking the door first, your Red agent should dominate this metric. Vice Versa.
- **ray/tune/sampler_results/custom_metrics/red_0/door_open_done_mean**
- **ray/tune/sampler_results/custom_metrics/blue_0/door_open_done_mean**

As mentioned above, if your agent learned to solve a particular scenario by eliminating the opponent first, your Red agent should dominate this metric. Vice Versa.
- **ray/tune/sampler_results/custom_metrics/red_0/eliminated_opponents_done_mean**
- **ray/tune/sampler_results/custom_metrics/blue_0/eliminated_opponents_done_mean**

{
    "PG_params": {

        "gamma":0.99
        
    },
    "PPO_params": {
        "lambda_" : 1.0,
        "kl_coeff" : 0.2,
        "kl_target" : 0.01,
        "clip_param" : 0.3,
        "grad_clip" : null,
        "vf_clip_param" : 10.0,
        "vf_loss_coeff" : 0.5,            
        "entropy_coeff" : 0.001,
        "sgd_minibatch_size" : 128,
        "num_sgd_iter" : 30
    }
}
For final submitions, pick the top 3 performing or representable results and present the training metrics via screenshots and specify the number of timesteps and policy updates needed to fulfill or surpass the Training Baseline Thresholds


**Tips:**
- Take a look at the configuration of `MultiGrid-CompetativeRedBlueDoor-v3-DTDE-1v1` in  [envs/__init__.py](multigrid/envs/__init__.py)
- You can filter the plots using the following filters:

```
eliminated_opponents_done_mean|episode_len_mean|num_agent_steps_trained|num_agent_steps_sampled|num_env_steps_sampled|num_env_steps_trained|episodes_total|red_0/learner_stats/cur_kl_coeff|red_0/learner_stats/entropy|red_0/learner_stats/grad_gnorm|red_0/learner_stats/kl|red_0/learner_stats/policy_loss|red_0/learner_stats/total_loss|red_0/learner_stats/vf_explained_var|red_0/learner_stats/vf_loss|red_0/num_grad_updates_lifetime|ray/tune/policy_reward_mean/red_0|ray/tune/policy_reward_mean/blue_0
```
- RLlib Tune may report metrics with different names but pointing to the same metric. For example, `ray/tune/sampler_results/custom_metrics/blue_0/door_open_done_mean` is the same as `ray/tune/custom_metrics/blue_0/door_open_done_mean` so just report one is fine.


- To visualize a specific checkpoint, use the following command:
```shell
python multigrid/scripts/visualize.py --env MultiGrid-CompetativeRedBlueDoor-v3-DTDE-1v1  --num-episodes 10  --load-dir submission/ray_results/PPO/PPO_MultiGrid-CompetativeRedBlueDoor-v3-DTDE-1v1_XXXX/checkpoint_YYY/checkpoint-YYY --render-mode human --gif DTDE-1v1-testing
```
##### Replace `XXXX` and `YYY` with the corresponding number of your checkpoint.


- If running on Colab, use the `%tensorboard` [line magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html) to achieve the same; see the [notebook](notebooks/homework1.ipynb) for more details.


## Task 5 - Homework Submission via Github Classroom

### Submission Requirements:

1. **CleanRL Agent**: 
    - Commit and push your best-performing cleanRL agent, ensuring it meets the minimum required thresholds described in the Task, to [submission/cleanRL](submission/cleanRL).
    - For videos, save them to [submission/cleanRL/videos](submission/cleanRL/videos). Please be mindful regarding video size and retain only the most representative ones. Rename the videos as needed for clarity.

2. **RLlib Agents**: 
    - Commit and push your best-performing RLlib agents and checkpoints, ensuring they satisfy the minimum thresholds described in the Task, to [submission/ray_results](submission/ray_results). And also your customized [submission/configs](submission/configs).

3. **RLlib Agents Evaluation Reports**: 
    - Commit and push relevant RLlib agent evaluation results: `<my_experiment>_eval_summary.csv`, `<my_experiment>_episodes_data.csv`, and `<my_experiment>.gif` to [submission/evaluation_reports](submission/evaluation_reports).

4. **Answers to Questions**:
    - For question answers, either:
      - Update the provided [homework2.ipynb](notebooks/homework2.ipynb) notebook, or 
      - Submit a separate `HW2_Answer.md` file under [submission](submission).

5. **MLFlow Artifacts**:
    - Ensure you commit and push the MLFlow artifacts to [submission](submission) (Which should be automatic).


#### Tips:
- Retain only the top-performing checkpoints in [submission/ray_results](submission/ray_results).
    - Refer to the baseline performance thresholds specified for each agent training task.
    - Uploading numerous checkpoints, particularly underperforming ones, may cause the CI/CD to fail silently due to time constraints.
    
- Executing [tests/test_evaluation.py](tests/test_evaluation.py) with `pytest` should generate and push the necessary results to [submission/evaluation_reports](submission/evaluation_reports).

- For an exemplar submission that fulfills all the requirements and successfully passing the Autograding Github Actions, please checkout [Example Submission](https://github.com/STRDeepRL/week-2-build-your-own-ppo-heng4str).

- Always place your submissions within the `submission/` directory. If opting for the notebook approach, please maintain your edited `homework2.ipynb` and related documents under `notebooks/`.

