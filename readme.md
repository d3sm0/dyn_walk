### PPO baseline

PPO implementation with KAFNETS networks, which are in a nutshell non parametric activation functions using linear combination
of Kernels. The implementation can be found in kafnets. Kernels available are: 'rbf' and 'periodic'.

The kafnet tf implementation can be found here [git](https://gitlab.com/d3sm0/kafnets.git)

The demo is tested on MUJOCO if you don't have it, follow the instruction here [mujoco](https://github.com/openai/mujoco-py)

Most of the parameters can be changed in the config file. If you want to use different activation functions, that has to be
set in worker/agent/act.

To execute the tensorboard
```commandline
tensorboard --logidr log-files/env_name
```
Data can also be analyzed via log.csv file in log-files/env-name/current_date

Repo to check:
- [openai baseline](https://github.com/openai/baselines)

Papers:
- [kafnets](https://arxiv.org/pdf/1707.04035.pdf)
- [trpo](https://arxiv.org/abs/1502.05477)
- [ppo](https://arxiv.org/abs/1707.06347)