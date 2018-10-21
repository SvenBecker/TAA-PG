🌍
*[English](README.md) ∙ [German](README_de.md)*

# Reinforcement Learning for Tactical Asset Allocation

This project contains the training and testing of multiple reinforcement learning agents given a portfolio management environment.
> The interaction between environment and agent is given by: \
> Environment <> Runner <> Agent <> Model

## Important files and folders

- Data aquisation: [data.py](env/data/data.py)
- Environment: [environment.py](env/environment.py)
- Hyperparametertuning of the prediction model: [predictor.py](/env/predictor.py)
- Training of the prediction model: [predictor_network.py](env/predictor_network.py)
- Runner: [runner.py](run/runner.py)
- Basic configuration parameters: [config.py](config.py)
- Training: [train.py](run/train.py)
- Testing: [test.py](run/test.py)
- Agent configurations: [agent_config](model/agent_config/)
- Neural net configurations: [net_config](model/net_config/) 

### Dependencies

The proposed implementation was done mainly in python, therefore python version >=3.6 is required.
Furthermore following additional python packages are required:

- h5py==2.7.1
- Keras==2.1.3
- matplotlib==2.1.0
- numpy==1.14.1
- pandas==0.20.3
- pandas-datareader==0.5.0
- scikit-learn==0.19.1
- scipy==1.0.0
- seaborn==0.8.1
- tensorflow==1.4.0
- tensorflow-tensorboard==0.4.0rc3
- tensorforce==0.3.5.1

## Running the train file

For agent training you should run the file [train.py](run/train.py) using the console. \
For example:
```
python ~/path/to/file/run/train.py -at "clipping" -v 1
```

Changes to the environment and/or run parameters can be selected through the [train file](run/train.py),
[config file](config.py) and through pre specified flags shown below.

Modifications of the [agents](model/agent_config) and [models](model/net_config) must be specified through the belonging
config file (json format).

### Flags:

| Flag 1 | Flag 2 | Meaning |
|:----:|:----:|-----------|
| -d | --data | path of the environment.csv file |
| -sp | --split | train/test split |
| -th | --threaded | (bool) threaded runner oder single runner |
| -ac | --agent-config | path to the agent config file |
| -nw | --num-worker | number of threads if threaded is being selected |
| -ep | --epochs | number of epochs |
| -e | --episodes | number of episodes |
| -hz | --horizon | investment horizon |
| -at | --action-type | action typ: 'signal', 'signal_softmax', 'direct', 'direct_softmax', 'clipping' |
| -as | --action-space | action space: 'unbounded', 'bounded', 'discrete' |
| -na | --num-actions | number of dicrete actions given a discrete action space |
| -mp | --model-path | saving path of the agent model |
| --eph | --eval-path | saving path of the agent model of the evaluation files |
| -v | --verbose | console verbose level |
| -l | --load-agent | if given agent will be loaded based on a prior save point (path)|
| -ds | --discrete_states | discretization of the state space if true |
| -ss | -standardize-state | (bool) standardization or normalization of the state |
| -rs | --random-starts | (bool) random starts for each new episode |

## Running the test file

The execution of the [test file](run/test.py) is very similar to the one of the train file. 
There has to be a checkpoint in [saves](model/saves) for the selected agent.
```
python ~/path/to/project/run/test.py -l /project/model/saves/AgentName
```

The folder [saved_results](saved_results) contains multiple parameter constellations of pretrained agents. 

### Flags

| Flag | Flag 2 | Meaning |
|:----:|:----:|-----------|
| -d | --data | path of the environment.csv file |
| -ba | --basic-agent | selection of a [BasicAgenten](model/basic_agents.py): 'BuyAndHoldAgent', 'RandomActionAgent' |
| -sp | --split | train/test split |
| -ac | --agent-config | path to the agent config file |
| -e | --episodes | number of episodes |
| -hz | --horizon | investment horizon |
| -at | --action-type | action typ: 'signal', 'signal_softmax', 'direct', 'direct_softmax', 'clipping' |
| -as | --action-space | action space: 'unbounded', 'bounded', 'discrete' |
| -na | --num-actions | number of dicrete actions given a discrete action space |
| --eph | --eval-path | saving path of the agent model of the evaluation files |
| -v | --verbose | console verbose level |
| -l | --load-agent | if given agent will be loaded based on a prior save point (path) |
| -ds | --discrete_states | (bool) discretization of the state space if true |
| -ss | -standardize-state | (bool) standardization or normalization of the state |

## TensorBoard
The files [predictor.py](env/predictor.py) as well as [train.py](run/train.py) integrate
[TensorBoard](https://github.com/tensorflow/tensorboard).
TensorBoard can be loaded by typing
```
tensorboard --logdir path/to/project/env/board
tensorboard --logdir path/to/project/run/board
```

Got to localhost:6006 for the results.

## Credits

- [TensorForce](https://github.com/reinforceio/tensorforce)
