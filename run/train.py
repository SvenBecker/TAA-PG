import argparse
import os
import json
import sys
sys.path.append("../")

from multiprocessing import cpu_count
from collections import namedtuple
from tensorforce.agents import agents as AgentsDictionary, Agent

import config
from core.utils import PrepData
from core.basic_logger import get_logger
from env.environment import PortfolioEnv


_Defaults = namedtuple('Defaults', [
    'data',
    'split',
    'threaded',
    'agent_config',
    'net_config',
    'num_worker',
    'epochs',
    'episodes',
    'horizon',
    'action_type',
    'action_space',
    'num_actions',
    'model_path',
    'eval_path',
    'verbose',
    'load_agent',
    'discrete_states',
    'standardize_state',
    'random_starts'
])


def get_defaults():
    try:
        return _Defaults(
            data=config.ENV_DATA,
            split=config.TRAIN_SPLIT,
            threaded=False,
            agent_config=os.path.join(config.AGENT_CONFIG, 'ppo_sb.json'),
            net_config=os.path.join(config.NET_CONFIG, 'mlp3.json'),
            num_worker=None,
            epochs=config.EPOCHS,
            episodes=config.EPISODES,
            horizon=config.HORIZON,
            action_type='signal_softmax',
            action_space='bounded',                 # 'discrete', 'bounded', 'unbounded'
            num_actions=41,
            model_path=os.path.join(config.MODEL_DIR, 'saves'),
            eval_path=os.path.join(config.RUN_DIR, 'train'),
            verbose=1,
            load_agent=None,
            discrete_states=False,
            standardize_state=True,
            random_starts=True
            )

    except Exception as e:
        print(e)

        # in case of problems dealing with the config file or paths
        return _Defaults(
            data=None,
            split=0.75,
            threaded=True,
            agent_config=None,
            net_config=None,
            num_worker=None,
            epochs=50,
            episodes=50,
            horizon=20,
            action_type='signal_softmax',
            action_space='bounded',
            num_actions=11,
            model_path=None,
            eval_path=None,
            verbose=2,
            load_agent=None,
            discrete_states=False,
            standardize_state=True,
            random_starts=True
        )


def get_args(default):
    # Args: tuple with default values

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data', help="Environment data file",
                        default=default.data)
    parser.add_argument('-sp', '--split', help="Train test split",
                        default=default.split)
    parser.add_argument('-a', '--agent-config', help="Agent configuration file",
                        default=default.agent_config)
    parser.add_argument('-n', '--net-config', help="Network specification file",
                        default=default.net_config)
    parser.add_argument('-w', '--num-worker', help="Number of threads to run where the model is shared", type=int,
                        default=default.num_worker)
    parser.add_argument('-th', '--threaded', help="Threaded is True",
                        default=default.threaded)
    parser.add_argument('-at', '--action-type', help="How to change weights",
                        default=default.action_type)
    parser.add_argument('-ap', '--action-space', help="Action space continues or discrete",
                        default=default.action_space)
    parser.add_argument('-na', '--num_actions', help="Number of discrete actions", type=int,
                        default=default.num_actions)
    parser.add_argument('-eo', '--epochs', type=int, help="Number of epochs",
                        default=default.epochs)
    parser.add_argument('-e', '--episodes', type=int, help="Number of episodes per epoch",
                        default=default.episodes)
    parser.add_argument('-hz', '--horizon', type=int, help="Investment horizon",
                        default=default.horizon)
    parser.add_argument('-ep', '--eval-path', help="Save agent to this dir",
                        default=default.eval_path)
    parser.add_argument('-m', '--model-path', help="Save agent to this dir",
                        default=default.model_path)
    parser.add_argument('-v', '--verbose', help="Console printing level", type=int,
                        default=default.verbose)
    parser.add_argument('-l', '--load-agent', help="Load agent from this dir",
                        default=default.load_agent)
    parser.add_argument('-ds', '--discrete-states', help="Discrete state space true/false",
                        default=default.discrete_states)
    parser.add_argument('-ss', '--standardize-state', help="Standardize or normalize state true/false",
                        default=default.standardize_state)
    parser.add_argument('-rs', '--random-starts', help="If true returns the same episode starts in a random order",
                        default=default.random_starts)

    return parser.parse_args()


class TrainAgent(object):

    def __init__(self, arguments):
        import config
        self.args = arguments
        self.logger = get_logger(filename='tmp/train.log', logger_name='TrainLogger')
        self.logger.debug(self.args)

        self.train, self.scaler = self._get_data()

        # build environment
        self.environment = PortfolioEnv(
            self.train,
            nb_assets=config.NB_ASSETS,
            horizon=self.args.horizon,
            window_size=config.WINDOW_SIZE,
            portfolio_value=config.PORTFOLIO_VALUE,
            assets=config.ASSETS,
            risk_aversion=config.RISK_AVERSION,
            scaler=self.scaler,
            predictor=config.PREDICTION_MODEL,
            cost_buying=config.COST_BUYING,
            cost_selling=config.COST_SELLING,
            action_type=self.args.action_type,
            action_space=self.args.action_space,
            optimized=True,
            num_actions=self.args.num_actions,
            discrete_states=self.args.discrete_states,
            standardize=self.args.standardize_state,
            episodes=self.args.episodes,
            epochs=self.args.epochs,
            random_starts=self.args.random_starts
        )

        # load agent config
        with open(self.args.agent_config, 'r') as fp:
            self.agent_config = json.load(fp=fp)

        # load network config
        if self.args.net_config:
            with open(self.args.net_config, 'r') as fp:
                self.network_spec = json.load(fp=fp)

        try:
            print(f'Agent spec {self.agent_config}'
                  f'\nNetwork spec {self.network_spec}'
                  f'\nEnvironment spec: {self.environment.env_spec()}\n')

            self.logger.info(f'\nAgent spec: {self.agent_config}'
                             f'\nNetwork spec: {self.network_spec}'
                             f'\nEnvironment spec: {self.environment.env_spec()}\n')
        except Exception:
            # in case of using one of the basic agents
            pass

        # check if the agent can be saved
        if self.args.model_path:
            save_dir = os.path.dirname(self.args.model_path)
            if not os.path.isdir(save_dir):
                try:
                    os.mkdir(save_dir, 0o755)
                except OSError:
                    raise OSError("Cannot save agent to dir {} ()".format(save_dir))

        # check if training evaluation files can be saved
        if self.args.eval_path:
            save_dir = os.path.dirname(self.args.eval_path)
            if not os.path.isdir(save_dir):
                try:
                    os.mkdir(save_dir, 0o755)
                except OSError:
                    raise OSError("Cannot save evaluation to dir {} ()".format(save_dir))

        if self.agent_config['type'] == 'trpo_agent':
            # for some reason trpo + tensorboard does not work
            self.logger.warning('TensorBoard will not be supported')

            # build agent
            self.agent = Agent.from_spec(
                spec=self.agent_config,
                kwargs=dict(
                    states_spec=self.environment.states,
                    actions_spec=self.environment.actions,
                    network_spec=self.network_spec
                )
            )

        else:
            # summary spec for TensorBoard
            self.summary_spec = dict(directory="./board/",
                                     steps=50,
                                     labels=[
                                         'regularization',
                                         'losses',
                                         'variables',
                                         'actions',
                                         'states',
                                         'rewards',
                                         'gradients',
                                         'configuration'
                                     ])

            # build agent
            self.agent = Agent.from_spec(
                spec=self.agent_config,
                kwargs=dict(
                    states_spec=self.environment.states,
                    actions_spec=self.environment.actions,
                    network_spec=self.network_spec,
                    summary_spec=self.summary_spec
                )
            )

        # if there is a pre trained agent -> continue training
        if self.args.load_agent:
            load_dir = os.path.dirname(self.args.load_agent)
            if not os.path.isdir(load_dir):
                raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
            self.agent.restore_model(self.args.load_agent)

    def _get_data(self):
        prep = PrepData(horizon=self.args.horizon,
                        window_size=config.WINDOW_SIZE,
                        nb_assets=config.NB_ASSETS,
                        split=self.args.split)

        # get data file and extract data
        data = prep.get_data(self.args.data)

        # train/test split
        train = data[0:int(self.args.split * data.shape[0])]

        # prediction model scaler based on training data
        scaler = prep.get_scaler(train)

        return train, scaler

    def run_single(self):
        try:
            from run.runner import Runner
        except ModuleNotFoundError:
            from runner import Runner

        run = Runner(
            self.agent,
            self.environment,
            epochs=self.args.epochs,
            episodes=self.args.episodes,
            horizon=self.args.horizon,
            mode='train',
            verbose=self.args.verbose,
            model_path=self.args.model_path,
            seed=92
            )

        run.run()
        run.close(
            save_full=self.args.eval_path + '/full_' + str(self.agent) + '_evaluation.csv',
            save_short=self.args.eval_path + '/short_' + str(self.agent) + '_evaluation.csv'
        )

    def run_threaded(self):
        import config
        from copy import deepcopy
        try:
            from run.runner import ThreadedRunner
        except ModuleNotFoundError:
            from runner import ThreadedRunner

        from model.worker import build_worker

        # if number of worker is not given set the number of worker = cpu cores
        if self.args.num_worker is None:
            num_worker = cpu_count()
        else:
            num_worker = self.args.num_worker

        environments = [self.environment]
        for _ in range(num_worker - 1):
            env = PortfolioEnv(
                self.train,
                nb_assets=config.NB_ASSETS,
                horizon=self.args.horizon,
                window_size=config.WINDOW_SIZE,
                portfolio_value=config.PORTFOLIO_VALUE,
                assets=config.ASSETS,
                risk_aversion=config.RISK_AVERSION,
                scaler=self.scaler,
                predictor=config.PREDICTION_MODEL,
                cost_buying=config.COST_BUYING,
                cost_selling=config.COST_SELLING,
                action_type=self.args.action_type,
                optimized=True,
                num_actions=self.args.num_actions,
                discrete_states=self.args.discrete_states
            )
            environments.append(env)

        # build agents
        agents = [self.agent]

        # copy the configurations for all worker
        agent_configs = []
        for i in range(num_worker):
            worker_config = deepcopy(self.agent_config)
            agent_configs.append(worker_config)

        # build worker
        for i in range(num_worker - 1):
            config = agent_configs[i]
            agent_type = config.pop('type', None)
            worker = build_worker(AgentsDictionary[agent_type])(
                states_spec=environments[0].states,
                actions_spec=environments[0].actions,
                network_spec=self.network_spec,
                model=self.agent.model,
                **config
            )
            agents.append(worker)

        run_threaded = ThreadedRunner(
            agents,
            environments,
            epochs=self.args.epochs,
            episodes=self.args.episodes,
            horizon=self.args.horizon,
            verbose=self.args.verbose,
            model_path=self.args.model_path,
            seed=92
        )

        run_threaded.run()

        run_threaded.close(
            save_full=self.args.eval_path + '/full_evaluation.csv',
            save_short=self.args.eval_path + '/short_evaluation.csv'
        )


if __name__ == '__main__':
    # get default parameters
    defaults = get_defaults()

    # get arguments
    # they are equal to the default ones if not given by flags
    arguments = get_args(defaults)

    run = TrainAgent(arguments)
    if arguments.threaded:
        # run a threaded runner
        run.run_threaded()
    else:
        # run a single runner
        run.run_single()
