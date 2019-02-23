import argparse
import os
import json
import sys
sys.path.append("../")

from collections import namedtuple
from tensorforce.agents import Agent

import config
from core.utils import PrepData
from core.basic_logger import get_logger
from env.environment import PortfolioEnv


_Defaults = namedtuple('Defaults', [
    'data',
    'split',
    'agent_config',
    'net_config',
    'episodes',
    'horizon',
    'action_type',
    'action_space',
    'num_actions',
    'eval_path',
    'verbose',
    'load_agent',
    'basic_agent',
    'discrete_states',
    'standardize_state'
])


def get_defaults():
    try:
        return _Defaults(
            data=config.ENV_DATA,
            split=config.TRAIN_SPLIT,
            agent_config=os.path.join(config.AGENT_CONFIG, 'ppo_sb.json'),
            net_config=os.path.join(config.NET_CONFIG, 'mlp3.json'),
            episodes=config.EPISODES,
            horizon=config.HORIZON,
            action_type='signal_softmax',
            action_space='bounded',
            num_actions=41,
            eval_path=os.path.join(config.RUN_DIR, 'test'),
            verbose=2,
            load_agent=os.path.join(config.MODEL_DIR, 'saves', 'PPOAgent'),
            basic_agent=None,
            discrete_states=True,
            standardize_state=False
        )

    except Exception as e:
        print(e)

        # in case of problems dealing with the config file or paths
        return _Defaults(
            data=None,
            split=0.75,
            agent_config=None,
            net_config=None,
            episodes=50,
            horizon=20,
            action_type='signal_softmax',
            action_space='bounded',
            num_actions=11,
            eval_path=None,
            verbose=2,
            load_agent=None,
            basic_agent=None,
            discrete_states=False,
            standardize_state=True
        )


def get_args(default):

    # Args: namedtuple with default values
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data', help="Environment data file",
                        default=default.data)

    parser.add_argument('-sp', '--split', help="Train test split",
                        default=default.split)

    parser.add_argument('-a', '--agent-config', help="Agent configuration file",
                        default=default.agent_config)

    parser.add_argument('-n', '--net-config', help="Network specification file",
                        default=default.net_config)

    parser.add_argument('-at', '--action-type', help="How to change weights",
                        default=default.action_type)

    parser.add_argument('-ap', '--action-space', help="Action space continues or discrete",
                        default=default.action_space)

    parser.add_argument('-na', '--num_actions', help="Number of discrete actions", type=int,
                        default=default.num_actions)

    parser.add_argument('-e', '--episodes', type=int, help="Number of episodes per epoch",
                        default=default.episodes)

    parser.add_argument('-hz', '--horizon', type=int, help="Investment horizon",
                        default=default.horizon)

    parser.add_argument('-ep', '--eval-path', help="Save agent to this dir",
                        default=default.eval_path)

    parser.add_argument('-v', '--verbose', help="Console printing level", type=int,
                        default=default.verbose)

    parser.add_argument('-l', '--load-agent', help="Load agent from this dir",
                        default=default.load_agent)

    parser.add_argument('-ba', '--basic-agent', help="Load basic agent",
                        default=default.basic_agent)

    parser.add_argument('-ds', '--discrete-states', help="Discrete state space true/false",
                        default=default.discrete_states)

    parser.add_argument('-ss', '--standardize-state', help="Standardize or normalize state true/false",
                        default=default.standardize_state)

    return parser.parse_args()


class TestAgent(object):

    def __init__(self, info):
        self.args = info
        if not os.path.isdir("tmp"):
            try:
                os.mkdir("tmp", 0o755)
            except OSError:
                raise OSError("Cannot create directory `tmp`")
        self.logger = get_logger(filename='tmp/test.log', logger_name='TestLogger')
        self.logger.debug(self.args)

        self.test, self.scaler = self._get_data()

        # build environment
        self.environment = PortfolioEnv(
            self.test,
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
            standardize=self.args.standardize_state,
            episodes=self.args.episodes,
            epochs=1,
            random_starts=False
        )

        # check if there is a valid path for saving the evaluation files
        if self.args.eval_path:
            save_dir = os.path.dirname(self.args.eval_path)
            if not os.path.isdir(save_dir):
                try:
                    os.mkdir(save_dir, 0o755)
                except OSError:
                    raise OSError("Cannot save evaluation to dir {} ()".format(save_dir))

        # build agent, either basic or rl agent
        if self.args.basic_agent is None:

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
                pass

            self.agent = Agent.from_spec(
                 spec=self.agent_config,
                 kwargs=dict(
                    states_spec=self.environment.states,
                    actions_spec=self.environment.actions,
                    network_spec=self.network_spec
                 )
            )

            # try to load a pre trained agent
            if self.args.load_agent:
                load_dir = os.path.dirname(self.args.load_agent)
                if not os.path.isdir(load_dir):
                    raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
                self.agent.restore_model(directory=self.args.load_agent)

        else:
            if self.args.basic_agent == 'random':
                from model.basic_agents import RandomActionAgent
                self.agent = RandomActionAgent(action_shape=(config.NB_ASSETS,))
            else:
                from model.basic_agents import BuyAndHoldAgent
                self.agent = BuyAndHoldAgent(action_shape=(config.NB_ASSETS,))

    def _get_data(self):
        prep = PrepData(horizon=self.args.horizon,
                        window_size=config.WINDOW_SIZE,
                        nb_assets=config.NB_ASSETS,
                        split=self.args.split)
        data = prep.get_data(self.args.data)                # get data
        test = data[int(self.args.split * data.shape[0]):]      # test data split
        scaler = prep.get_scaler(data[0: int(args.split * data.shape[0])])
        return test, scaler

    def run(self):
        try:
            from run.runner import Runner
        except ModuleNotFoundError:
            from runner import Runner

        run_test = Runner(
            self.agent,
            self.environment,
            epochs=1,
            episodes=self.args.episodes,
            horizon=self.args.horizon,
            mode='test',
            verbose=self.args.verbose,
            model_path=None,
            seed=92
            )

        run_test.run()
        run_test.close(
            save_full=self.args.eval_path + '/full_' + str(self.agent) + '_evaluation.csv',
            save_short=self.args.eval_path + '/short_' + str(self.agent) + '_evaluation.csv'
        )


if __name__ == '__main__':
    # get default parameters
    defaults = get_defaults()

    # get arguments
    # they are equal to the default ones if not given by flags
    args = get_args(defaults)

    # run test
    run = TestAgent(args)
    run.run()
