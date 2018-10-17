import numpy as np
import pandas as pd
import os
import time
import threading

from core.basic_logger import get_logger

"""
Runner file:
Environment <-> Runner <-> Agent
Some issues with multithreading -> random seeds are not thread safe.
"""

class BasicRunner(object):
    def __init__(
            self,
            epochs=50,
            episodes=100,
            horizon=20,
            verbose=0,
            model_path=None,
            seed=92
    ):
        """
        Args:
            :param epochs: (int) max epochs
            :param episodes: (int) max episodes
            :param horizon: (int) investment horizon
            :param verbose: (int) console printing level
            :param model_path: (str) path for model saves
            :param seed: (int) seed for random number generator
        """
        self.epochs = epochs
        self.episodes = episodes
        self.horizon = horizon
        self.verbose = verbose
        self.seed = seed
        self.logger = get_logger(filename='tmp/run.log', logger_name='RunLogger')

        if model_path is None:
            self.model_path = os.path.join(os.getcwd(), "models", "saves")
        else:
            self.model_path = model_path

        self.start = None
        self.history = []
        self.global_epoch_reward = -100

    def __str__(self):
        return str(self.__class__.__name__)

    def _run(self):
        # must be overridden in child class
        return NotImplementedError

    def _close(self):
        # must be overridden in child class
        return NotImplementedError

    def episode_finished(self, info, episode=1, thread_id=None):
        """
        Args:
            :param info: (object) list -> [reward, value]
            :param episode: (int) current episode
            :param thread_id: (int) optional thread id
        """
        if self.verbose >= 2:
            if thread_id is not None:
                print(f'Worker {thread_id} finished episode {episode + 1}. '
                      f'Reward: {info[0]}, '
                      f'Portfolio value: {info[1]}, '
                      f'Portfolio return: {info[2]}')
            else:
                print(f'Finished episode {episode + 1}. '
                      f'Reward: {info[0]}, '
                      f'Portfolio value: {info[1]}, '
                      f'Portfolio return: {info[2]}')

    def epoch_finished(self, epoch=1, thread_id=None):
        """
        Args:
            :param epoch: (int) current epoch
            :param thread_id: (int) optional thread id
        """
        if self.verbose < 1:

            if thread_id is not None:
                print(f'Worker {thread_id} finished epoch {epoch + 1}: '
                      f'Average reward: {np.mean(self.history[-1][:, 0], axis=0)}, '
                      f'Average return: {np.mean(self.history[-1][:, 2], axis=0)}, '
                      f'Average value {np.mean(self.history[-1][:, 1], axis=0)}')
            else:
                print(f'Finished epoch {epoch + 1}: '
                      f'Average reward: {np.mean(self.history[-1][:, 0], axis=0)}, '
                      f'Average return: {np.mean(self.history[-1][:, 2], axis=0)}, '
                      f'Average value {np.mean(self.history[-1][:, 1], axis=0)}')
        else:
            columns = ['Episode Reward', 'Portfolio Value',
                       'Portfolio Return', 'Sharpe Ratio',
                       'Portfolio Variance', 'Cumulative Costs']

            df = pd.DataFrame(self.history[-1], columns=columns,
                              index=range(1, self.episodes + 1))

            mean = pd.DataFrame(df.mean(axis=0), columns=['Average'])
            std = pd.DataFrame(df.std(axis=0), columns=['Std Deviation'])
            maximum = pd.DataFrame(df.max(axis=0), columns=['Maximum'])
            minimum = pd.DataFrame(df.min(axis=0), columns=['Minimum'])

            evaluation = pd.DataFrame(pd.concat([mean, std, maximum, minimum], axis=1))

            if thread_id is not None:
                print(f'\nWorker {thread_id} finished epoch {epoch + 1}')
            else:
                print(f'\nFinished epoch {epoch + 1}')

            print(evaluation)
            self.logger.debug(df)
            self.logger.info(f'{77 * "#"}\n{evaluation}')

    def run(self):
        self.start = time.time()
        return self._run()

    def close(self, save_full=None, save_short=None):
        """
        Args
            :param save_full: (str) path for full epoch evaluation file if not None
            :param save_short: (str) path for small epoch evaluation file if not None

        :return: _close() method of child class
        """
        columns = ['Episode Reward', 'Portfolio Value',
                   'Portfolio Return', 'Sharpe Ratio',
                   'Portfolio Variance', 'Cumulative Costs']

        if len(self.history) == 1:
            evaluation = pd.DataFrame(self.history[0], index=range(1, self.episodes + 1),
                                      columns=columns).rename_axis('Episode')
        else:
            evaluation = pd.DataFrame(
                pd.concat([pd.DataFrame(epoch, index=range(1, self.episodes + 1),
                                        columns=columns).rename_axis('Episode') for epoch in self.history],
                          axis=1, keys=['Epoch ' + str(i + 1) for i in range(len(self.history))]))

        self.logger.info(f'{77 * "#"}\n{evaluation}')
        if self.verbose >= 2:
            print('\n', evaluation)

        mean = pd.DataFrame(evaluation.mean(axis=0), columns=['Average'])
        std = pd.DataFrame(evaluation.std(axis=0), columns=['Std Deviation'])
        var = pd.DataFrame(evaluation.var(axis=0), columns=['Variance'])
        maximum = pd.DataFrame(evaluation.max(axis=0), columns=['Maximum'])
        minimum = pd.DataFrame(evaluation.min(axis=0), columns=['Minimum'])

        short = pd.DataFrame(pd.concat([mean, std, var, maximum, minimum], axis=1))

        self.logger.info(f'{77 * "#"}\n{short}')
        print(short)

        # save evaluation files
        if save_full is not None:
            # contains episode rewards etc
            evaluation.to_csv(save_full)

        if save_short is not None:
            # contains epoch averages etc
            short.to_csv(save_short)

        return self._close()


class Runner(BasicRunner):
    def __init__(
            self,
            agent,
            environment,
            epochs=50,
            episodes=100,
            horizon=20,
            mode='train',
            verbose=0,
            model_path=None,
            seed=92
    ):
        """
        Args:
            :param agent: (object) rl or basic agent
            :param environment: (object) portfolio environment for agent
            :param mode: (str) specify run mode -> 'train', 'test'
        """
        super(Runner, self).__init__(
            epochs=epochs,
            episodes=episodes,
            horizon=horizon,
            verbose=verbose,
            model_path=model_path,
            seed=seed
        )

        self.agent = agent
        self.environment = environment
        self.mode = mode
        if self.mode == 'test':
            # for testing only one epoch is required
            self.epochs = 1

        self.model_path = os.path.join(self.model_path, str(self.agent))

    def _run(self):

        for epoch in range(self.epochs):

            # reset the environment random seed -> same episode start points for next epoch
            self.environment.seed(seed=self.seed)

            # after an epoch finished return same episode entry point order
            state = self.environment.reset_epoch()

            reward = []
            value = []
            returns = []
            sharpe = []
            variance = []
            costs = []

            for episode in range(self.episodes):
                # reset the agent
                self.agent.reset()
                # reset the episode reward
                episode_reward = 0

                for step in range(self.horizon):

                    # get next step and results for the taken action
                    if self.mode == 'test':
                        # do action based on observation
                        action = self.agent.act(state, deterministic=True)

                        # for testing no agent observations regarding the reward are required
                        result = self.environment.execute(action)
                    else:
                        # do action based on observation
                        action = self.agent.act(state, deterministic=False)

                        result = self.environment.execute(action)

                        # agent receives new observation and the result of his action(s)
                        self.agent.observe(terminal=result.done, reward=result.reward)

                    # update information regarding the current portfolio
                    info = result.info['info']

                    # increase episode reward by step reward
                    episode_reward += result.reward

                    # update state
                    state = result.state

                # saves the episode results
                reward.append(episode_reward)
                value.append(info.portfolio_value)
                returns.append(value[-1] / self.environment.init_portfolio_value - 1)
                sharpe.append(info.sharpe_ratio)
                variance.append(info.portfolio_variance)
                costs.append(self.environment.episode_costs)

                # does some printing
                self.episode_finished([reward[-1], value[-1], returns[-1]], episode=episode)

                # reset the environment
                if episode < self.episodes - 1:
                    state = self.environment.reset()

            # epoch finished -> if that is the best epoch so far (average reward): save the model
            if np.mean(reward) > self.global_epoch_reward and self.mode != 'test':
                try:
                    self.agent.save_model(directory=os.path.join(self.model_path, str(self.agent)))
                    self.logger.info(f'Agent has been saved to {os.path.join(self.model_path, str(self.agent))}')
                    print('\n> Agent has been saved')
                    self.global_epoch_reward = np.mean(reward)
                except AttributeError:
                    pass
                except FileNotFoundError:
                    if not os.path.isdir(self.model_path):
                        try:
                            os.mkdir(self.model_path, 0o755)
                        except OSError:
                            raise OSError("Cannot save agent to dir {} ()".format(self.model_path))
                    self.agent.save_model(directory=os.path.join(self.model_path, str(self.agent)))
                    self.logger.info(f'Agent has been saved to {os.path.join(self.model_path, str(self.agent))}')
                    print('\n> Agent has been saved')
                    self.global_epoch_reward = np.mean(reward)
                except Exception as e:
                    self.logger.error(e)

            self.history.append(np.array([reward, value, returns, sharpe, variance, costs]).T)

            if self.mode != 'test':
                # some console and logger printing
                self.epoch_finished(epoch=epoch)

        print('Finished run. Time: ', time.time() - self.start)

    def _close(self):
        # close agent and environment
        self.agent.close()
        self.environment.close()


class ThreadedRunner(BasicRunner):
    def __init__(
            self,
            agents,
            environments,
            epochs=50,
            episodes=100,
            horizon=20,
            verbose=0,
            model_path=None,
            seed=92
    ):
        """
        Args:
            :param agents: (object) list of agent objects
            :param environments: (object) list of environment objects
        """
        super(ThreadedRunner, self).__init__(
            epochs=epochs,
            episodes=episodes,
            horizon=horizon,
            verbose=verbose,
            model_path=model_path,
            seed=seed
        )

        if len(agents) != len(environments):
            self.logger.critical(f'Each agent requires its own environment. '
                                 f'{len(agents)} and {len(environments)} were given.')

        self.agents = agents
        self.environments = environments

        self.global_epoch_reward = -100
        self.model_path = os.path.join(self.model_path, str(self.agents[0]))
        if not os.path.isdir(self.model_path):
            try:
                os.mkdir(self.model_path, 0o755)
            except OSError:
                raise OSError("Cannot save agent to dir {} ()".format(self.model_path))

    def _run_thread(self, thread_id, agent, environment):
        """
        Args:
            :param thread_id: (int) worker/thread id
            :param agent: (object) worker agent
            :param environment: (object) worker environment
        """
        from core.utils import get_flatten
        state = get_flatten(environment.state)    # has to be declared because of some threaded error
        for epoch in range(self.epochs):
            environment.seed(seed=self.seed)
            state = environment.reset_epoch()
            reward = []
            value = []
            returns = []
            sharpe = []
            variance = []
            costs = []

            for episode in range(self.episodes):
                # reset worker
                agent.reset()

                # reset cumulative episode reward
                episode_reward = 0

                for step in range(self.horizon):

                    # obtain worker action
                    action = agent.act(state, deterministic=False)

                    # get result
                    result = environment.execute(action)
                    info = result.info['info']

                    episode_reward += result.reward

                    # worker receives result
                    agent.observe(reward=result.reward, terminal=result.done)

                value.append(info.portfolio_value)
                returns.append(value[-1] / environment.init_portfolio_value - 1)
                variance.append(info.portfolio_variance)
                sharpe.append(info.sharpe_ratio)
                reward.append(episode_reward)
                costs.append(environment.episode_costs)

                # some console printing
                self.episode_finished([reward[-1], value[-1], returns[-1]],
                                      episode=episode, thread_id=thread_id)

                # reset the environment -> new episode entry point
                if episode < self.episodes - 1:
                    state = environment.reset()

            # epoch finished -> if that is the best epoch so far (average reward): save the model
            if np.mean(reward) > self.global_epoch_reward:
                self.global_epoch_reward = np.mean(reward)
                try:
                    # saves the model which performed best
                    agent.save_model(directory=os.path.join(self.model_path, str(self.agents[0])))
                    print("\nSaving agent after epoch {}".format(epoch + 1))
                except Exception as e:
                    self.logger.error(e)

            self.history.append(np.array([reward, value, returns, sharpe, variance, costs]).T)

            # some logger and console printing
            self.epoch_finished(epoch=epoch, thread_id=thread_id)

    def _run(self):

        # build threads
        threads = [threading.Thread(target=self._run_thread,
                                    args=(t, self.agents[t], self.environments[t]),
                                    kwargs={})
                   for t in range(len(self.agents))]

        # start threads
        [t.start() for t in threads]

        # join threads
        [t.join() for t in threads]

        print('All threads finished. Time: {}'.format(time.time() - self.start))

    def _close(self):
        # close all agents and environments
        [agent.close() for agent in self.agents]
        [env.close() for env in self.environments]


if __name__ == '__main__':
    """
    Run the Buy and Hold Agent on the environment to see if the runner and environment 
    are correctly working.
    """
    import config
    from model.basic_agents import BuyAndHoldAgent
    from env.environment import PortfolioEnv
    from core.utils import PrepData

    prep = PrepData(horizon=10,
                    window_size=config.WINDOW_SIZE,
                    nb_assets=config.NB_ASSETS,
                    split=config.TRAIN_SPLIT)
    data = prep.get_data(config.ENV_DATA)  # get data file and extract data
    train = data[0:int(config.TRAIN_SPLIT * data.shape[0])]  # train/test split
    scaler = prep.get_scaler(train)
    agent = BuyAndHoldAgent(action_shape=(10,))
    env = PortfolioEnv(data, scaler=scaler, action_type='signal', random_starts=False)
    run = Runner(agent, env, mode='test', verbose=2, episodes=config.EPISODES, epochs=config.EPOCHS)
    run.run()
    run.close(
        save_full=os.path.join(config.RUN_DIR, 'train') + '/full_BuyAndHoldAgent_evaluation.csv',
        save_short=os.path.join(config.RUN_DIR, 'train') + '/short_BuyAndHoldAgent_evaluation.csv')

