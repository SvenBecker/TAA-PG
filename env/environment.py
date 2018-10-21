import os
import numpy as np
import pandas as pd
from collections import namedtuple
from keras.models import load_model
from sklearn.preprocessing import scale, normalize

import config
from core.optimize import WeightOptimize
from core.basic_logger import get_logger
from core.utils import get_flatten

"""
PortfolioEnv is the main Environment class. DataEnv is being used as some kind of
data source class and the Portfolio class keeps track of the current portfolio.
DataEnv <-> PortfolioEnv <-> Portfolio
"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        # hide TensorFlow warnings


_Step = namedtuple('Step',
                   ['state',
                    'reward',
                    'done',
                    'info'])

_PortfolioInfo = namedtuple('Portfolio',
                            ['weights',
                             'old_weights',
                             'new_weights',
                             'init_weights',
                             'asset_returns',
                             'portfolio_return',
                             'predictions',
                             'portfolio_value',
                             'new_portfolio_value',
                             'old_portfolio_value',
                             'portfolio_variance',
                             'sharpe_ratio',
                             'transaction_costs'])


def env_step(state,
             reward,
             done,
             **kwargs):
    """
    Args:
        :param state: (object) state representation for the agent
        :param reward: (float) step reward
        :param done: (bool) true if episode has finished
        :param kwargs: additional info
    :return: namedtuple object
    """
    return _Step(state, reward, done, kwargs)


def portfolio_info(weights=None,
                   old_weights=None,
                   new_weights=None,
                   init_weights=None,
                   asset_returns=None,
                   predictions=None,
                   portfolio_value=0,
                   new_portfolio_value=0,
                   old_portfolio_value=0,
                   portfolio_return=0,
                   portfolio_variance=0,
                   sharpe_ratio=0,
                   transaction_costs=0):
    """
    collects some portfolio data for easier access and better overview
    """
    return _PortfolioInfo(weights,
                          old_weights,
                          new_weights,
                          init_weights,
                          asset_returns,
                          portfolio_return,
                          predictions,
                          portfolio_value,
                          new_portfolio_value,
                          old_portfolio_value,
                          portfolio_variance,
                          sharpe_ratio,
                          transaction_costs)


class DataEnv:
    # provides data for the portfolio environment class.
    def __init__(
            self,
            data,
            assets,
            scaler=None,
            window_size=100,
            horizon=20,
            predictor=None,
            standardize=True,
            random_starts=True,
            episodes=100,
            epochs=500
    ):
        # see PortfolioEnv class for explanation on parameters
        self.data = np.array(data)
        self.length = data.shape[0]             # data sequence length
        self.features = self.data.shape[1]      # asset returns + additional features
        self.asset_names = assets               # list of asset names
        self.window_size = window_size          # number of past observations for further calculations
        self.horizon = horizon                  # sequence length of each episode (fixed)
        self.nb_assets = len(assets)
        self.scaler = scaler                    # import input scaler
        self.standardize = standardize
        self.predictor = load_model(predictor)  # load the predictor model
        self.episodes = episodes
        self.episode = 0
        self.epoch = 0
        self.episode_start = None
        self.episode_starts = []
        self.random_starts = random_starts
        if self.random_starts:
            self._start_array = np.array([np.random.permutation(np.arange(0, self.episodes))
                                          for _ in range(epochs)])

        self._DataInfo = namedtuple('DataInfo', 'state window asset_returns mean variance '
                                                'correlation covariance prediction')

    def __str__(self):
        return str(self.__class__.__name__)

    def data_info(self, state, window, asset_returns, mean, variance, correlation, covariance, prediction):
        return self._DataInfo(state, window, asset_returns, mean, variance, correlation, covariance, prediction)

    def get_window(self, episode_step=0):
        _start = self.episode_start - self.window_size + episode_step
        _end = self.episode_start + episode_step
        window = self.data[_start: _end]

        asset_window = window[:, 0:len(self.asset_names)]

        if self.scaler is not None:
            data_state, prediction = self._get_state(self.scaler.transform(window),
                                                     np.mean(asset_window, axis=0),
                                                     np.var(asset_window, axis=0),
                                                     np.corrcoef(asset_window.T))
        else:
            data_state, prediction = self._get_state(scale(window, axis=0),
                                                     np.mean(asset_window, axis=0),
                                                     np.var(asset_window, axis=0),
                                                     np.corrcoef(asset_window.T))
        '''
        State = (weights, scaled_mean, scaled_variance, scaled_predictions, correlation_matrix)
        Weights are being included in the PortfolioEnv class, 
        because the DataEnv doesn't need any weight info 
        '''

        info = self.data_info(
            data_state,
            window,
            asset_window,
            np.mean(asset_window, axis=0),
            np.var(asset_window, axis=0),
            np.corrcoef(asset_window.T),
            np.cov(asset_window.T),
            prediction
        )
        return info

    def _set_episode_start(self):
        # select an episode start at random
        if self.random_starts:
            if len(self.episode_starts) <= self.episodes:
                episode_start = np.random.randint(low=self.window_size,
                                                  high=self.length - self.horizon)
                self.episode_starts.append(episode_start)

            else:
                episode_starts = self.episode_starts[1:]
                episode_start = episode_starts[self._start_array[self.epoch][self.episode]]
                self.episode += 1
                if self.episode == self.episodes:
                    self.episode = 0
                    self.epoch += 1

            # returns the same episode starts in a random order for each epoch
            return episode_start
        else:
            # returns the same episode starts in the same order for each epoch
            return np.random.randint(low=self.window_size,
                                     high=self.length - self.horizon)

    def _get_state(self, window, mean, var, corr):
        """
        Args:
            :param window: (object) price history [time_frame x features]
            :param mean: (list) asset mean returns
            :param var: (list) asset variance
            :param corr: (object) asset correlation matrix
        :return: (object) state without portfolio weights
        """
        # Data for the prediction model has to be shape (window_size, features, 1)
        reshaped_window = np.reshape(window, (1, window.shape[0], window.shape[1]))
        prediction = self.predictor.predict_on_batch(reshaped_window)

        if self.standardize:
            # values are being standardized
            obs = np.concatenate((scale(mean, axis=0),                  # scale mean returns
                                  scale(var, axis=0),                   # scale asset return variance
                                  scale(prediction[0])))                # scale predictions
        else:
            # values are being normalized
            obs = np.concatenate((normalize(mean.reshape(-1, 1), axis=0).reshape(1, len(mean))[0],
                                  normalize(var.reshape(-1, 1), axis=0).reshape(1, len(var))[0],
                                  normalize(prediction[0].reshape(-1, 1).reshape(1, len(prediction[0])))[0]
                                  ))

        obs = obs.reshape(3, self.nb_assets)                            # reshape the state
        state = np.concatenate((obs, corr))                             # not the final state (no portfolio weights)
        return state, prediction

    def reset(self):
        # has to be called after one episode has finished
        self.episode_start = self._set_episode_start()


class Portfolio:
    """
    Provides some useful utility for basic portfolio calculations.
    Has to be called for each new episode.
    """
    def __init__(
            self,
            portfolio_value=100,
            risk_aversion=1,
            cost_selling=0,
            cost_buying=0,
            fix_cost=0,
            action_type='signal_softmax'
    ):
        # see PortfolioEnv class for explanation on parameters

        self.init_portfolio_value = portfolio_value
        self.portfolio_value = portfolio_value
        self.cost_selling = cost_selling
        self.cost_buying = cost_buying
        self.fix_cost = fix_cost
        self.action_type = action_type
        self.risk_aversion = risk_aversion
        self.weights = None
        self.new_weights = None
        self.covariance = None
        self.portfolio_return = 0
        self.cost = 0
        self.variance = 0
        self.sharpe = 0

    def __str__(self):
        return str(self.__class__.__name__)

    def update(self, actions):
        # update portfolio weights based on given actions
        self.new_weights = self._set_weights(actions)

        # calculate weight difference
        _weight_diff = np.array(self.new_weights) - np.array(self.weights)

        # estimate costs for trading
        self.cost = self._get_cost(_weight_diff)

        # update portfolio value
        self.portfolio_value = self.portfolio_value - self.cost

        return self.new_weights, self.cost, self.portfolio_value

    def get_next_step(self, asset_returns, covariance):
        """
        Args:
            :param asset_returns: (list) asset returns price_t / price_{t-1} - 1
            :param covariance: (object) variance-covariance matrix based on current window
        :return: reward, weights, portfolio value
        """
        # step forward and get new window
        self.covariance = covariance

        # get new weights
        self.weights = self._get_weights(asset_returns)

        # calculate portfolio return
        self.portfolio_return = np.dot(self.new_weights, asset_returns)

        # calculate new portfolio variance based on updated window
        self.variance = self._get_variance()

        # get reward based on reward function
        step_reward = self._get_reward()

        # update portfolio value
        self.portfolio_value = self.portfolio_value * (1 + self.portfolio_return)

        # calculate sharpe ratio
        self.sharpe = self._sharpe_ratio()

        return step_reward, self.weights, self.portfolio_value

    def reset(self, weights, covariance):
        # resets the portfolio value and the asset weights to their initial value
        self.portfolio_value = self.init_portfolio_value
        self.weights = weights
        self.covariance = covariance
        self.variance = self._get_variance()
        self.sharpe = self._sharpe_ratio()

    def _set_weights(self, actions):
        # change the portfolio weights based on the given actions.

        if self.action_type == 'signal':
            # new w_i = w_i * a_i / sum[i:n; w_i * a_i]
            s = np.dot(self.weights, actions)
            if s == 0:
                # all actions are probably zero -> numerical issue
                return self.weights
            else:
                return [w * a / s
                        for w, a in zip(actions, self.weights)]

        elif self.action_type == 'signal_softmax':
            # new w_i = exp(w_i * a_i) / sum[i:n; exp(w_i * a_i)]
            s = sum([np.e**(w * a)
                     for w, a in zip(self.weights, actions)])

            return [np.e**(w * a) / s
                    for w, a in zip(actions, self.weights)]

        elif self.action_type == 'direct':
            # new w_i = a_i -> new weight = action
            if sum(actions) != 1:
                raise ValueError('Action values have to sum up to 1')
            for a in actions:
                if a < 0 or a > 1:
                    raise ValueError('Action values have to be between 0 and 1')
            return actions

        elif self.action_type == 'direct_softmax':
            # softmax function -> new weight = softmax(action)
            s = sum([np.e ** a
                     for a in actions])

            return [(np.e ** a) / s
                    for a in actions]

        elif self.action_type == 'clipping':
            # clips the weights
            a = [max(0, min(actions[0], 1))]
            for i in range(len(actions) - 2):
                a.append(max(0, min(actions[i + 1], 1 - sum(a))))
            a.append(1 - sum(a))
            return a

    def _get_weights(self, asset_returns):
        # change of portfolio weights given possible deviations in asset returns.
        return [wi * (1 + ri) / sum([w * (1 + r) for w, r in zip(self.new_weights, asset_returns)])
                for wi, ri in zip(self.new_weights, asset_returns)]

    def _get_cost(self, weight_diff):
        # cost for trading based on trading volume
        cost = 0
        sum_weight_diff = sum(abs(weight_diff))     # sum over absolute weight difference
        if round(sum_weight_diff, 5) != 0:
            cost += self.fix_cost   # add fix costs if fix cost are given and weights were modified
            if self.cost_selling == self.cost_buying:
                cost += sum_weight_diff * self.portfolio_value * self.cost_buying
            else:
                for diff in weight_diff:
                    if diff >= 0:
                        cost += self.cost_buying * diff * self.portfolio_value
                    else:
                        cost -= self.cost_selling * diff * self.portfolio_value
        return cost

    def _get_reward(self):
        # returns the perceived reward (utility) of the portfolio.
        return self.portfolio_return - self.risk_aversion / 2 * self.variance - \
               self.cost / self.portfolio_value

    def _sharpe_ratio(self, risk_free=0.0):
        # reward-to-Variability-Ratio (R_p - R_f) / sigma_R_p
        # risk free rate on daily returns can be assumed to be zero
        return (self.portfolio_value / self.init_portfolio_value - 1 - risk_free) \
                / (np.sqrt(self.variance))

    def _get_variance(self):
        # returns the portfolio variance
        return np.dot(np.dot(np.array(self.weights).T, self.covariance), self.weights)


class PortfolioEnv(object):
    """
    Creates a portfolio environment, similar to the TensorForce or Gym environment:
    Gym: https://github.com/openai/gym/blob/522c2c532293399920743265d9bc761ed18eadb3/gym/core.py
    TensorForce: https://github.com/reinforceio/tensorforce/blob/master/tensorforce/environments/environment.py
    """
    def __init__(
            self,
            data,
            assets=config.ASSETS,
            nb_assets=config.NB_ASSETS,
            horizon=config.HORIZON,
            action_space='unbounded',
            window_size=config.WINDOW_SIZE,
            portfolio_value=config.PORTFOLIO_VALUE,
            risk_aversion=config.RISK_AVERSION,
            num_actions=11,
            cost_buying=config.COST_BUYING,
            cost_selling=config.COST_SELLING,
            cost_fix=config.COST_FIX,
            predictor=config.PREDICTION_MODEL,
            optimized=True,
            action_type='signal_softmax',
            scaler=None,
            standardize=True,
            action_bounds=(0.0, 1.0),
            discrete_states=False,
            state_labels=(50,),
            episodes=config.EPISODES,
            epochs=config.EPOCHS,
            random_starts=True,
    ):

        """
        Args:
            :param data: (object) environment data
            :param assets: (list) asset names
            :param horizon: (int) investment horizon -> max episode time steps
            :param window_size: (int) sequence length of the data used for doing prediction
                                and estimating variance, covariance etc
            :param portfolio_value: (int or float) initial portfolio value
            :param risk_aversion: (int or float) constant risk aversion of investor/agent
            :param cost_selling, cost_buying: (float) relative cost of selling and buying assets
            :param cost_fix: (float) costs for being allowed to trade on each time step
            :param predictor: (str) h5 file with prediction model (hyper-)parameters
            :param optimized: (bool) for using either optimized or naive weights
            :param action_type: (str) how to change weights based on actions:
                                'signal', 'signal_softmax', 'direct', 'direct_softmax', 'clipping'
            :param action_space: (str) specifies action space -> 'bounded', 'unbounded' or 'discrete'
            :param num_actions: (int) number of possible action values for each action
                                if action space is discrete
            :param scaler: (object) scaler object
            :param standardize: (bool) use normalization or standardization for state scaling
            :param action_bounds: (tuple) upper and lower bound for continuous actions
            :param discrete_states: (bool) true for discrete state space
            :param state_labels: (tuple) number of labels for state discretization per state row
        """
        # build logger
        self.logger = get_logger(filename='tmp/env.log', logger_name='EnvLogger')
        self.data = data
        self.horizon = horizon
        self.window_size = window_size
        self.risk_aversion = risk_aversion
        self.init_portfolio_value = portfolio_value
        self.optimized = optimized
        self.action_type = action_type
        self.action_space = action_space
        self.num_actions = num_actions
        self.action_bounds = action_bounds
        self.discrete_states = discrete_states
        self.state_labels = state_labels
        self.standardize = standardize,
        self.episodes = episodes
        self.epochs = epochs

        self.step = 0
        self.episode = 0

        try:
            if assets is None:
                self.asset_names = ['A' + str(i + 1) for i in range(nb_assets)]
            else:
                self.asset_names = assets
            if nb_assets is None:
                self.nb_assets = len(self.asset_names)
            else:
                self.nb_assets = nb_assets
            if len(self.asset_names) != nb_assets:
                self.asset_names = ['A' + str(i + 1) for i in range(nb_assets)]
        except Exception as e:
            self.logger.error(e)

        # build portfolio object
        self.portfolio = Portfolio(
            portfolio_value=self.init_portfolio_value,
            risk_aversion=risk_aversion,
            fix_cost=cost_fix,
            cost_selling=cost_selling,
            cost_buying=cost_buying,
            action_type=action_type)

        # build data object
        self.data_env = DataEnv(
            data,
            assets,
            horizon=horizon,
            window_size=window_size,
            scaler=scaler,
            predictor=predictor,
            standardize=standardize,
            random_starts=random_starts,
            episodes=episodes,
            epochs=epochs
        )

        # reset epoch
        self.reset_epoch()

    def __str__(self):
        return str(self.__class__.__name__)

    def _step(self, action):

        # update current portfolio based on agent action
        new_weights, cost, portfolio_value = self.portfolio.update(action)
        self.episode_costs += cost

        # see PortfolioEnv.execute() for explanation
        info = self.data_env.get_window(episode_step=self.step)
        self._update(info)

        # make a step forward
        reward, weights, new_portfolio_value = self.portfolio.get_next_step(self.asset_returns[-1], self.covariance)

        self.episode_reward += reward

        info = portfolio_info(weights=new_weights,
                              old_weights=self.weights,
                              new_weights=weights,
                              init_weights=self.init_weights,
                              asset_returns=self.asset_returns[-1],
                              predictions=self.prediction,
                              portfolio_value=portfolio_value,
                              new_portfolio_value=new_portfolio_value,
                              old_portfolio_value=self.portfolio_value,
                              portfolio_return=new_portfolio_value / portfolio_value - 1,
                              portfolio_variance=self.portfolio.variance,
                              sharpe_ratio=self.portfolio.sharpe,
                              transaction_costs=cost)

        # update portfolio values
        self.weights = new_weights
        self.portfolio_value = new_portfolio_value
        self.portfolio_variance = self.portfolio.variance

        # true if episode has finished
        done = bool(self.step >= self.horizon)

        # update state
        self.state = np.concatenate((np.reshape(self.weights, (1, self.nb_assets)), self.__state), axis=0)

        # discretize the state array if selected
        if self.discrete_states:
            state = self._state_discretization()
        else:
            state = self.state

        # flatten the state array and reduce state size -> returns 1d array
        state = get_flatten(state)

        self.step += 1

        # should not be happening when using a runner
        if self.step > self.horizon:
            self.reset()

        return env_step(state, reward, done, info=info)

    def _reset(self):

        # is being called after finishing an episode -> get a new episode
        self.episode += 1
        self.step = 0
        self.episode_costs = 0
        self.episode_reward = 0

        # get new episode start -> get new data observations
        self.data_env.reset()
        info = self.data_env.get_window()

        # update some variables for easier access
        self._update(info)

        # get a new episode start
        self.entry_point = self.data_env.episode_start

        self.logger.debug(f'{20 * "#"} Episode {self.episode} {50 * "#"}')
        self.logger.debug(f'Episode entry point:{self.entry_point}')

        # estimate initial weights for the assets (optimized or naive)
        self.init_weights = self._get_init_weights(self.asset_returns, self.covariance, optimized=self.optimized)
        self.weights = self.init_weights

        # reset the portfolio
        self.portfolio.reset(covariance=self.covariance, weights=self.weights)
        self.portfolio_value = self.portfolio.init_portfolio_value
        self.portfolio_variance = self.portfolio.variance

        self.logger.debug(f'Initial weights for episode {self.episode}:\n{self.weights}')

        # returns the initial episode environment state
        self.state = np.concatenate((np.reshape(self.weights, (1, self.nb_assets)), self.__state), axis=0)

        # discretize the state array if selected
        if self.discrete_states:
            state = self._state_discretization()
        else:
            state = self.state

        # flatten the array -> returns a 1d array
        state = get_flatten(state)
        return state

    def _get_init_weights(self, asset_returns, covariance, optimized=True):
        """
        Args:
            :param asset_returns: (object) past asset returns
            :param covariance: (object) covariance matrix
            :param optimized: (bool) for doing optimization
        :return: semi-optimized weights based on past mean variance or naive 1/n weights
        """
        return WeightOptimize(asset_returns, covariance, nb_assets=self.nb_assets,
                              risk_aversion=self.risk_aversion).optimize_weights(method='SLSQP') \
            if optimized else [1/self.nb_assets for _ in range(self.nb_assets)]

    def _update(self, info):
        self.__state = info.state
        self.window = info.window
        self.asset_mean = info.mean
        self.asset_variance = info.variance
        self.asset_returns = info.asset_returns
        self.correlation = info.correlation
        self.covariance = info.covariance
        self.prediction = info.prediction

    def close(self):
        """
        Close environment. No other method calls possible afterwards.
        Currently not implemented.
        """
        pass

    def reset_epoch(self):
        # is called at the start for each new epoch
        self.episode = 0
        return self._reset()

    def reset(self):
        # is called after an episode has finished
        self.logger.debug(f'Episode {self.episode} has finished.'
                          f'\nCumulative Reward:{self.episode_reward}'
                          f'\nPortfolio Value: {self.portfolio_value}'
                          f'\nPortfolio Return: {self.portfolio.portfolio_return}'
                          f'\nPortfolio Variance: {self.portfolio.variance}'
                          f'\nSharpe Ration: {self.portfolio.sharpe}\n')
        return self._reset()

    def execute(self, action):
        """
        Args:
            :param action: (list) agent actions to execute
        """
        return self._step(action)

    def seed(self, seed=None):
        return np.random.seed(seed)

    def _state_discretization(self, nb_labels=(50,)):
        if len(nb_labels) == 1:
            nb_labels = [nb_labels[0] for _ in range(self.state.shape[0])]

        # quantil based state discretization
        # TODO: some issues with tensorforce agent NNs (expects float32 gets int32...), maybe wait for an update
        _state = np.array([pd.qcut(self.state[i], nb_labels[i], labels=False, duplicates='drop')
                           for i in range(self.state.shape[0])], dtype='int32')
        return _state

    def env_spec(self):
        # returns a dict of environment configurations
        return dict(
            data_shape=self.data.shape,
            epochs=self.epochs,
            episodes=self.episodes,
            horizon=self.horizon,
            window_size=self.window_size,
            portfolio_value=self.init_portfolio_value,
            risk_aversion=self.risk_aversion,
            optimized_weights=self.optimized,
            action_type=self.action_type,
            action_space=self.action_space,
            discrete_states=self.discrete_states,
            standardize=self.standardize,
            num_actions=self.num_actions
        )

    @property
    def states(self):
        """
        Return the state space.
        Returns: dict of state properties (shape and type).
        => weight + scaled mean + scaled variance + scaled predictions + correlation
        """

        if self.discrete_states:
            # returns discrete state space shape
            return dict(shape=(0.5 * self.nb_assets * (self.nb_assets + 7),), type='int')
        else:
            # returns continuous state space shape
            return dict(shape=(0.5 * self.nb_assets * (self.nb_assets + 7),), type='float')

    @property
    def actions(self):
        """
        Return the action space.
        Returns: dict of action properties (continuous, number of actions)
        """
        # discrete action space -> categorical distribution
        if self.action_space == 'discrete':
            return dict(shape=(self.nb_assets,), num_actions=self.num_actions, type='int')

        # continuous action space with upper and lower bounds -> beta distribution
        elif self.action_space == 'bounded':
            return dict(shape=(self.nb_assets,), type='float',
                        min_value=self.action_bounds[0], max_value=self.action_bounds[1])

        # unbounded continuous action space (default) -> gaussian distribution
        else:
            return dict(shape=(self.nb_assets,), type='float')


def pretty_print_state(array, assets=config.ASSETS):
    """
    Args:
        :param array: (object) state array
        :param assets: (list) asset names
    :return: state as pandas data frame object
    """
    index = ['Weights', 'Scaled Mean', 'Scaled Variance', 'Scaled Predicted Return'] + assets
    return pd.DataFrame(array, index=index, columns=assets)


if __name__ == '__main__':
    '''
    Just to see if the environment is working as intended
    '''
    df = pd.read_csv(config.ENV_DATA, sep=r';|,', engine='python')
    df = df.drop(df.columns[0], axis=1)
    env = PortfolioEnv(df, action_type='signal_softmax', discrete_states=True, action_space='unbounded')
    print(env.actions)
    env.reset_epoch()
    for step in range(200):
        agent_action = np.random.rand(10)
        result = env.execute(agent_action)
        print('\nStep', step + 1)
        print(pretty_print_state(env.state))
