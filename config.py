import datetime
import os

# portfolio parameters
ASSETS = ['IWD', 'IWF', 'IWO', 'IWN', 'EFA', 'IEMG', 'TIP', 'GOVT', 'GLD', 'VNQ']   # asset symbols
START_DATE = datetime.date(2013, 1, 1)  # start data for historical data
END_DATE = datetime.date(2018, 2, 12)   # end date for historical data
PORTFOLIO_VALUE = 1000                  # initial portfolio value
RISK_AVERSION = 1                       # constant rate of relative risk aversion
NB_ASSETS = len(ASSETS)                 # number of assets

# environment parameters
WINDOW_SIZE = 100       # length of historical data provided on each time step
COST_SELLING = 0.0025   # (transaction-)costs for selling assets based on trading volume
COST_BUYING = 0.0025    # (transaction-)costs for buying assets based on trading volume
COST_FIX = 0            # fix costs for trading (absolute)
TRAIN_SPLIT = 0.75      # train/test data split

# run parameters
EPOCHS = 200            # number of max episodes runs
EPISODES = 100         # number of episodes on each epoch
HORIZON = 20            # max number of time steps for each episode

# directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(ROOT_DIR, 'run')
RUN_TMP = os.path.join(RUN_DIR, 'tmp')
MODEL_DIR = os.path.join(ROOT_DIR, 'model')
AGENT_CONFIG = os.path.join(MODEL_DIR, 'agent_config')
NET_CONFIG = os.path.join(MODEL_DIR, 'net_config')
ENV_DIR = os.path.join(ROOT_DIR, 'env')
DATA_DIR = os.path.join(ENV_DIR, 'data')
DATABASE_DIR = os.path.join(DATA_DIR, 'database')
IMAGE_DIR = os.path.join(ROOT_DIR, 'img')

# file names
ENV_DATA = DATA_DIR + '/environment.csv'
PREDICTION_MODEL = ENV_DIR + '/prediction_model.h5'
