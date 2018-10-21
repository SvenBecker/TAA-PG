import sys
import time
import pandas as pd
import pandas_datareader as web

import config
from core.basic_logger import get_logger

"""
This file is being used to obtain data from yahoo.finance and fred.
"""

logger = get_logger(filename='data.log', logger_name='DataLogger')


class RequestData:
    def __init__(self, symbols, source=None, start=None, end=None, names=None):
        """
        Args:
            :param symbols: (list) containing ticker symbol; ['IWD', 'GLD']
            :param source: data source; 'yahoo', 'fred', 'google'
            :param start: (object) start date object
            :param end: (object) end date object
            :param names: (list) column names for data frame object
        """
        self.symbols = self.names = list(symbols)
        self.source = source
        self.start = start
        self.end = end

        if names is not None:
            self.names = list(names)

        logger.info(f'Collecting data for {self.names} from {self.source}')
        logger.info(f'Time frame:{self.start} - {self.end}')

        self.dataset = self._concat_data()

    def get_data(self, symbol, name):
        start = time.time()
        while True:
            try:
                # start an api request
                if self.source == 'fred':
                    r = web.DataReader(symbol, self.source, self.start, self.end)
                else:
                    r = web.DataReader(symbol, self.source, self.start, self.end)['Adj Close']
                logger.info(f'Data for {name} has been collected')
                print(f'Data for {name} has been collected')
                # break loop if request has been successful
                break
            except Exception as e:
                logger.error(e)

                # sleep for 1 second and restart request
                time.sleep(1)
                if time.time() - start > 300:
                    # if single request takes more than 5 minutes exit process
                    logger.critical(f'Request for {name} has failed')
                    sys.exit()
        return r

    def _concat_data(self):
        # concatenate data
        data = []
        for symbol, name in zip(self.symbols, self.names):
            data.append(self.get_data(symbol, name))
        dataset = pd.DataFrame(pd.concat(data, axis=1))
        dataset.columns = self.names
        logger.info(f'Data shape: {dataset.shape}')
        print(f'\nData shape:{dataset.shape}\n')
        return dataset


if __name__ == '__main__':
    # read database csv files in env/data/database to obtain symbols for the following request
    portfolio_symbols = pd.read_csv('database/portfolio.csv', sep=r',|;', engine='python')['Symbol']
    fred_symbols = pd.read_csv('database/fred.csv', sep=r',|;', engine='python')['Symbol']
    yahoo_symbols = pd.read_csv('database/yahoo.csv', sep=r',|;', engine='python')['Symbol']

    start = time.time()

    portfolio_ds = RequestData(portfolio_symbols, source='yahoo', start=config.START_DATE, end=config.END_DATE).dataset
    yahoo_dataset = RequestData(yahoo_symbols, source='yahoo', start=config.START_DATE, end=config.END_DATE,
                                names=pd.read_csv(config.DATABASE_DIR + '/yahoo.csv', sep=r',|;', engine='python')
                                ['Name']).dataset
    fred_dataset = RequestData(fred_symbols, source='fred', start=config.START_DATE, end=config.END_DATE,
                               names=pd.read_csv(config.DATABASE_DIR + '/fred.csv', sep=r',|;', engine='python')
                               ['Name']).dataset
    logger.info(f'Request time: {time.time() - start}')

    # saves an csv file (is being used to do some visualization)
    portfolio_ds.to_csv('asset_price.csv')

    # for the environment only the asset daily returns are required + drop first row
    portfolio_dataset = portfolio_ds.pct_change(1).dropna()

    # concatenate yahoo and fred data and interpolate missing data
    feature_dataset = pd.DataFrame(pd.concat([yahoo_dataset, fred_dataset], axis=1)).interpolate(method='linear')

    # concatenate asset data and additional data + drop non trading days like weekends
    environment = pd.DataFrame(pd.concat([portfolio_dataset, feature_dataset], axis=1)).dropna()

    # saves the environment data as a csv file
    environment.to_csv(config.ENV_DATA)
    logger.info(f'Done collecting data. Environment shape: {environment.shape}')
    logger.info(f'Data has been saved to {config.ENV_DATA}')
    print(environment.head(5))
