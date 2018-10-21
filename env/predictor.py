import time
import os
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import RMSprop, Adam, Adadelta, Adagrad, SGD
from keras.callbacks import EarlyStopping

from core.utils import PrepData
from core.basic_logger import get_logger
"""
For building the neural nets the deep learning library Keras on top of TensorFlow is being used:
Keras documentation: https://keras.io
(Hyper-)parameters can be selected at the bottom of this file.
This parameters can further be used to build the final prediction model (see: env/predictor_network.py)
"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # hide messy TensorFlow warnings

logger = get_logger(filename='tmp/prediction.log', logger_name='PredLogger')


def build_model(
        layer,
        dropout,
        rnn_type='LSTM',
        optimizer='rmsprop',
        activation='sigmoid',
        nb_preds=10):
    """
    Args:
        :param layer: (list) containing units for each layer
        :param dropout: (list) containing dropout rate for each layer (same length as layer)
        :param rnn_type: (str) recurrent net type, only LSTM and GRU
        :param optimizer: (str) gradient optimization algorithm, 'sgd', 'adam', 'rmsprop'
        :param activation: (str) activation function, 'sigmoid', 'tanh', 'relu'
        :param nb_preds: (int) number of prediction = number of asset predictions
    :return: prediction model (object)
    """

    model = Sequential()

    # build lstm network
    if rnn_type == 'LSTM':
        model.add(LSTM(units=layer[0],
                       input_shape=(100, 25),
                       return_sequences=True,
                       activation=activation))
        model.add(Dropout(dropout[0]))

        # add additional lstm layers
        for l, d in zip(layer[1:-1], dropout[1:-1]):
            model.add(LSTM(
                units=l,
                return_sequences=True,
                activation=activation))
            model.add(Dropout(d))
        model.add(LSTM(
            units=layer[-1],
            return_sequences=False,
            activation=activation))

    # build gru network (for some reason currently not working on windows os)
    elif rnn_type == 'GRU':
        model.add(GRU(units=layer[0],
                      input_shape=(100, 25),
                      return_sequences=True,
                      activation=activation))
        model.add(Dropout(dropout[0]))

        # add additional gru layers
        for l, d in zip(layer[1:-1], dropout[1:-1]):
            model.add(GRU(
                units=l,
                return_sequences=True,
                activation=activation))
            model.add(Dropout(d))
        model.add(GRU(
            units=layer[-1],
            return_sequences=False,
            activation=activation))
    else:
        logger.error(f'Model type {rnn_type} not implemented')

    model.add(Dropout(dropout[-1]))

    # linear activation for regression
    model.add(Dense(units=nb_preds, activation='linear'))
    # model.add(Activation('linear'))

    # compile model
    model.compile(loss="mse", optimizer=optimizer, metrics=['mae', 'mse'])

    return model


def evaluation_table(
        score=None,
        layer=None,
        dropout=None,
        start=None,
        rnn_type=None,
        batch_size=None,
        epochs=None,
        activation=None,
        optimizer=None,
        learning_rate=None,
        decay=None,
        epsilon=None
        ):

    # parameters which are being shown in the evaluation file
    return {
        'Type': rnn_type,
        'Activation': activation,
        'Optimizer': optimizer,
        'Learning Rate': learning_rate,
        'Decay Rate': decay,
        'Batch Size': batch_size,
        'Epochs': epochs,
        'Layer Units': layer,
        'Dropout Rates': dropout,
        'Time': time.time() - start,
        'Epsilon': epsilon,
        'MAE': score[1],
        'MSE': score[2],
    }


def save_evaluation(
        evaluation,
        name='output',
        sort_by='MAE',
        json=False,
        csv=True):
    """
    Args:
        :param evaluation: (object) evaluation file
        :param name: (str) output filename
        :param sort_by: (str) metrics for sorting ('MAE', 'MAE')
        :param json: (bool) write json file
        :param csv: (bool) write csv file
    """
    evaluation = pd.DataFrame(evaluation).sort_values(by=sort_by)
    try:
        if json:
            # write json file
            evaluation.to_json(name + '.json')
            logger.info(f'File has been saved to {name + ".json"}')
        if csv:
            # write csv file
            evaluation.to_csv(name + '.csv', sep=';')
            logger.info(f'File has been saved to {name + ".csv"}')

    except Exception as e:
        logger.error(e)

    return evaluation


def parameter_dict(nb_runs=None,
                   optimizer=None,
                   activation=None,
                   batch_size=None,
                   nb_layer=None,
                   nn_type=None,
                   epochs=None,
                   layer_units=None,
                   dropout_rates=None,
                   learning_rate=None,
                   decay=None,
                   epsilon=None
                   ):
    """
    Args:
        :param nb_runs: (int) number of runs for grid search
        :param optimizer: (tuple) gradient descent algorithm ('adam', 'rmsprop')
        :param activation: (tuple)
        :param batch_size: (tuple) high and low batch size value (10,20)
        :param nb_layer: (tuple) high and low number of layers (3,6)
        :param nn_type: (tuple) rnn type ('LSTM', 'GRU')
        :param epochs: (tuple) high and low number of epochs (10,15)
        :param layer_units: (tuple) number of cells on each layer (16, 32, 64)
        :param dropout_rates: (tuple) high low dropout rates (1, 10) <- has to be between 0 and 10
        :param learning_rate: (tuple) optimization learning rate (1e-2, 1e-3)
        :param decay: (tuple) decay rates (0.0, 0.01)
        :param epsilon: (tuple) epsilon values
    :return: (dict) (hyper-)parameter values
    """
    return dict(nb_runs=nb_runs,
                optimizer=optimizer,
                activation=activation,
                batch_size=batch_size,
                nb_layer=nb_layer,
                nn_type=nn_type,
                epochs=epochs,
                layer_units=layer_units,
                dropout_rates=dropout_rates,
                learning_rate=learning_rate,
                decay=decay,
                epsilon=epsilon)


def RandomizedSearch(
        param_dict,
        eval_name='eval',
        data_file='data/environment.csv',
        window_size=100,
        split=0.75,
        horizon=10,
        nb_assets=10,
        log_return=True,
        verbose=True,
        sort_by='MAE'):

    logger.info(f'Evaluation on {param_dict["nb_runs"]} runs has started')
    evaluation = []
    nb_model = 1

    verbose = 1 if verbose else 0

    prep = PrepData(
        horizon=horizon,
        window_size=window_size,
        split=split,
        nb_assets=nb_assets,
        log_return=log_return)
    data, _ = prep.run_default(data_file)
    trainX, trainY, testX, testY = data

    logger.info(f'Train Data Shape: X:{trainX.shape}, Y:{trainY.shape}')
    logger.info(f'Test Data Shape: X:{testX.shape}, Y:{testY.shape}')

    if verbose:
        print(f'\nTrain Data Shape: X:{trainX.shape}, Y:{trainY.shape}')
        print(f'Test Data Shape: X:{testX.shape}, Y:{testY.shape}\n')

    try:
        for _ in range(param_dict['nb_runs']):

            # randomize parameters
            nn_type = np.random.choice(param_dict['nn_type'], 1)[0]
            learning_rate = np.random.choice(param_dict['learning_rate'], 1)[0]
            decay = np.random.choice(param_dict['decay'], 1)[0]
            nb_layers = np.random.randint(low=param_dict['nb_layer'][0],
                                          high=param_dict['nb_layer'][1])
            batch = np.random.randint(low=param_dict['batch_size'][0],
                                      high=param_dict['batch_size'][1])
            epoch = np.random.randint(low=param_dict['epochs'][0], high=param_dict['epochs'][1])
            layer_units = tuple(np.random.choice(param_dict['layer_units'], nb_layers))
            dropout_rates = tuple(np.random.randint(low=param_dict['dropout_rates'][0],
                                                    high=param_dict['dropout_rates'][1], size=nb_layers) / 10)
            activation = np.random.choice(param_dict['activation'], 1)[0]
            optimizer_type = np.random.choice(param_dict['optimizer'], 1)[0]
            epsilon = np.random.choice(param_dict['epsilon'], 1)[0]

            if optimizer_type == 'sgd':
                optimizer = SGD(lr=learning_rate,
                                nesterov=True,
                                momentum=0.9)
                epsilon = 0
                decay = 0

            elif optimizer_type == 'adam':
                optimizer = Adam(lr=learning_rate,
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 epsilon=epsilon,
                                 decay=decay)

            elif optimizer_type == 'adagrad':
                optimizer = Adagrad(lr=learning_rate,
                                    epsilon=epsilon,
                                    decay=decay)

            elif optimizer_type == 'adadelta':
                optimizer = Adadelta(lr=learning_rate,
                                     rho=0.95,
                                     epsilon=epsilon,
                                     decay=decay)

            else:
                optimizer = RMSprop(lr=learning_rate,
                                    rho=0.9,
                                    epsilon=epsilon,
                                    decay=decay)

            start = time.time()

            # add early stopping
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0)

            try:
                # build and compile model
                model = build_model(layer_units,
                                    dropout_rates,
                                    rnn_type=nn_type,
                                    optimizer=optimizer,
                                    activation=activation,
                                    nb_preds=nb_assets)
            except Exception as e:
                logger.error(e)

            try:
                # train model
                model.fit(trainX,
                          trainY,
                          batch_size=batch,
                          epochs=epoch,
                          validation_split=0.1,
                          verbose=verbose,
                          callbacks=[early_stopping])
            except Exception as e:
                logger.error(e)

            # evaluate model
            score = model.evaluate(testX, testY, batch_size=batch, verbose=verbose)

            evaluation.append(
                evaluation_table(
                    score=score,
                    layer=layer_units,
                    dropout=dropout_rates,
                    start=start,
                    rnn_type=nn_type,
                    batch_size=batch,
                    epochs=epoch,
                    activation=activation,
                    optimizer=optimizer_type,
                    learning_rate=learning_rate,
                    decay=decay,
                    epsilon=epsilon
                ))
            print('Model ', nb_model, evaluation[-1])
            logger.info(evaluation[-1])
            nb_model += 1

        # save the evaluation file to estimate best (hyper-)parameter combinations
        evaluation = save_evaluation(evaluation, name=eval_name, sort_by=sort_by)
        logger.info(f'Models on this run sorted by {sort_by}\n{evaluation}')
        print(f'\n{77 * "#"}\nModels on this run sorted by {sort_by}:\n{evaluation}')
    except Exception as e:
        logger.error(e)


def full_eval(head=5, sort_by='MAE'):
    ls = []
    try:
        tmp = os.path.join(os.getcwd(), 'tmp') + '/'
        # check tmp directory for all evaluation files
        for file in os.listdir(tmp):
            if file.endswith('.csv') and not file.startswith('Full'):
                ls.append(pd.read_csv(tmp + file, sep=';', index_col=0))

        df = pd.DataFrame(pd.concat(ls, axis=0))

        # sort models by lowest MAE or MSE
        df = df.sort_values(by=sort_by)
        df.to_csv(tmp + 'FullEval.csv', sep=';')
        logger.info('Evaluation file has been updated.')
        print(f'\n{77 * "#"}\n {df.head(head)}')
        # save the configurations of the best model
        df.iloc[0].to_json(tmp + '/predictor_spec.json')
        logger.info(f'Best model has been saved to {tmp + "predictor_spec.json"}')
        return df

    except Exception as e:
        logger.warning(e)


if __name__ == '__main__':

    # build parameter dict
    params = parameter_dict(nb_runs=50,
                            optimizer=('adam', 'rmsprop'),
                            activation=('sigmoid', 'relu'),
                            batch_size=(24, 32),
                            nb_layer=(4, 5),
                            nn_type=('LSTM', 'LSTM'),                    # gru implementation is only working on linux
                            epsilon=(1e-8, 1e-7, 1e-6, 1e-5),
                            epochs=(4, 10),
                            layer_units=(24, 32, 40, 48, 56, 64, 74, 96, 114, 128),
                            dropout_rates=(1, 7),                       # <- [0.1, 0.2, 0.3, 0.4, 0.5]
                            learning_rate=(1e-2, 1e-2),
                            decay=(0.1, 0.1))

    start = time.time()

    # run randomized search to find good hyperparameters for the prediction model
    RandomizedSearch(
        params,
        verbose=False,
        eval_name=os.path.join(os.getcwd(), 'tmp') + '/' + time.strftime("%d_%m_%Y_%H_%M", time.gmtime()))

    logger.info(f'Computation time: {time.time()  - start}')

    # print the best 10 models of all runs based on mae and save the best model parameters
    full_eval(head=10, sort_by='MAE')
