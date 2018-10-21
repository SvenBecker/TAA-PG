import re
import pandas as pd
import os
from keras.optimizers import RMSprop, Adam, Adadelta, Adagrad, SGD
from keras.callbacks import TensorBoard, EarlyStopping, TerminateOnNaN

from env.predictor import build_model
from core.utils import PrepData
import config

"""
In case there is no pre trained prediction model, run this file to obtain one.
The (hyper-)parameters have to be given. The ones here presented (down below) worked out
quit nicely. They were evaluated using randomized search (see: rl/env/predictor.py file).
"""


def main():

    # loading and preprocessing data
    prep = PrepData(
        horizon=10,
        window_size=config.WINDOW_SIZE,
        split=config.TRAIN_SPLIT,
        nb_assets=config.NB_ASSETS,
        log_return=True
    )

    data, _ = prep.run_default(config.ENV_DATA)     # data has been scaled already
    trainX, trainY, testX, testY = data

    # load model specs
    predictor_spec = pd.read_json(os.path.join(os.getcwd(), 'tmp') + '/predictor_spec.json',
                                  typ='series')

    print('\nNetwork specs:\n', predictor_spec)

    # add keras callbacks
    # terminate on nan
    terminate_nan = TerminateOnNaN()

    # early stopping
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)

    # add TensorBoard
    tensorboard = TensorBoard(
        log_dir='./board',
        histogram_freq=1,
        batch_size=predictor_spec['Batch Size'])

    if predictor_spec['Optimizer'] == 'sgd':
        optimizer = SGD(lr=predictor_spec['Learning Rate'],
                        nesterov=True)

    elif predictor_spec['Optimizer'] == 'adam':
        # specify adam optimizer
        optimizer = Adam(lr=predictor_spec['Learning Rate'],
                         beta_1=0.9,
                         beta_2=0.999,
                         epsilon=predictor_spec['Epsilon'],
                         decay=predictor_spec['Decay Rate'])

    elif predictor_spec['Optimizer'] == 'adagrad':
        # specify adagrad optimizer
        optimizer = Adagrad(lr=predictor_spec['Learning Rate'],
                            epsilon=predictor_spec['Epsilon'],
                            decay=predictor_spec['Decay Rate'])

    elif predictor_spec['Optimizer'] == 'adadelta':
        # specify adadelta optimizer
        optimizer = Adadelta(lr=predictor_spec['Learning Rate'],
                             rho=0.95,
                             epsilon=predictor_spec['Epsilon'],
                             decay=predictor_spec['Decay Rate'])
    else:
        # specify rmsprop optimizer
        optimizer = RMSprop(lr=predictor_spec['Learning Rate'],
                            rho=0.9,
                            epsilon=predictor_spec['Epsilon'],
                            decay=predictor_spec['Decay Rate'])

    # TODO: string manipulation: string -> list/tuple
    model = build_model([int(layer) for layer in
                         re.findall(r'[0-9]*', predictor_spec['Layer Units']) if layer.isdigit()],
                        [float(dropout) for dropout in
                         re.findall(r'[0-9]*', predictor_spec['Dropout Rates']) if dropout.isdigit()],
                        rnn_type=predictor_spec['Type'],
                        optimizer=optimizer,
                        activation=predictor_spec['Activation'],
                        nb_preds=config.NB_ASSETS)

    # train prediction model
    model.fit(
        trainX,
        trainY,
        batch_size=predictor_spec['Batch Size'],
        epochs=20,
        validation_split=0.1,
        verbose=0,
        callbacks=[tensorboard, terminate_nan, early_stopping]
    )

    # evaluate trained prediction model on test data
    score = model.evaluate(
        testX,
        testY,
        batch_size=predictor_spec['Batch Size'],
        verbose=0
    )
    model.save(config.PREDICTION_MODEL)
    print('\nModel has been saved\nMSE: ', score[0])
    print('MAE: ', score[1])

    return model, score


if __name__ == '__main__':
    main()
