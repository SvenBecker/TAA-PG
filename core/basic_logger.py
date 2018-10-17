import logging


def get_logger(filename='logs.log', logger_name=__name__):

    logging.basicConfig(level=logging.INFO,
                        filename=filename,
                        filemode='w',
                        format='%(name)s %(asctime)s %(levelname)s: %(message)s',
                        datefmt='%H:%M:%S')

    logger = logging.getLogger(logger_name)

    # add stream handler which prints to stderr
    ch = logging.StreamHandler()

    # modify stream handler log format
    formatter_ch = logging.Formatter(
        '%(levelname)s - %(message)s')

    ch.setFormatter(formatter_ch)

    # set stream handler log level
    ch.setLevel(logging.WARNING)

    logger.addHandler(ch)

    # return logger object
    return logger
