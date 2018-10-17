import pandas as pd
import os


if __name__ == '__main__':
    for file in os.listdir(os.getcwd()):
        if file.endswith('.csv'):
            df = pd.read_csv(file, index_col=0)
            print('\n', df)
