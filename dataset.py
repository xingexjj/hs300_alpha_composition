import os
from tqdm import tqdm
import pickle

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd


class AlphaDataset(Dataset):

    def __init__(self, PATH, start, end, period = 20, save_info = False):
        '''
            Create new alpha dataset.
            PATH   (str): path to load data and store dataset.
            start  (str): start time of the dataset, format 'YYYYmmdd'.
            end    (str): end time of the dataset, format 'YYYYmmdd'.
            period (int): length of one single x data.
            save_info (bool): whether to save dataset info or not.
        '''
        # Load from pickle if dataset already exists
        LOAD_PATH = f'{PATH}/alpha_dataset/{start}_{end}_{period}'
        if os.path.exists(LOAD_PATH):

            print('Dataset exists. Start loading.')

            with open(f'{LOAD_PATH}/x.pkl', 'rb') as f:
                self.x = pickle.load(f)
            with open(f'{LOAD_PATH}/y.pkl', 'rb') as f:
                self.y = pickle.load(f)
            
            assert self.x.shape[0] == self.y.shape[0], 'x and y should get same length'
            assert self.x.shape[1] == period, "a piece of x data's length should be equal to period"
            print(f'Successfully load dataset from {LOAD_PATH} !')

        # Create dataset and dump pickle
        else:

            print('Start creating dataset.')

            self.x = []
            self.y = []
            info = []

            # Load alpha
            alpha_list = []
            for file_name in os.listdir(f'{PATH}/alpha_list'):
                if file_name.endswith('.pkl'):
                    alpha_list.append(pd.read_pickle(f'{PATH}/alpha_list/{file_name}')[start: end])

            # Load return data
            close = pd.read_pickle(f'{PATH}/HS300_close.pkl').loc[start: end]
            # Compute monthly return
            ret = (close.shift(-20) - close) / close
            
            # Get data from each stock
            for stock_name in tqdm(alpha_list[0].columns, position = 0):
                # Get alpha values for each stock
                stock = pd.concat([alpha[stock_name].rename(alpha.index.name) for alpha in alpha_list], 
                                axis = 1)
                stock.index.name = stock_name
                # Get period stock data and handle NaN values
                for stock_p in list(stock.rolling(period))[period - 1:]:
                    if not stock_p.isna().any().any() and not np.isnan(ret[stock_name].loc[stock_p.index[-1]]):
                        # add a piece of data if x and y do not conclude NaN
                        self.x.append(stock_p.values)
                        self.y.append(ret[stock_name].loc[stock_p.index[-1]])
                        # save stock name and date info for test use
                        if save_info:
                            info.append({'stock name': stock_name, 'date': stock_p.index[-1]})

            self.x = torch.FloatTensor(np.array(self.x).astype('float'))
            self.y = torch.FloatTensor(self.y)

            # Dump pickle
            DUMP_PATH = LOAD_PATH
            os.mkdir(DUMP_PATH)
            with open(f'{DUMP_PATH}/x.pkl', 'wb') as f:
                pickle.dump(self.x, f)
            with open(f'{DUMP_PATH}/y.pkl', 'wb') as f:
                pickle.dump(self.y, f)
            with open(f'{DUMP_PATH}/info.pkl', 'wb') as f:
                pickle.dump(info, f)

            print(f'Successfully create dataset and dump to {DUMP_PATH} !')

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)