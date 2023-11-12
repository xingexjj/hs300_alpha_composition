import os
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import same_seed
from loss import ICLoss
from dataset import AlphaDataset
from model import LSTM, DNN


def trainer(train_loader, valid_loader, model, device, config):
    '''
    Train model with train loader and evaluate with valid loader.
    device ('cuda' or 'cpu'): device to train and evaluate
    '''
    criterion = nn.MSELoss()
    # criterion = ICLoss

    # optimizer = torch.optim.SGD(model.parameters(), lr = config['learning_rate'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.1, verbose=True) 

    # Writer of tensorboard.
    writer = SummaryWriter()

    # Create directory of saving models.
    if not os.path.exists(config['MODEL_PATH']):
        os.mkdir(config['MODEL_PATH'])

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], np.inf, 0, 0

    train_loss_record = []
    valid_loss_record = []

    for epoch in range(n_epochs):
        # Set your model to train mode.
        model.train() 
        loss_record = []    

        # tqdm is a package to visualize your training process
        train_pbar = tqdm(train_loader, position = 0)
        for x, y in train_pbar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x).squeeze()
            loss = criterion(pred, y)
            loss.backward()

            torch.nn.utils.clip_grad_value_(parameters=model.parameters(), clip_value = 0.5)
            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)

            optimizer.step()
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
        train_loss_record.append(mean_train_loss)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        # Set your model to evaluation mode
        model.eval()
        loss_record = []
        for x, y in valid_loader:

            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x).squeeze()
                loss = criterion(pred, y)

            loss_record.append(loss.detach().item())

        scheduler.step(sum(loss_record))
        mean_valid_loss = sum(loss_record)/len(loss_record)
        valid_loss_record.append(mean_valid_loss)

        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss}, Valid loss: {mean_valid_loss}, lr: {optimizer.param_groups[0]["lr"]}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss

            SAVE_PATH = f'{config["MODEL_PATH"]}/{config["train_start"]}_{config["train_end"]}'
            if not os.path.exists(SAVE_PATH):
                os.mkdir(SAVE_PATH)

            with open(f'{SAVE_PATH}/{epoch+1}.pkl', 'wb') as f:
                pickle.dump(model.state_dict(), f)

            print('Saving model with loss {:.5f}...'.format(best_loss))
            early_stop_count = 0 
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')  
            with open('./train_loss_record.pkl', 'wb') as f:
                pickle.dump(train_loss_record, f)
            with open('./valid_loss_record.pkl', 'wb') as f:
                pickle.dump(valid_loss_record, f)
            return 

def tester(test_loader, model, config, device):
    criterion = ICLoss

    # Set your model to evaluation mode.
    model.eval() 
    preds = []
    loss_record = []
    for x, y in tqdm(test_loader):
        x, y = x.to(device), y.to(device)                      
        with torch.no_grad():                   
            pred = model(x).squeeze()      
            loss = criterion(pred, y)    

        preds.append(pred.detach().cpu())  
        loss_record.append(loss.detach().item())
    
    mean_test_loss = np.nanmean(loss_record)

    print(f'Test loss: {mean_test_loss}')
        
    preds = torch.cat(preds, dim=0).numpy()
    SAVE_PATH = f'{config["MODEL_PATH"]}/{config["train_start"]}_{config["train_end"]}/preds'
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    with open(f'{SAVE_PATH}/{config["test_start"]}_{config["test_end"]}.pkl', 'wb') as f:
        pickle.dump(preds, f)
    

def run(config):
    # Set seed for reproducibility
    same_seed(config['seed'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert config['type'] in ['train', 'test', 'all'], 'type should be train, test or all'

    if config['type'] in ['train', 'all']:

        train_dataset = AlphaDataset(config['DATA_PATH'], config['train_start'], config['train_end'], config['period'])   
        valid_dataset = AlphaDataset(config['DATA_PATH'], config['valid_start'], config['valid_end'], config['period'])

        # print out the data size.
        print(f'train_data size: {train_dataset.x.shape}\nvalid_data size: {valid_dataset.x.shape}')

        # Pytorch data loader loads pytorch dataset into batches.
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

        model = DNN(input_size = train_dataset.x.shape[1] * train_dataset.x.shape[2]).to(device)
        # model = LSTM(input_size = train_dataset.x.shape[-1]).to(device)

        trainer(train_loader, valid_loader, model, device, config)

    if config['type'] in ['test', 'all']:

        test_dataset = AlphaDataset(config['DATA_PATH'], config['test_start'], config['test_end'], config['period'], save_info = True)

        # print out the data size.
        print(f'test_data size: {test_dataset.x.shape}')

        # Pytorch data loader loads pytorch dataset into batches.
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

        model = DNN(input_size = test_dataset.x.shape[1] * test_dataset.x.shape[2]).to(device)
        # model = LSTM(input_size = test_dataset.x.shape[-1]).to(device)
        
        # Load model
        LOAD_PATH = f'{config["MODEL_PATH"]}/{config["train_start"]}_{config["train_end"]}'
        assert os.path.exists(LOAD_PATH), 'Model does not exist!'
        last_epoch = -1
        for epoch in range(config['n_epochs']):
            if os.path.exists(f'{LOAD_PATH}/{epoch+1}.pkl'):
                last_epoch = epoch
        assert last_epoch >= 0, 'Model does not exist!'
        
        with open(f'{LOAD_PATH}/{last_epoch+1}.pkl', 'rb') as f:
            state_dict = pickle.load(f)
        model.load_state_dict(state_dict)

        tester(test_loader, model, config, device)


        
        


                
