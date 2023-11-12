import torch.nn as nn

class LSTM(nn.Module):
    '''LSTM Model'''
    def __init__(self, input_size, hidden_size = 32):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            batch_first = True
        )

        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        
        return self.linear(self.lstm(x)[1][0].squeeze())
    
class DNN(nn.Module):
    '''DNN Model'''
    def __init__(self, input_size):
        super(DNN, self).__init__()
        self.relu_stack = nn.Sequential(
            nn.Linear(200, 128),
            nn.RReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.RReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.RReLU()
        )

    def forward(self, x):
        output = self.relu_stack(x.reshape(x.shape[0], -1))
        return output
    


    




