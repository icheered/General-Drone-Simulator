
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class ParameterEstimator(nn.Module):
    def __init__(self, sequence_length, hidden_dim, batch_size, num_lstms, input_neurons, output_neurons):
        super(ParameterEstimator, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_lstms = num_lstms

        # LSTM layer that processes the sequence
        self.lstm = nn.LSTM(input_size=sequence_length, hidden_size=self.hidden_dim, num_layers=num_lstms, batch_first=True)
        
        # Dense layer to predict the parameters
        lstm_additional_state = 2 # Hidden state and cell state
        self.fc1 = nn.Linear(self.hidden_dim*self.num_lstms*(lstm_additional_state+input_neurons), output_neurons)

    def forward(self, x):
        # Convert input to torch.float32
        x = x.float()
        hidden_states = Variable(torch.zeros(self.num_lstms, x.shape[0], self.hidden_dim), requires_grad=False)
        cell_states = Variable(torch.zeros(self.num_lstms, x.shape[0], self.hidden_dim), requires_grad=False)
        
        x, (hidden_states, cell_states) = self.lstm(x, (hidden_states, cell_states))

        x = x.contiguous().view(x.shape[0], -1)
        hidden_states = hidden_states.contiguous().view(x.shape[0], -1)
        cell_states = hidden_states.contiguous().view(x.shape[0], -1)
        out = torch.cat((x, hidden_states, cell_states), 1)
        fout = self.fc1(out)
        return fout.view(-1, fout.shape[1])

    def custom_loss(self, y_pred, y_true):
        squared_error = torch.pow(abs(y_true - y_pred), 2)
        mean_weighted_error = torch.mean(squared_error) * torch.max(squared_error)
        return mean_weighted_error
    
    def RMSE(self, y_pred, y_true):
        # Calculate the squared error
        squared_error = torch.pow(y_true - y_pred, 2)
        return torch.sqrt(torch.mean(squared_error)).item()

    
    def pre_process(self, traj, labels, window, noise_level=0):
        X = []
        if len(traj[0]) < window:
            for row in traj:
                padded_row = row + [0] * (window - len(row))
                X.append(padded_row)
        elif len(traj[0]) > window:
            X = [row[-window:] for row in traj]
        else:
            X = traj

        if noise_level:
            noiseX = np.random.normal(0, noise_level, size=(len(X), len(X[0]))).astype('float32')
            noiseX = 1 + np.clip(noiseX, -3 * noise_level, 3 * noise_level)
            X = X * noiseX
        
        # Reshape X to the desired dimensions (250x9x20)
        X = np.array(X)
        X = X.transpose(0, 2, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(labels)
        y = scaler.transform(labels).astype('float32')
        return torch.tensor(X), torch.tensor(y)
