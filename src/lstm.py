
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
        self.lstm = nn.LSTM(input_size=input_neurons, hidden_size=self.hidden_dim, num_layers=num_lstms, batch_first=True)
        
        # Normalization layer
        # self.batchnorm = nn.BatchNorm1d(num_features=input_neurons)

        # Dense layer to predict the parameters
        lstm_additional_state = 2 # Hidden state and cell state
        # self.fc1 = nn.Linear(self.hidden_dim*self.num_lstms*(lstm_additional_state+input_neurons), output_neurons)
        self.fc1 = nn.Linear(self.hidden_dim*self.num_lstms*(lstm_additional_state), output_neurons)

        # self.fc1 = nn.Linear(self.hidden_dim*self.num_lstms*(input_neurons), output_neurons)
        # self.fc2

    def forward(self, x):
        # Convert input to torch.float32
        x = x.float()
        # print(x.shape)
        # x = self.batchnorm(x)
        hidden_states = Variable(torch.zeros(self.num_lstms, x.shape[0], self.hidden_dim), requires_grad=False)
        cell_states = Variable(torch.zeros(self.num_lstms, x.shape[0], self.hidden_dim), requires_grad=False)
        
        x, (hidden_states, cell_states) = self.lstm(x, (hidden_states, cell_states))

        # print(x.shape)
        # print(hidden_states.shape)
        # print(cell_states.shape)

        x = x.contiguous().view(x.shape[0], -1)
        hidden_states = hidden_states.contiguous().view(x.shape[0], -1)
        cell_states = hidden_states.contiguous().view(x.shape[0], -1)
        # print(x.shape)
        # print(hidden_states.shape)
        # print(cell_states.shape)
        out = torch.cat((hidden_states, cell_states), 1)
        fout = self.fc1(out)
        return fout.view(-1, fout.shape[1])

    def custom_loss(self, y_pred, y_true):
        squared_error = torch.pow(abs(y_true - y_pred), 2)
        RMS_error = torch.sqrt(torch.mean(squared_error))
        variance = 1 + torch.var(squared_error)
        return RMS_error*variance
    
    def RMSE(self, y_pred, y_true):
        # Calculate the squared error
        squared_error = torch.pow(y_true - y_pred, 2)
        return torch.sqrt(torch.mean(squared_error))

    
    def pre_process(self, traj, labels, window, noise_level=0):
        X = []
        traj = np.array(traj)
        if traj.ndim == 2:
            if len(traj) < window:
                X = np.zeros((len(traj[0]), window))
                X[:, :len(traj)] = traj.transpose()
            elif len(traj) > window:
                X = traj.transpose()
                X = X[:, -window:]
            else:
                X = traj.transpose()
        else:
            X = traj
        
        if noise_level:
            num_states = int(len(X)/len(labels))
            noiseX = np.random.normal(0, noise_level, size=(len(X), len(X[0]), num_states)).astype('float32')
            noiseX = 1 + np.clip(noiseX, -3 * noise_level, 3 * noise_level)
            X = X * noiseX
        
        # Reshape X to the desired dimensions
        X = np.array(X)
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
            X = X.transpose(2, 1, 0)
        else:
            X = X.transpose(0, 1, 2)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit([[0.1, 0.1], [1, 2]])
        y = np.array(labels)
        if y.ndim == 1:
            y = scaler.transform(y.reshape(1, -1)).astype('float32')
        else:
            y = scaler.transform(y).astype('float32')
        return torch.tensor(X), torch.tensor(y)