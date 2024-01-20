import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
torch.manual_seed(1)
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler

from src.utils import read_config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to find GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

# Read config and set up tensorboard logging
config = read_config("config.yaml")
filename = "best_model"

# Reading trajectories from csv file and add padding if necessary
df = pd.read_csv('trajectories.csv', header=None, names=list(range(1001))).dropna(axis='columns', how='all')
df = df.fillna(0)
trajectories = df.values.astype('float32')

# Reading masses from csv file
df = pd.read_csv('labels.csv', header=None)
labels = df.values.astype('float32')

# Setting LSTM network parameters
seq_len = 120
hidden_len = 1000
batch_size = 100
num_states = int(len(trajectories)/len(labels))

# Generating measurements with right shape en length
def create_dataset(trajectories, labels, window=seq_len, scale=True, noise_level=0, scale_features=False):
    """Transform trajectories and labels into prediction dataset 
    
    Args:
        trajectories: (num_states*N x 1001), num. states times num. samples x max_episode_steps
        labels: (N x k) list, num. samples x num. parameters
        window: length of individual sequence, padded with zeros
        scale: rescale the labels based on min-max normalization
        noise_level: scalar percentage between 0 and 1, adds noise on the states 
    """ 
    # Find number of states
    num_states = int(len(trajectories)/len(labels))

    # Create feature and target lists
    X, y = [], []
    for i in range(len(labels)):
        feature = trajectories[i*num_states:(i+1)*num_states, :window]
        if scale_features:
            high = np.array([1., 5., 1., 5., np.pi, 20.]) # maximum value for the states
            low = np.concatenate([-high, np.zeros(num_states-len(high))]).reshape(num_states,1) * np.ones_like(feature) # Adds a zero for every motor
            high = np.concatenate([high, np.ones(num_states-len(high))]).reshape(num_states,1) * np.ones_like(feature) # Adds a one for every motor
            # if i == 89:
            #     print(high)
            #     print("Unscaled feature:")
            #     print(feature)
            feature = (feature - low) / (high - low)
            feature = np.array([[0 if element == 0.5 else element for element in row] for row in feature])
                # print("Scaled feature")
                # print(feature)
        target = labels[i,:]
        X.append(feature.astype('float32'))
        y.append(target)
    # Apply min-max scaling to labels
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(y)
        y = scaler.transform(y).astype('float32')

    if noise_level > 0:
        # high = np.array([1., 5., 1., 5., np.pi, 20.]) # maximum value for the states
        # high = np.concatenate([high, np.ones(num_states-len(high))]) # Adds a one for every motor
        # for i in range(int(len(feature)/num_states)):
        #     noise = noise_level * np.random.normal(np.zeros((len(high), window)), np.tile(high/3,(window,1)).T, size=(len(high),window)).astype('float32')
        #     X[i*num_states:(i+1)*num_states] = X[i*num_states:(i+1)*num_states] + noise
        noiseX = np.random.normal(0,noise_level,size=(len(X),len(X[0]),window)).astype('float32')
        noiseX = 1 + np.clip(noiseX, -3 * noise_level, 3 * noise_level)
        X = X * noiseX
        noisey = np.random.normal(0,noise_level,size=(len(y),len(y[0]))).astype('float32')
        noisey = 1 + np.clip(noisey, -3 * noise_level, 3 * noise_level)
        y = y * noisey

    return torch.tensor(X), torch.tensor(y)

# Divide into training set and testing set
train_size = int(len(labels) * 0.6)
test_size = len(labels) - train_size
X, y = create_dataset(trajectories, labels, window=seq_len, scale=True, noise_level=0)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]
print("Train size: " + str(X_train.shape[0]) + " Test size: " + str(X_test.shape[0]))

class EstimateParams(nn.Module):
    def __init__(self, sequence_length, hidden_dim, batch_size, num_lstms, num_params):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_lstms = num_lstms

        # LSTM layer that processes the sequence
        self.lstm = nn.LSTM(input_size=sequence_length, hidden_size=self.hidden_dim, num_layers=num_lstms, batch_first=True, dropout=0)
        
        # Dense layer to predict the parameters
        extra_hidden = 30
        self.fc1 = nn.Linear(self.hidden_dim*self.num_lstms*(2+8), num_params*extra_hidden)
        self.fc2 = nn.Linear(num_params*extra_hidden, num_params)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_t = Variable(
            torch.zeros(self.num_lstms, x.shape[0], self.hidden_dim), requires_grad=False)
        c_t = Variable(
            torch.zeros(self.num_lstms, x.shape[0], self.hidden_dim), requires_grad=False)
        x, (h_t, c_t) = self.lstm(x, (h_t, c_t))
        x = x.contiguous().view(x.shape[0],-1)
        h_t = h_t.contiguous().view(x.shape[0],-1)
        c_t = h_t.contiguous().view(x.shape[0],-1)
        # print(x.shape)
        # print(h_t.shape)
        out = torch.cat((x, h_t, c_t), 1)
        # print(out.shape)
        fout = self.fc1(out)
        fout = self.fc2(fout)
        return fout.view(-1,fout.shape[1])
    
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Calculate the squared error
        squared_error = torch.pow(abs(y_true - y_pred), 2.5)

        # Apply a weight to the squared error
        weighted_error = squared_error * torch.clip(torch.exp(squared_error), -10, 10)

        # Calculate the mean of the weighted errors
        mean_weighted_error = torch.mean(weighted_error)

        return mean_weighted_error
    
def RMSE(y_pred, y_true):
    # Calculate the squared error
    squared_error = torch.pow(y_true - y_pred, 2)
    return np.sqrt(torch.mean(squared_error))

# Create model, optimizer, loss function, and data loader
model = EstimateParams(sequence_length=seq_len, hidden_dim=hidden_len, batch_size=batch_size, num_lstms=1, num_params=len(y[0]))
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = CustomLoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)

n_epochs = 2000
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        # Extract tensors from the tuple
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = RMSE(y_pred, y_train)
        y_pred = model(X_test)
        test_rmse = RMSE(y_pred, y_test)
        print(y_test - np.array(model(X_test)))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

with torch.no_grad():
    # Calculate scaled labels
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(labels)
    labels_scaled = scaler.transform(labels).astype('float32')
    
    train_error_plot = labels_scaled[:train_size,:]-np.array(model(X_train))

    # shift test predictions for plotting
    test_plot = np.ones_like(labels) * np.nan
    test_error_plot = labels_scaled[train_size:,:]-np.array(model(X_test))
# plot
# plt.plot(labels)
plt.plot(range(train_size), train_error_plot, c='r')
plt.plot(range(train_size, len(labels)), test_error_plot, c='g')
plt.show()