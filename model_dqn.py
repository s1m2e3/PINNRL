import torch
import numpy as np
import torch.nn as nn
import statsmodels.api as sm
# import pytorch_forecasting 


def reshape_tensor(x, batch_size):
    original_shape = x.size()
    num_elements = x.numel()
    new_shape = (batch_size,) + original_shape[1:]
    if num_elements != new_shape[0] * torch.tensor(new_shape[1:]).prod():
        raise ValueError("Number of elements in tensor does not match new shape")
    return x.view(new_shape)

class NN(nn.Module):
    def __init__(self, input_size1, hidden_size, output_size,lr=0.001,weight_decay=0,sarimax=None):
        super(NN, self).__init__()
        
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.number_of_nodes = hidden_size
        self.fc1 = nn.Linear(in_features=input_size1,  out_features=output_size,dtype=torch.float).to(dev)
        self.relu = nn.ReLU().to(dev)
        self.fc2 = nn.Linear(in_features= output_size, out_features=hidden_size,dtype=torch.float).to(dev)
        self.relu = nn.ReLU().to(dev)
        self.fc3 = nn.Linear(in_features= hidden_size, out_features=output_size,dtype=torch.float).to(dev)
        self.optimizer = torch.optim.SGD(self.parameters(),lr=lr,weight_decay=weight_decay)
        self.criterion = nn.SmoothL1Loss()
        self.sarimax = sarimax
        self.epsilon=1
        self.epsilon_decay=1e-4
    def forward(self, x):
        if torch.cuda.is_available():
            dev = "cuda:0"
            
        else:
            dev = "cpu"
        device = torch.device(dev)
        out = torch.tensor(x,dtype=torch.float).to(device)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)

        return out
    def forward_linear(self,x):
        if torch.cuda.is_available():
            dev = "cuda:0"
            
        else:
            dev = "cpu"
        device = torch.device(dev)
        out = torch.tensor(x,dtype=torch.float).to(device)
        out = self.fc1(out)
        out = self.relu(out)
        
        return out

    def sarimax_pred(self,stream):
        self.sarimax = self.sarimax.append([stream[-1]],disp=False)
        return torch.tensor(self.sarimax.forecast(),dtype=torch.float)

    def train(self,num_epochs,x_train_data,y_train_data,autocorr=False,lambda_=0.001):
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        device = torch.device(dev)
        if autocorr:
            sarimax_data = [list(x_train_data[stream,:,0]) for stream in range(len(x_train_data))]
            sarimax_predictions = [self.sarimax_pred(sarimax_data[i]) for i in range(len(sarimax_data))]
            sarimax_predictions_tensor=sarimax_predictions[0]
            for i in range(len(sarimax_predictions)-1):
                sarimax_predictions_tensor=torch.vstack((sarimax_predictions_tensor,sarimax_predictions[i+1])) 
        
        # x_train_data = torch.tensor(np.array(x_train_data),dtype=torch.float).to(device)
        # y_train_data = torch.tensor(np.array(y_train_data),dtype=torch.float).to(device)
        
        # print(self.forward(x_train_data)-y_train_data)
        # print(self.criterion(self.forward(x_train_data),y_train_data))
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            outputs = self.forward(x_train_data)
            
            loss = self.criterion(outputs, y_train_data)
            if autocorr:
                output_first=self.forward_linear(x_train_data)[:,0]
                loss += self.criterion(output_first,sarimax_predictions_tensor)*lambda_
            
            loss.backward()
            
            self.optimizer.step()
        # print(epoch)    
        # Print training statistics
            if (epoch+1) % 10 == 0:
                # print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                    # .format(epoch+1, num_epochs, epoch+1, len(x_train_data), loss.item()))
                print("mse ",loss,"number of nodes:",self.number_of_nodes)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,input_sequence_length,output_sequence_length):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        print(self.hidden_size)
        self.num_layers = num_layers
        if torch.cuda.is_available():
            dev = "cuda:0"
            
        else:
            dev = "cpu"
        self.lstm = nn.LSTM(input_size,hidden_size, num_layers,batch_first=True,dtype=torch.float).to(dev)
        self.relu = nn.ReLU()
        self.output_sequence_length = output_sequence_length
        self.input_sequence_length = input_sequence_length
        self.fc1 = nn.Linear(hidden_size, output_size,dtype=torch.float).to(dev)
        # self.fc2 = nn.Linear(512, 512,dtype=torch.float).to(dev)
        # self.fc3 = nn.Linear(512, output_size,dtype=torch.float).to(dev)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(),lr=0.1)


    def forward(self, x):
        if torch.cuda.is_available():
            dev = "cuda:0"
            
        else:
            dev = "cpu"
        device = torch.device(dev)
        x = torch.tensor(x,dtype=torch.float).to(device)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,dtype=torch.float).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,dtype=torch.float).to(x.device)
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        out = self.relu(out[:,self.input_sequence_length-self.output_sequence_length:,:])
        out = self.fc1(out)
        # out = self.relu(out)
        # out = self.fc2(out)
        # out = self.relu(out)
        # out = self.fc3(out)
        
        return out

    def train(self,num_epochs,x_train_data,y_train_data):
        if torch.cuda.is_available():
            dev = "cuda:0"
            
        else:
            dev = "cpu"
        device = torch.device(dev)
        x_train_data = torch.tensor(x_train_data,dtype=torch.float).to(device)
        y_train_data = torch.tensor(y_train_data,dtype=torch.float).to(device)
        
        for epoch in range(num_epochs):
            
            self.optimizer.zero_grad()
            outputs = self.forward(x_train_data)
            
            loss = self.criterion(outputs, y_train_data)
            loss.backward()
            self.optimizer.step()
            

            # Print training statistics
            if (epoch+1) % 100 == 0:
                print(torch.cuda.get_device_name(0))
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                    .format(epoch+1, num_epochs, epoch+1, len(x_train_data), loss.item()))
        print("mse ",loss,"number of nodes:",512)
        print("number of hidden:",self.hidden_size,"number of hidden layers:",self.num_layers)

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(),lr=0.1)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.linear(out)
        return out, hidden
    def train(self,num_epochs,x_train_data,y_train_data):
        if torch.cuda.is_available():
            dev = "cuda:0"
            
        else:
            dev = "cpu"
        device = torch.device(dev)
        x_train_data = torch.tensor(x_train_data,dtype=torch.float).to(device)
        y_train_data = torch.tensor(y_train_data,dtype=torch.float).to(device)
        
        # Training loop
        for epoch in range(num_epochs):
            
            self.optimizer.zero_grad()
            outputs, hidden = self(x_train_data, hidden)
            
            loss = self.criterion(outputs, y_train_data)
            loss.backward()
            self.optimizer.step()
            
            # Print training statistics
            if (epoch+1) % 100 == 0:
                print(torch.cuda.get_device_name(0))
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                    .format(epoch+1, num_epochs, epoch+1, len(x_train_data), loss.item()))
        print("mse ",loss,"number of nodes:",512)
        print("number of hidden:",self.hidden_size,"number of hidden layers:",self.num_layers)
