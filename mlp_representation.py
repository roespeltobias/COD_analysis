from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Define a simple MLP class with multiple hidden layers appended in a loop
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        # Create a ModuleList to store hidden layers
        self.hidden_layers = nn.ModuleList()
        
        # Append the first hidden layer: from input_dim to the first hidden dimension
        self.hidden_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # Loop over the remaining hidden dimensions to add additional layers
        for i in range(1, len(hidden_dims)):
            self.hidden_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        
        # Define the output layer: from the last hidden dimension to output_dim
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        # Activation function
        self.relu = nn.ReLU()
        
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # Pass the input through all hidden layers with ReLU activation
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        # Pass through the output layer (without activation if using CrossEntropyLoss)
        x = self.output_layer(x)
        return x
    
    def train_model(self, input, target, lr, n_epochs):
        loss_list = []
        
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in tqdm(range(n_epochs), desc="Training Epochs"):
            optimizer.zero_grad()          # Reset the gradients for the current epoch
            outputs = self.forward(input)       # Forward pass
            loss = self.criterion(outputs, target)  # Calculate the loss
            loss.backward()                # Backward pass (compute gradients)
            optimizer.step()               # Update model parameters
            loss_list.append(loss.item())
        plt.plot(loss_list)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()
        plt.close()