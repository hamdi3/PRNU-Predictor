import torch
import torch.nn as nn
import platform

# Define the network
class FCNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        """
        Initializes the FCNClassifier module.

        Args:
            input_size (int): The size of the input tensor.
            hidden_size (int): The number of hidden units in the network.
            num_classes (int): The number of output classes.
        """
        super(FCNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu1 = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(hidden_size, num_classes)  # Second fully connected layer

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = x.view(x.size(0), -1)  # Reshape input tensor
        x = self.fc1(x)  # First fully connected layer
        x = self.relu1(x)  # ReLU activation
        x = self.fc2(x)  # Second fully connected layer
        return x

# Load Pretrained Model
input_size = 480 * 480  # The image height and width on which the network was trained
hidden_size = 64
num_classes = 11
model = FCNClassifier(input_size, hidden_size, num_classes)

# Check the system for the path formatting
if platform.system() == 'Windows':
    model.load_state_dict(torch.load(r'Data\Models\PRNU_Classifier.pth'))  # Use raw string with 'r' to preserve backslashes
else:
    model.load_state_dict(torch.load('Data/Models/PRNU_Classifier.pth'))  # Use forward slashes for Linux
