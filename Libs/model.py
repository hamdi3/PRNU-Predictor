import torch
import torch.nn as nn

# Define the network
class FCNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FCNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

# Load Pretrained Model
input_size = 480 * 480 # The image height and width on the which the network was trained 
hidden_size = 64  
num_classes = 11 
model = FCNClassifier(input_size, hidden_size, num_classes)

model.load_state_dict(torch.load('Data/Models/PRNU_Classifier.pth'))