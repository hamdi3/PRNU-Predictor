import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .local_prnu import get_PRNU
from .utils import get_key_from_value

# Define the FCN classifier
class FCNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FCNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

# Define the training function
def train_classifier(model, train_dataloader, val_dataloader, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0

        model.train()
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        model.eval()
        for inputs, labels in val_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

        val_loss /= len(val_dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")

# Define the test function
def test_classifier(model, test_dataloader):
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    num_correct = 0
    num_samples = 0

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            num_correct += (predicted == labels).sum().item()
            num_samples += labels.size(0)

    test_loss /= len(test_dataloader)
    accuracy = num_correct / num_samples

    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

def load_model(weight_path):
    input_size = 1 * 480 * 480
    hidden_size = 64
    num_classes = 11
    model = FCNClassifier(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(weight_path))
    return model

model = load_model('Data/Models/PRNU_Classifyer.pth')

label_dict = {'FrontCamera-GalaxyA13-225951': 0, 'FrontCamera-GalaxyA13-225952': 1, 'Logitech Brio210500': 2, 'Logitech Brio210504': 3, 'Logitech Brio210506': 4, 'Logitech C50596011268': 5, 'Logitech C50596011268_2': 6, 'Logitech C50596011268_3': 7, 'Nikon_Zfc': 8, 'RückCamera-GalaxyA13-225951': 9, 'Rückcamera-GalaxyA13-225952': 10}

def predict_image(img_path, model = model,label_dict = label_dict):
    image_prnu = torch.tensor(np.array(get_PRNU(img_path)))
    prediction = model(image_prnu)
    _, predicted = torch.max(prediction, 1)
    return get_key_from_value(label_dict,predicted.item())