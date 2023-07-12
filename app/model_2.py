import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

import numpy as np

import warnings
warnings.filterwarnings("ignore")


#하이퍼 파라미터 및 상수
RANDOM_SEED = 2022

BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
EPOCHS = 10

lr = 0.01
momentum = 0.5
log_interval = 200

USE_CUDA = torch.cuda.is_available()
print("Device : {0}".format("GPU" if USE_CUDA else "CPU"))
device = torch.device("cuda" if USE_CUDA else "cpu")
cpu_device = torch.device("cpu")


#랜덤 시드 지정
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)

print('Random Seed : {0}'.format(RANDOM_SEED))



#dataload
# Define the path to your dataset folder
data_path = './data/100_1'

# Define the transformation to apply to the images
transform = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale if necessary
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the training dataset
train_dataset = ImageFolder(root=data_path, transform=transform)

# Create a data loader for the training dataset
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Load the test dataset
test_dataset = ImageFolder(root=data_path, transform=transform)

# Create a data loader for the test dataset
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=True
)

# Create a data loader for testing with batch size 1 (if needed)
test_loader_bs1 = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=True
)


#신경망 선언
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        h5 = F.relu(self.fc5(h4))
        h6 = self.fc6(h5)
        return F.log_softmax(h6, dim=1)
    

#신경망, 옵티마이저 선언
model = Model().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


#학습, 테스트 함수
def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(log_interval, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() 
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format
          (test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


#학습 및 실험
for epoch in range(1, EPOCHS+1):
    train(log_interval, model, device, train_loader, optimizer, epoch)
    test(log_interval, model, device, test_loader)

#모델 저장
# Create the directory if it doesn't exist
os.makedirs("./model", exist_ok=True)

# Save the model and optimizer state to the specified directory
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, "./model/model.pth")