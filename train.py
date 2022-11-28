import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from similarity_dataset import SimilarityDataset
from similarity_model import SimilarityNet
from tqdm import tqdm

def train():

    batch_size = 4

    # Dataset
    train_dataset = SimilarityDataset()

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    for input1, input2 in train_dataloader:
        print(input1.shape)
        print(input2.shape)
        break
    
    # Network
    net = SimilarityNet()

    # criterion
    criterion = nn.CosineEmbeddingLoss()

    # Optimizer
    optimizer = optim.Adam(net.parameters())

    # Device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    print(f'Using {device} device')

    net = net.to(device)

    # Epoch
    epochs = 10

    for epoch in range(epochs):
        print(f'Epoch : {epoch} ==========')
        train_loss = 0.0
        for input1, input2 in train_dataloader:
            input1 = input1.to(device)
            input2 = input2.to(device)
            labels = torch.ones(size=(batch_size,), dtype=torch.float, device=device)

            output1 = net(input1)
            output2 = net(input2)

            loss = criterion(output1, output2, labels)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print({loss.item()})
        
        print(f'Train Loss : {train_loss}')
        return

        

if __name__ == '__main__':
    train()