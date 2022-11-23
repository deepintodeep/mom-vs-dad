import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from similarity_dataset import SimilarityDataset
from similarity_model import SimilarityNet

def train():
    # Dataset
    train_dataset = SimilarityDataset()
    val_dataset = None

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True
    )
    val_dataloader = None

    for inputs, labels in train_dataloader:
        print(inputs.shape)
        print(len(labels))
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
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)

            loss = criterion()
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Train Loss : {train_loss}')

        

if __name__ == '__main__':
    train()