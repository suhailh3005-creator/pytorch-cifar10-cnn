# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

try:
  from .model import SimpleCNN
except ImportError:
  from model import SimpleCNN


def main():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'Using compute device: {device}')

  # Transforms & DataLoaders
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  trainset = torchvision.datasets.CIFAR10(
      root='./data', train=True, download=True, transform=transform
  )
  trainloader = torch.utils.data.DataLoader(
      trainset, batch_size=32, shuffle=True, num_workers=2
  )

  model = SimpleCNN().to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  epochs = 3
  for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
      inputs, labels = inputs.to(device), labels.to(device)

      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      if (i + 1) % 400 == 0:
        print(
            f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(trainloader)}],'
            f' Loss: {running_loss / 400:.4f}'
        )
        running_loss = 0.0

  print('Training Complete.')

  # Save weights to artifacts folder
  os.makedirs('artifacts', exist_ok=True)
  weight_path = os.path.join('artifacts', 'cifar10_cnn.pth')
  torch.save(model.state_dict(), weight_path)
  print(f'Model weights successfully saved to {weight_path}')


if __name__ == '__main__':
  main()