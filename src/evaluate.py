# src/evaluate.py
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

try:
  from .model import SimpleCNN
except ImportError:
  from model import SimpleCNN

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    classification_report,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = (
    'plane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
)


def main():
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  testset = torchvision.datasets.CIFAR10(
      root='./data', train=False, download=True, transform=transform
  )
  testloader = torch.utils.data.DataLoader(
      testset, batch_size=32, shuffle=False, num_workers=2
  )

  model = SimpleCNN().to(device)
  weight_path = os.path.join('artifacts', 'cifar10_cnn.pth')

  if not os.path.exists(weight_path):
    raise FileNotFoundError(
        f'Weight file not found at {weight_path}. Run train.py first.'
    )

  model.load_state_dict(torch.load(weight_path, map_location=device))
  model.eval()

  all_preds = []
  all_targets = []

  with torch.no_grad():
    for inputs, labels in testloader:
      inputs = inputs.to(device)
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)

      all_preds.extend(preds.cpu().numpy())
      all_targets.extend(labels.numpy())

  print('\n--- Detailed Classification Report ---')
  print(
      classification_report(
          all_targets, all_preds, target_names=classes, digits=4
      )
  )

  cm = confusion_matrix(all_targets, all_preds)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

  fig, ax = plt.subplots(figsize=(10, 10))
  disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
  plt.title('CIFAR-10 CNN Confusion Matrix')
  plt.tight_layout()

  plot_path = os.path.join('artifacts', 'confusion_matrix.png')
  plt.savefig(plot_path, dpi=300)
  print(f'Saved confusion matrix plot to {plot_path}')


if __name__ == '__main__':
  main()