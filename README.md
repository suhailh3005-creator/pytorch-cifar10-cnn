# PyTorch CIFAR-10 CNN

A small convolutional neural network trained on CIFAR-10 with PyTorch. Training saves a state-dict checkpoint; evaluation reports per-class precision, recall, and F1, and writes a confusion-matrix image.

## Project Layout

- `src/model.py`: reusable `SimpleCNN` architecture.
- `src/train.py`: CIFAR-10 training data, loss, optimizer, and training loop.
- `src/evaluate.py`: test-set metrics and confusion-matrix generation.
- `data/`: downloaded CIFAR-10 data; excluded from Git.
- `artifacts/`: model checkpoint and generated plot.

## Setup

Python 3.10 or newer is recommended. From this directory, create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

## Train and Evaluate

Run both commands from the project root. Training downloads CIFAR-10 into `data/` when needed and saves weights to `artifacts/cifar10_cnn.pth`.

```bash
python -m src.train
python -m src.evaluate
```

The evaluator writes `artifacts/confusion_matrix.png`. The training and evaluation scripts automatically select CUDA when available and otherwise use CPU.

## Model

The network uses two padded 3x3 convolution layers (16 and 32 channels), each followed by ReLU and 2x2 max pooling. A 128-unit fully connected layer and a 10-class output layer produce CIFAR-10 logits. Inputs are converted to tensors and normalized per RGB channel to mean and standard deviation 0.5. Training uses cross-entropy loss, Adam with learning rate 0.001, batch size 32, and three epochs.

## Evaluation Results

Results below are from the validated three-epoch CPU run with seed 42 on the 10,000-image CIFAR-10 test split. Results can vary across devices and library versions.

| Class | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| Plane | 0.7525 | 0.6840 | 0.7166 |
| Car | 0.8782 | 0.6200 | 0.7268 |
| Bird | 0.5288 | 0.5690 | 0.5482 |
| Cat | 0.4949 | 0.5330 | 0.5132 |
| Deer | 0.5919 | 0.6250 | 0.6080 |
| Dog | 0.6386 | 0.5230 | 0.5750 |
| Frog | 0.7490 | 0.7220 | 0.7352 |
| Horse | 0.7568 | 0.7530 | 0.7549 |
| Ship | 0.7509 | 0.8140 | 0.7812 |
| Truck | 0.6332 | 0.8320 | 0.7191 |
| Accuracy |  |  | 0.6675 |
| Macro average | 0.6775 | 0.6675 | 0.6678 |
| Weighted average | 0.6775 | 0.6675 | 0.6678 |