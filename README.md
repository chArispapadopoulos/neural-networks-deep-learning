# Neural Networks & Deep Learning – Course Projects

Three assignments for the Neural Networks – Deep Learning course, Aristotle University of Thessaloniki (2025-26). The intermediate and 1st assignments share one report; the 2nd and 3rd have their own.

## 1. Deep CNN on CIFAR-10 – `Neural Networks Intermediate and 1st Assignment - Report.pdf`
The starting point is a set of classical baselines on the raw CIFAR-10 pixels (k-NN and Nearest Centroid with scikit-learn):

| Classifier | Test accuracy |
|---|---|
| k-NN, k = 1 | 35.39% |
| k-NN, k = 3 | 33.03% |
| Nearest Centroid | 27.74% |

On top of that, a PyTorch CNN trained with back-propagation, built up step by step through an ablation study (MLP → simple CNN → deeper blocks → BatchNorm → Dropout / weight decay → LR scheduler → data augmentation → wider and deeper network).

- Final model: 3 convolutional blocks (64 → 128 → 256 filters), BatchNorm, ReLU, MaxPool, Dropout 0.3/0.3/0.4 in the blocks and 0.5 in the dense part, a 512-unit dense layer.
- Training: Adam (weight decay 1e-5), batch 64, CosineAnnealingLR, 80 epochs, RandomHorizontalFlip + RandomCrop + RandomErasing.
- **Test accuracy 92.58%** (vs 35.39% for the best classical baseline).

| Step in the ablation | Test accuracy |
|---|---|
| MLP (1 hidden layer) | 46.67% |
| Single-layer CNN | 59.54% |
| 2 conv blocks | 69.19% |
| + BatchNorm | 74.00% |
| + more filters | 79.13% |
| + Dropout, lower LR | 82.27% |
| + weight decay | 84.15% |
| + ReduceLROnPlateau | 84.50% |
| + data augmentation | 86.24% |
| final 3-block model | 92.58% |

## 2. SVMs: animals vs vehicles on CIFAR-10 – `Neural Networks 2nd Assignment - Report.pdf`
Binary task (6 animal classes vs 4 vehicle classes) with GPU SVMs (cuML), class balancing and 5-fold GridSearchCV over linear, polynomial and RBF kernels. Compared against 1-NN, 3-NN, Nearest Centroid and an MLP with hinge loss.

| Model | Test accuracy |
|---|---|
| SVM, RBF kernel (C = 3.94, γ = 0.00056) | 90.18% |
| SVM, polynomial (degree 3, coef0 = 1) | 89.68% |
| MLP, hinge loss | 87.00% |
| Linear SVM | 81.68% |
| 3-NN / 1-NN | 80.75% / 79.93% |
| Nearest Centroid | 70.65% |

## 3. Generative MLP autoencoder on MNIST – `Neural Networks 3rd Assignment - Report.pdf`
An encoder–decoder MLP that takes an image of digit *x* and generates an image of digit *x + 1*. A separate "judge" CNN classifies the generated images to measure how recognisable they are.

- Compared bottleneck sizes (16 → 128), MSE vs BCE loss, ReLU vs Tanh, CosineAnnealing vs ReduceLROnPlateau, random vs fixed targets, with 5-fold cross-validation.
- Best: "Super" model (784 → 1024 → 512 → 256 → 128), BCE loss, ReLU, Cosine Annealing: **98.34%** judge accuracy.
- Comparison with PCA at the same latent size: with 16 components PCA reaches 78.07% even though it only copies the input, while the 16-neuron MLP reaches 95.64% on the harder N → N+1 task.

## Files
```
Neural Networks 1st Assignment - Code.ipynb                   CNN on CIFAR-10
Neural Networks 2nd Assignment - Code.ipynb                   SVM experiments
Neural Networks 3rd Assignment - Code.ipynb                   autoencoder, judge CNN, PCA comparison
Neural Networks Intermediate and 1st Assignment - Report.pdf  report for assignment 1
Neural Networks 2nd Assignment - Report.pdf                   report for assignment 2
Neural Networks 3rd Assignment - Report.pdf                   report for assignment 3
```

## How to run
The notebooks were run on Google Colab with a GPU. The SVM notebook needs RAPIDS cuML.
```bash
pip install torch torchvision scikit-learn numpy pandas matplotlib seaborn
```
CIFAR-10 and MNIST are public datasets and are not included in this repository. The notebooks download them through torchvision or read the official CIFAR-10 python batches (https://www.cs.toronto.edu/~kriz/cifar.html).

## Tech
Python, PyTorch, scikit-learn, cuML, NumPy, pandas, Matplotlib.