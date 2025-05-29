
# CIFAR-100 Classification and CNN Architectures

This repository presents the work conducted for **Lab Assignment 2** in the course *Image and Video Technology and Analysis*, focused on Convolutional Neural Networks (CNNs) and their performance on a subset of the CIFAR-100 dataset.

## Overview

The assignment is structured into two main parts:

- **Part 1: Theoretical Analysis** of classic CNN architectures
- **Part 2: Experimental Evaluation** of those architectures and a custom-designed one

We implemented, trained, and evaluated several models using the same conditions to ensure fair comparisons. The core objective was to understand how different design choices affect the performance and generalization of CNNs.

---

## Part 1 – Theoretical Section

In the first section, we analyzed three foundational CNN models:

1. **LeNet** – a shallow network originally designed for digit classification.
2. **AlexNet** – a deeper model that popularized CNNs through its success in ImageNet 2012.
3. **VGG** – a very deep and regular architecture using stacked 3×3 convolutions.

For each model, we provided:
- Detailed layer-by-layer structure
- Parameter count, activation functions, and pooling strategies
- Data augmentation and regularization techniques
- Reported performance on datasets like ImageNet or MNIST

This section includes comparison tables in markdown and extensive commentary to highlight each model’s strengths and limitations.

---

## Part 2 – Experimental Section

In this part, we used a subset of CIFAR-100 derived with a `team_seed` based on the last digits of the student ID. We **trained and compared all three architectures (LeNet, AlexNet, VGG)** under common hyperparameter settings and then **developed our own custom CNN (MyCNN)** tailored to the dataset constraints.

### Architectures Implemented
- LeNet
- AlexNet
- VGG-like variant (adapted for 32×32 input size)
- **MyCNN** – A custom model with 3 convolutional blocks and 2 FC layers

We systematically tested each of the above, and based on performance, we refined **MyCNN** using overfitting mitigation techniques and transfer learning.

---

## Experimental Techniques

We explored and analyzed the effects of the following factors:

1. **Overfitting Control**
   - Dropout (0.3, 0.5)
   - Data Augmentation (horizontal flips, brightness, contrast, etc.)

2. **Architecture Depth**
   - Shallow vs. deeper convolutional stacks

3. **Batch Size**
   - Comparisons of batch sizes (e.g., 32 vs 64) and their impact on convergence

4. **Optimizer Comparison**
   - Training with `Adam` vs `SGD` under identical setups

5. **Transfer Learning**
   - Using **VGG19** and **EfficientNetB0** backbones
   - Two cases:
     - Freezing all convolutional layers and training only the classification head
     - Fine-tuning last layers along with the head

---

## Evaluation

Each model was evaluated using:
- **F1-Score** as the main metric (both validation and test sets)
- **Training vs Validation performance plots**
- Comparative discussions of training behavior, convergence, and generalization

All results are included in the notebook with visualizations and markdown commentary.

---

## Key Observations

- **Deeper ≠ Better**: VGG performed well, but overfitted easily on the limited dataset. LeNet was more stable, but underfit.
- **Custom model (MyCNN)** achieved the best balance between expressiveness and generalization.
- **Data augmentation and dropout** significantly improved validation performance.
- **Transfer learning**, especially when freezing convolutional layers, gave excellent F1 scores even with little training.
- **Fine-tuning** performed worse than freezing, unless more epochs were allowed.

---

## How to Run

You can run the notebook in **Google Colab** or locally using **Jupyter Notebook**.

Required packages:
```bash
torch
torchvision
numpy
matplotlib
scikit-learn

## !!!
The project was created as part of the course "Image and video Processing" at School of Electrical and Computer Engineering, NTUA and the aim of this work is NOT to present it as my own research but as my approach to the problems given.

