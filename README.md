# CIFAR-100 Classification and CNN Architectures

This repository contains the implementation and analysis for **Lab Assignment 2** in the course *Image and Video Technology and Analysis*. The lab focuses on training Convolutional Neural Networks (CNNs) using a subset of the CIFAR-100 dataset and comparing the performance of different architectures and configurations.

## Contents

- Theoretical comparison of LeNet, AlexNet, and VGG architectures
- Implementation of CNNs from scratch using PyTorch
- Experimental comparisons with varying architecture depth, dropout, batch size, and more
- Evaluation metrics (accuracy/loss plots)
- Commentary and observations in markdown cells

---

## Part 1 – Theoretical Section

We reviewed three classic CNN architectures used in image recognition:

1. **LeNet** – designed for handwritten digit recognition (MNIST)
2. **AlexNet** – a deeper network that won the ImageNet competition in 2012
3. **VGG** – a very deep architecture based on stacked convolutional blocks

For each architecture, we compared:

- Number and type of layers
- Filter/kernel sizes
- Activation functions used
- Total number of parameters
- Pooling strategies
- Use of dropout

This information was compiled into a comparison table followed by a short analysis of each model's efficiency and accuracy.

---

## Part 2 – Experimental Section

We evaluated the performance of all three architectures (LeNet, AlexNet, VGG) on a custom subset of CIFAR-100. After that, we also designed and tested a custom architecture of our own, aiming to improve performance based on insights gained from the theoretical models.

### Experiments Conducted

1. **Dropout Variation**
   - Compared different dropout rates (e.g., 0.3 vs 0.5) and analyzed the impact on generalization and overfitting.

2. **Depth Variation**
   - Increased the number of convolutional layers to observe the effect of depth on learning capability.

3. **Batch Size Comparison**
   - Tested batch sizes such as 32 and 64 to study training stability and model convergence.

4. **Optimizer Comparison**
   - Compared `SGD` and `Adam` optimizers under identical conditions to evaluate convergence speed and final accuracy.

5. **Weight Initialization**
   - Tried different weight initialization strategies and documented their effect on early training behavior.

6. **Data Augmentation**
   - Applied basic augmentation techniques (e.g., horizontal flips, normalization) to improve generalization.

---

## Evaluation

Each experiment includes:
- Accuracy and loss plots (training vs validation)
- Final test accuracy
- Observations and insights documented in markdown cells

All outputs are included in the notebook for easy inspection.

---

## Observations

- Deeper networks generally performed better up to a point, after which overfitting became noticeable.
- Higher dropout values (0.5) helped mitigate overfitting compared to lower values.
- Batch size 64 produced smoother and more stable training curves than 32.
- Adam converged faster and achieved higher accuracy than SGD in most trials.
- Data augmentation positively impacted validation performance, especially in limited data scenarios.
- Our custom architecture achieved competitive results when designed with a balanced depth and regularization strategy.

---

## Usage

Open the notebook in Google Colab or run it locally using Jupyter Notebook. Ensure the necessary Python packages are installed (`torch`, `torchvision`, `matplotlib`, etc.). All experiments are reproducible within the notebook.

---

## Author

**Nikolaos Katsaidonis**  
National Technical University of Athens – School of Electrical & Computer Engineering
