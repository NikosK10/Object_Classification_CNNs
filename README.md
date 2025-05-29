# CIFAR-100 Classification and CNN Architectures

This repository contains the implementation and analysis for **Lab Assignment 2** in the course *Image and Video Technology and Analysis*. The lab focuses on training Convolutional Neural Networks (CNNs) using a subset of the CIFAR-100 dataset and comparing the performance of different architectures and configurations.

## 📁 Contents

- 📄 Theoretical comparison of LeNet, AlexNet, and VGG architectures
- 🧠 Implementation of CNNs from scratch using PyTorch
- 🔁 Experimental comparisons with varying architecture depth, dropout, batch size, and more
- 📊 Evaluation metrics (accuracy/loss plots)
- 📝 Commentary and observations in markdown cells

---

## 📌 Part 1 – Theoretical Section

We reviewed three classic CNN architectures used in image recognition:

1. **LeNet** (Digit recognition, MNIST)
2. **AlexNet** (ImageNet Classification)
3. **VGG** (Deep convolutional blocks)

For each architecture, we compared:

- Number and type of layers
- Filter/kernel sizes
- Activation functions used
- Total number of parameters
- Pooling strategies
- Use of dropout

This was summarized in a comparative table with a brief discussion on performance and efficiency.

---

## 🧪 Part 2 – Experimental Section

The implementation used a custom CIFAR-100 subset derived via a `team_seed` parameter set to the last 3 digits of my student ID.

### ✅ Baseline Model

- 3 convolutional layers with ReLU activation
- MaxPooling and Dropout layers
- Fully connected classifier
- Trained for 20 epochs

### 🔄 Experiments Conducted

1. **Dropout Variation**
   - Compared different dropout rates (e.g., 0.3 vs 0.5) and their effect on overfitting

2. **Depth Variation**
   - Increased the number of convolutional layers to examine improvements in feature learning

3. **Batch Size Comparison**
   - Trained the same model with different batch sizes (32 vs 64) to observe training stability and generalization

4. **Optimizer Comparison**
   - Tested `SGD` vs `Adam` optimizers under same conditions

5. **Weight Initialization**
   - Applied different initialization schemes and discussed their impact

6. **Data Augmentation**
   - Introduced transformations such as horizontal flip and normalization to boost model robustness

---

## 📈 Evaluation

Each experiment includes:
- Accuracy and loss plots (train vs validation)
- Final test accuracy
- Observations and conclusions in markdown form

Metrics and figures are printed and visualized directly within the notebook.

---

## 🧠 Observations

- Deeper architectures improved accuracy up to a point, beyond which overfitting appeared.
- Dropout 0.5 performed better in preventing overfitting compared to 0.3.
- Batch size 64 achieved smoother loss convergence than 32.
- Adam optimizer converged faster than SGD, especially in early epochs.
- Data augmentation improved generalization, particularly on smaller training sets.

---

## 🧾 Usage

Open the notebook in Google Colab or locally with Jupyter and run all cells. Make sure to adjust the path to your CIFAR-100 subset and ensure all dependencies are installed (e.g., `torch`, `torchvision`, `matplotlib`).

---

## 👤 Author

**Nikolaos Katsaidonis**  
Student ID: 03121868  
National Technical University of Athens – School of Electrical & Computer Engineering
