# Time Series Forecasting: LSTM "From Scratch"

This project implements a Deep Learning architecture (LSTM) entirely "from scratch," using only the standard Python library `numpy`.

## Objectives Achieved

- **Modularity:** Separation between mathematical functions, the (local) LSTM cell, and the temporal model (global Backpropagation Through Time, or "BPTT").
- **Informed Initialization:** Use of Xavier/Glorot initialization (suited for Sigmoid and Tanh activation functions) and setting the "forget gate" bias to 1 to mitigate the *vanishing gradient* problem.
- **Ablation Study:** Comparison of convergence dynamics with and without the *forget gate* using a cyclical time series.

## How to Reproduce the Experiment (Live Evaluation)

1. **Prerequisites:** Ensure you have Python installed, along with the necessary core libraries. 
```bash
pip install numpy matplotlib
```

2. **Running the Training and Ablation Study:**
```bash
python main.py
```
*Note: The script will automatically generate the dataset (a sine wave), train both versions of the model (with and without the forget gate), display loss logs (A - Y), and generate a comparison plot.*

## Project Architecture

*   `functions.py`: Activation functions and their analytical derivatives (using the chain rule).
*   `lstm.py`: The LSTM cell, featuring rigorous dimension management, Hadamard products, and local backpropagation.
*   `model.py`: The temporal model (unrolling), accumulation of temporal gradients, and SGD updates (W = W - alpha * dW).
*   `main.py`: Generation of the cyclical dataset, the training loop, and the creation of the comparison plot.
