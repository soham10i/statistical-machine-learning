# Deep Learning: Multi-Layer Perceptron (MLP)

## Overview

This document provides a comprehensive overview of Multi-Layer Perceptrons (MLPs), covering the following key topics:
*   Limitations of linear models and the need for MLPs.
*   The crucial role of non-linear activation functions.
*   Mathematical demonstration that stacking linear layers is insufficient.
*   Commonly used activation functions: Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Swish, GELU.
*   Architecture of MLPs: Input, Hidden, and Output Layers.
*   Forward propagation in an L-layer network.
*   Tailoring MLP architecture for classification and regression tasks.
*   The Universal Approximation Theorem and its implications.
*   Model training using gradient-based optimization and the Backpropagation algorithm.
*   Detailed mathematical derivation of backpropagation for a 2-layer network.
*   General backpropagation equations for an L-layer network.

## From Softmax Regression to MLPs

Linear models like Softmax Regression (for classification) or Linear Regression (for regression) are fundamental in machine learning. They model a direct, linear relationship between the input features and the output. For an input vector $x$, a linear model computes an output $y$ (or scores for classes) using a linear transformation: $y = W^T x + b$, where $W$ represents the weights and $b$ is the bias.

**Limitation of Linear Models:**
The primary limitation of these models is that they can only capture **affine-linear relationships**. If the underlying relationship between the input features and the target variable is non-linear, linear models will perform poorly as they lack the capacity to represent such complexities.

**Stacking Layers for Increased Complexity:**
To overcome this limitation and enable models to learn more complex, non-linear patterns, the concept of stacking layers is introduced. The idea is that by adding more layers of computation, the model can build up more intricate representations of the data. A Multi-Layer Perceptron (MLP) is formed by stacking one or more "hidden" layers between the input and output layers.

**Crucial Point: Stacking Linear Layers Results in an Equivalent Linear Layer**

A common misconception might be that simply adding more linear layers (i.e., layers without a non-linear activation function) will increase the model's expressive power to capture non-linearities. However, this is not the case. Stacking multiple linear layers is mathematically equivalent to a single linear layer.

Let's demonstrate this for a two-layer network.
Consider an input vector $x$.
The output of the first linear layer, $h^{(1)}$, is:
$h^{(1)} = W^{(1)T} x + b^{(1)}$

Now, if we feed this output $h^{(1)}$ into a second linear layer to get the final output $y$:
$y = W^{(2)T} h^{(1)} + b^{(2)}$

Substitute the expression for $h^{(1)}$ into the second equation:
$y = W^{(2)T} (W^{(1)T} x + b^{(1)}) + b^{(2)}$
$y = W^{(2)T} W^{(1)T} x + W^{(2)T} b^{(1)} + b^{(2)}$

Let $W_{eff}^T = W^{(2)T} W^{(1)T}$. This means $W_{eff} = (W^{(1)} W^{(2)})^T$.
And let $b_{eff} = W^{(2)T} b^{(1)} + b^{(2)}$.

Then the equation for $y$ becomes:
$y = W_{eff}^T x + b_{eff}$

This final equation is in the same form as a single linear layer ($Wx+b$ or $W^Tx+b$). Therefore, no matter how many linear layers we stack, the resulting model is still just a linear model. It cannot learn non-linear relationships. This highlights the necessity of introducing non-linearity into the network, which is achieved through activation functions.

## The Role of Non-Linear Activation Functions

To enable MLPs to model complex, non-linear relationships, a **non-linear activation function** is applied after the linear transformation in each hidden layer (and often in the output layer, depending on the task).

If $z^{(l)}$ is the linear output of layer $l$ (where $h^{(l-1)}$ is the output of the previous layer), computed as $z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}$ (using the convention where $W^{(l)}$ is $n_l \times n_{l-1}$), the activation function $f^{(l)}$ is applied to $z^{(l)}$ to produce the layer's output $h^{(l)}$:
$h^{(l)} = f^{(l)}(z^{(l)})$

**Desirable Properties for Activation Functions:**
When choosing an activation function, several properties are desirable:
1.  **Non-linearity:** This is crucial, as discussed above, to allow the network to learn complex patterns.
2.  **Differentiability:** Activation functions need to be differentiable (or at least mostly differentiable) to enable gradient-based optimization methods like backpropagation. The derivative allows us to understand how changes in the function's input affect its output, which is essential for updating the model's weights.
3.  **Easily Computable Derivatives:** Since derivatives are calculated frequently during training, it's important that they are computationally efficient to calculate.
4.  **Avoidance of Saturation:** Some activation functions (like sigmoid and tanh) saturate for very large positive or negative inputs, meaning their output becomes flat. In these flat regions, the gradient is close to zero. This can lead to the "vanishing gradient" problem.
5.  **Zero-Centered Output (for some functions):** Functions like tanh that produce outputs centered around zero can sometimes help in speeding up convergence.
6.  **Non-dying Neurons:** Functions like ReLU can cause neurons to "die" if their input consistently falls into a region where the gradient is zero. Variants like Leaky ReLU address this.

### Commonly Used Activation Functions

Here are some commonly used activation functions in deep learning:

#### 1. Sigmoid (Logistic)
*   **Mathematical Formula:**
    $\sigma(x) = \frac{1}{1 + e^{-x}}$
*   **Derivative's Formula:**
    $\sigma'(x) = \sigma(x)(1 - \sigma(x))$
*   **Characteristics:**
    -   Outputs values between 0 and 1.
    -   Used for binary classification output probability.
    -   Suffers from vanishing gradients due to saturation.
    -   Output is not zero-centered.

#### 2. Hyperbolic Tangent (Tanh)
*   **Mathematical Formula:**
    $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$
*   **Derivative's Formula:**
    $\tanh'(x) = 1 - \tanh^2(x)$
*   **Characteristics:**
    -   Outputs values between -1 and 1.
    -   Zero-centered output.
    -   Still suffers from saturation, but generally preferred over sigmoid in hidden layers if saturation is a concern and zero-centering is desired.

#### 3. Rectified Linear Unit (ReLU)
*   **Mathematical Formula:**
    $\text{ReLU}(x) = \max(0, x)$
*   **Derivative's Formula:**
    $\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \le 0 \end{cases}$
*   **Characteristics:**
    -   Non-linear, computationally efficient.
    -   Alleviates vanishing gradient for $x>0$.
    -   Can suffer from "dying ReLU" problem (neurons get stuck at output 0).
    -   Not zero-centered. Most popular activation for hidden layers.

#### 4. Leaky ReLU (LReLU)
*   **Mathematical Formula:**
    $\text{LReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \le 0 \end{cases}$ (e.g., $\alpha=0.01$)
*   **Derivative's Formula:**
    $\text{LReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha & \text{if } x \le 0 \end{cases}$
*   **Characteristics:**
    -   Fixes dying ReLU by allowing a small non-zero gradient for negative inputs.
    -   Parametric ReLU (PReLU) learns $\alpha$.

#### 5. Exponential Linear Unit (ELU)
*   **Mathematical Formula:**
    $\text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha (e^x - 1) & \text{if } x \le 0 \end{cases}$ (e.g., $\alpha=1.0$)
*   **Derivative's Formula:**
    $\text{ELU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha e^x & \text{if } x \le 0 \end{cases}$ (which is $\text{ELU}(x) + \alpha$ for $x \le 0$)
*   **Characteristics:**
    -   Outputs can be negative, closer to zero-mean.
    -   More computationally expensive than ReLU.
    -   Can lead to faster learning and better generalization.

#### 6. Swish (SiLU - Sigmoid Linear Unit when $\beta=1$)
*   **Mathematical Formula:**
    $\text{Swish}(x) = x \cdot \sigma(\beta x) = \frac{x}{1 + e^{-\beta x}}$
*   **Derivative's Formula:**
    $\text{Swish}'(x) = \text{Swish}(x) + \sigma(\beta x)(1 - \text{Swish}(x))$
*   **Characteristics:**
    -   Smooth, non-monotonic. Outperforms ReLU on deeper models.
    -   More computationally expensive.

#### 7. Gaussian Error Linear Unit (GELU)
*   **Mathematical Formula:**
    $\text{GELU}(x) = x \cdot \Phi(x)$, where $\Phi(x)$ is the CDF of the standard normal distribution.
    Approximation: $0.5x \left(1 + \tanh\left[\sqrt{2/\pi}(x + 0.044715x^3)\right]\right)$
*   **Derivative's Formula:**
    $\text{GELU}'(x) = \Phi(x) + x \phi(x)$, where $\phi(x)$ is PDF of standard normal.
*   **Characteristics:**
    -   Smooth approximation of ReLU, non-monotonic.
    -   Used in models like BERT, GPT. Computationally expensive.

## MLP Architecture and Forward Propagation

An MLP consists of:
1.  **Input Layer ($h^{(0)} = x$):** Receives input features.
2.  **Hidden Layers ($l=1, \dots, L-1$):** Perform transformations.
3.  **Output Layer ($l=L$):** Produces the final prediction.

**Forward Propagation in an L-Layer Network (Slide 12 convention):**
*   $h^{(0)} = x$: Input vector (shape $n_0 \times 1$).
*   For each layer $l$ from $1$ to $L$:
    *   $W^{(l)}$: Weight matrix for layer $l$ (shape $n_l \times n_{l-1}$).
    *   $b^{(l)}$: Bias vector for layer $l$ (shape $n_l \times 1$).
    *   $z^{(l)}$: Pre-activation (linear combination) for layer $l$:
        $z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}$ (shape $n_l \times 1$)
    *   $f^{(l)}(\cdot)$: Activation function for layer $l$.
    *   $h^{(l)}$: Activation (output) of layer $l$:
        $h^{(l)} = f^{(l)}(z^{(l)})$ (shape $n_l \times 1$)

The final output of the network is $\hat{y} = h^{(L)}$.

**Task-Dependent Architecture and Output Activation:**

**1. MLP for Classification (N classes):**
*   **Output Layer:** $N$ nodes.
*   **Activation Function:** **Softmax** function:
    $\text{Softmax}(z_i^{(L)}) = \frac{e^{z_i^{(L)}}}{\sum_{j=1}^{N} e^{z_j^{(L)}}}$ for $i=1, \dots, N$.
    Converts logits $z^{(L)}$ into a probability distribution.
*   **Loss Function:** **Cross-Entropy Loss**. For true one-hot label $y$ and predicted probabilities $\hat{y}$:
    $L = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)$

**2. MLP for Regression (single output):**
*   **Output Layer:** One node (or more for multi-target regression).
*   **Activation Function:** **Linear (identity)** function, $f(z) = z$.
*   **Loss Function:** **Mean Squared Error (MSE)**. For true value $y$ and prediction $\hat{y}$:
    $L = \frac{1}{2} (\hat{y} - y)^2$

## Universal Approximation Theorem

**Statement:**
A feedforward neural network with a single hidden layer containing a finite number of neurons and using a non-linear "squashing" activation function (like sigmoid) can approximate any continuous function on compact subsets of $\mathbb{R}^n$ to any desired degree of accuracy. (ReLU and other non-polynomial activations also work).

**Important Caveats:**
1.  **Not Constructive:** Doesn't say how many neurons are needed or how to find weights.
2.  **Approximation Quality:** Fixed neuron count doesn't guarantee quality.
3.  **Deeper Networks Often Better:** Deep networks can be more efficient and generalize better than shallow, wide ones.
4.  **Optimization Challenges:** Finding optimal weights is hard.
5.  **Primary Impact:** Boosted confidence in neural networks' capabilities.

## Model Training: Backpropagation

Models are trained by minimizing a loss function $L$ using gradient-based optimization.
**Backpropagation** is an algorithm to efficiently compute partial derivatives $\frac{\partial L}{\partial W_{ij}^{(l)}}$ and $\frac{\partial L}{\partial b_i^{(l)}}$ using the chain rule, propagating errors backward from the output layer.

### Detailed Mathematical Derivation (2-Layer Network, MSE Loss, Slide 23)

Consider a 2-layer MLP:
*   Input $x = h^{(0)}$
*   Layer 1: $z^{(1)} = W^{(1)}x + b^{(1)}$, $h^{(1)} = f^{(1)}(z^{(1)})$
*   Layer 2 (Output): $z^{(2)} = W^{(2)}h^{(1)} + b^{(2)}$, $\hat{y} = h^{(2)} = f^{(2)}(z^{(2)})$
*   Loss: $L = \frac{1}{2} (\hat{y} - y)^2$ (for scalar output $y, \hat{y}$)

**Intermediate Error Term Definition:** $\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}}$

**1. Gradients for Output Layer ($W^{(2)}, b^{(2)}$):**
*   $\delta^{(2)} = \frac{\partial L}{\partial z^{(2)}} = \frac{\partial L}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial z^{(2)}} = (\hat{y} - y) \cdot f^{(2)'}(z^{(2)})$
*   $\frac{\partial L}{\partial W_{ij}^{(2)}} = \frac{\partial L}{\partial z_i^{(2)}} \frac{\partial z_i^{(2)}}{\partial W_{ij}^{(2)}} = \delta_i^{(2)} h_j^{(1)}$.
    In vector form: $\frac{\partial L}{\partial W^{(2)}} = \delta^{(2)} (h^{(1)})^T$
    (Assuming $W^{(2)}$ is $n_2 \times n_1$, $\delta^{(2)}$ is $n_2 \times 1$, $h^{(1)}$ is $n_1 \times 1$)
*   $\frac{\partial L}{\partial b^{(2)}} = \delta^{(2)}$

**2. Gradients for Hidden Layer ($W^{(1)}, b^{(1)}$):**
*   To find $\delta^{(1)} = \frac{\partial L}{\partial z^{(1)}}$:
    $\delta^{(1)}_k = \sum_i \frac{\partial L}{\partial z_i^{(2)}} \frac{\partial z_i^{(2)}}{\partial h_k^{(1)}} \frac{\partial h_k^{(1)}}{\partial z_k^{(1)}}$
    $\frac{\partial z_i^{(2)}}{\partial h_k^{(1)}} = W_{ik}^{(2)}$ (element $i,k$ of $W^{(2)}$)
    $\frac{\partial h_k^{(1)}}{\partial z_k^{(1)}} = f^{(1)'}(z_k^{(1)})$
    So, $\delta^{(1)}_k = \left( \sum_i W_{ik}^{(2)} \delta_i^{(2)} \right) f^{(1)'}(z_k^{(1)})$
    In vector form: $\delta^{(1)} = (W^{(2)T} \delta^{(2)}) \odot f^{(1)'}(z^{(1)})$
    (where $\odot$ is element-wise product. $W^{(2)T}$ is $n_1 \times n_2$, $\delta^{(2)}$ is $n_2 \times 1$, result is $n_1 \times 1$)
*   $\frac{\partial L}{\partial W_{kj}^{(1)}} = \frac{\partial L}{\partial z_k^{(1)}} \frac{\partial z_k^{(1)}}{\partial W_{kj}^{(1)}} = \delta_k^{(1)} x_j$.
    In vector form: $\frac{\partial L}{\partial W^{(1)}} = \delta^{(1)} x^T$
    (Assuming $W^{(1)}$ is $n_1 \times n_0$, $\delta^{(1)}$ is $n_1 \times 1$, $x$ is $n_0 \times 1$)
*   $\frac{\partial L}{\partial b^{(1)}} = \delta^{(1)}$

This shows $\delta^{(1)}$ can be calculated recursively from $\delta^{(2)}$.

### General Backpropagation Equations (L-Layer Network, Single Sample - Slide 25)

For a single training sample $(x,y)$:
1.  **Forward Pass:** Compute all $z^{(l)}$ and $h^{(l)}$ for $l=1, \dots, L$. $h^{(0)}=x$.
2.  **Backward Pass:**
    *   **Output Layer ($l=L$):**
        $\delta^{(L)} = \frac{\partial L}{\partial h^{(L)}} \odot f^{(L)'}(z^{(L)})$
        (Note: For MSE, $\frac{\partial L}{\partial h^{(L)}} = h^{(L)} - y$. For Softmax output layer with Cross-Entropy loss, this entire term $\delta^{(L)}$ conveniently simplifies to $h^{(L)} - y$, where $y$ is the one-hot encoded true label vector.)
    *   **Hidden Layers ($l = L-1, \dots, 1$):** (Iterating backwards)
        $\delta^{(l)} = (W^{(l+1)T} \delta^{(l+1)}) \odot f^{(l)'}(z^{(l)})$
    *   **Gradients:** For $l=L, \dots, 1$:
        $\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (h^{(l-1)})^T$
        $\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$

These gradients are then used to update parameters, e.g., $W^{(l)} \leftarrow W^{(l)} - \eta \frac{\partial L}{\partial W^{(l)}}$, where $\eta$ is the learning rate.
For batch training (Batch Size $B > 1$), the gradients $\frac{\partial L}{\partial W^{(l)}}$ and $\frac{\partial L}{\partial b^{(l)}}$ are typically averaged over all $B$ samples in the batch before the parameter update step.
