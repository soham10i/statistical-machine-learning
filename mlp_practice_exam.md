# MLP Practice Exam

## Section 1: Questions

### Conceptual Questions (Easy/Medium)

1.  Why is simply stacking multiple hidden layers without a non-linear activation function not useful for learning complex patterns? Provide a mathematical justification.
2.  Explain the Universal Approximation Theorem. What are two of its major practical limitations?
3.  What is the purpose of the backpropagation algorithm? Why is it preferred over calculating the derivative of the loss function with respect to each weight independently?
4.  You are building an MLP to predict house prices (a regression task). What activation function would you use in the output layer and what loss function would be appropriate? Why?
5.  You are building an MLP to classify images into 10 categories (a multi-class classification task). Describe the setup of your output layer (number of nodes, activation function) and a suitable loss function.

### Calculation Questions (Medium/Hard)

**Question 1 (Network Parameters):**
You design a fully-connected MLP with one input layer, one hidden layer, and one output layer.
*   The input layer has 10 nodes.
*   The hidden layer has 20 nodes.
*   The output layer has 5 nodes.
Calculate the total number of trainable parameters (weights and biases) in this network. Show your calculations for each layer.

**Question 2 (Forward and Backward Pass - Based on Exercise Sheet):**
Consider a 3-input, 2-hidden-node, 1-output MLP.
*   **Network Parameters:**
    *   Weights from input to hidden layer ($W^{(1)}$):
        $W^{(1)} = \begin{pmatrix} 0.1 & 0.4 \\ 0.2 & 0.5 \\ 0.3 & 0.6 \end{pmatrix}$
        (Connecting 3 inputs to 2 hidden nodes. $W_{ij}^{(1)}$ is weight from input $i$ to hidden node $j$)
    *   Weights from hidden to output layer ($W^{(2)}$):
        $W^{(2)} = \begin{pmatrix} 0.7 \\ 0.8 \end{pmatrix}$
        (Connecting 2 hidden nodes to 1 output node. $W_{k}^{(2)}$ is weight from hidden node $k$ to output node)
    *   Assume all biases ($b^{(1)}, b^{(2)}$) are zero.
*   **Activation Functions:**
    *   Hidden layer ($h^{(1)}$): ReLU activation function.
    *   Output layer ($\hat{y}$): Sigmoid activation function.
*   **Given:**
    *   Input vector $x = \begin{pmatrix} 10 \\ 50 \\ -20 \end{pmatrix}$
    *   Ground-truth label $y = 0$.

**Tasks:**
*   **a) (Forward Pass):** Perform a complete forward pass. Calculate the values of $z^{(1)}$ (pre-activation hidden), $h^{(1)}$ (activation hidden), $z^{(2)}$ (pre-activation output), and the final prediction $\hat{y}$.
*   **b) (Loss Calculation):** Calculate the Mean Squared Error (MSE) loss for your prediction. Use the formula $L = \frac{1}{2} (\hat{y} - y)^2$.
*   **c) (Backward Pass):** Perform a complete backward pass to find the gradients of the loss with respect to all weights, i.e., $\frac{\partial L}{\partial W^{(2)}}$ and $\frac{\partial L}{\partial W^{(1)}}$. Show your intermediate calculations for $\delta^{(2)}$ and $\delta^{(1)}$.

## Section 2: Solutions

### Solutions to Conceptual Questions

1.  **Stacking Linear Layers:**
    Stacking multiple linear layers without non-linear activation functions is not useful for learning complex patterns because the composition of multiple linear transformations is itself a single linear transformation.
    *Mathematical Justification:*
    Let $h^{(1)} = W^{(1)T}x + b^{(1)}$ be the output of the first linear layer.
    Let $h^{(2)} = W^{(2)T}h^{(1)} + b^{(2)}$ be the output of the second linear layer.
    Substituting $h^{(1)}$ into the second equation:
    $h^{(2)} = W^{(2)T}(W^{(1)T}x + b^{(1)}) + b^{(2)}$
    $h^{(2)} = (W^{(2)T}W^{(1)T})x + (W^{(2)T}b^{(1)} + b^{(2)})$
    If we define $W_{eff}^T = W^{(2)T}W^{(1)T}$ and $b_{eff} = W^{(2)T}b^{(1)} + b^{(2)}$, then the equation becomes:
    $h^{(2)} = W_{eff}^T x + b_{eff}$
    This is the form of a single linear layer. Thus, no amount of stacking linear layers can introduce non-linearity, which is essential for modeling complex relationships in data.

2.  **Universal Approximation Theorem:**
    The Universal Approximation Theorem states that a feedforward neural network with a single hidden layer containing a finite number of neurons and using a non-linear activation function (like sigmoid or ReLU) can approximate any continuous function on compact subsets of $\mathbb{R}^n$ to an arbitrary degree of accuracy.
    *Major Practical Limitations:*
    *   **Non-Constructive:** The theorem doesn't tell us *how many* neurons are needed for a specific function or accuracy, nor does it provide a method to find the optimal weights and biases. The required number of neurons could be impractically large.
    *   **Optimization Difficulty:** Even if a network with the required capacity exists, training it (i.e., finding the parameters that achieve the desired approximation) can be a very challenging non-convex optimization problem. Deep learning relies on heuristic optimization algorithms like gradient descent, which may not find the global optimum.
    *   (Also, while one layer is sufficient, deeper networks are often more efficient and generalize better).

3.  **Purpose of Backpropagation:**
    The purpose of the backpropagation algorithm is to efficiently compute the gradients (partial derivatives) of the loss function with respect to all the parameters (weights and biases) in a neural network.
    *Why Preferred:*
    Backpropagation is preferred over calculating each derivative independently because it is much more computationally efficient. It applies the chain rule systematically, starting from the output layer and moving backward. This allows it to reuse computations. For example, the gradient of the loss with respect to a weight in an early layer depends on computations involving gradients from later layers. Backpropagation calculates these "error signals" (deltas) once per layer and propagates them backward, avoiding the redundant calculations that would occur if each $\frac{\partial L}{\partial W_{ij}^{(l)}}$ was computed from scratch. The complexity of backpropagation is roughly proportional to the complexity of a forward pass, making it feasible for large networks.

4.  **MLP for House Price Prediction (Regression):**
    *   **Output Layer Activation Function:** **Linear (or Identity) activation function**. This is because house prices are continuous values that can range freely (e.g., from thousands to millions) and are not restricted to a specific interval like [0,1] (sigmoid) or a probability distribution (softmax). A linear activation $f(z)=z$ allows the output node to produce any real-valued number.
    *   **Loss Function:** **Mean Squared Error (MSE)** or Mean Absolute Error (MAE). MSE, $L = \frac{1}{N}\sum (y_i - \hat{y}_i)^2$ (or $L = \frac{1}{2}(y - \hat{y})^2$ for a single sample), is commonly used because it penalizes larger errors more heavily and is differentiable, which is good for gradient-based optimization. MAE, $L = \frac{1}{N}\sum |y_i - \hat{y}_i|$, is less sensitive to outliers.

5.  **MLP for Image Classification (10 Categories):**
    *   **Output Layer Setup:**
        *   **Number of Nodes:** 10 nodes, one for each category. Each node will output a score or probability associated with its corresponding class.
        *   **Activation Function:** **Softmax activation function**. Softmax takes the raw output scores (logits) from the 10 nodes and converts them into a probability distribution, where each output is between 0 and 1, and all 10 outputs sum to 1. This gives the model's confidence for each class.
    *   **Suitable Loss Function:** **Cross-Entropy Loss** (specifically, Categorical Cross-Entropy). This loss function is designed for multi-class classification problems where the output is a probability distribution. It measures the dissimilarity between the predicted probability distribution (from softmax) and the true distribution (which is typically a one-hot encoded vector with 1 for the true class and 0 for others).

### Solutions to Calculation Questions

**Question 1 (Network Parameters):**
*   Input layer: 10 nodes ($n_0 = 10$)
*   Hidden layer: 20 nodes ($n_1 = 20$)
*   Output layer: 5 nodes ($n_2 = 5$)

**Parameters between Input and Hidden Layer:**
*   Weights ($W^{(1)}$): The weight matrix connecting $n_0$ input nodes to $n_1$ hidden nodes will have dimensions $n_0 \times n_1$ (or $n_1 \times n_0$). If $W^{(1)}$ is $n_1 \times n_0$, then number of weights = $n_1 \times n_0 = 20 \times 10 = 200$.
*   Biases ($b^{(1)}$): Each of the $n_1$ hidden nodes has one bias term. Number of biases = $n_1 = 20$.
*   Total parameters for hidden layer = $200 + 20 = 220$.

**Parameters between Hidden and Output Layer:**
*   Weights ($W^{(2)}$): The weight matrix connecting $n_1$ hidden nodes to $n_2$ output nodes will have dimensions $n_1 \times n_2$ (or $n_2 \times n_1$). If $W^{(2)}$ is $n_2 \times n_1$, then number of weights = $n_2 \times n_1 = 5 \times 20 = 100$.
*   Biases ($b^{(2)}$): Each of the $n_2$ output nodes has one bias term. Number of biases = $n_2 = 5$.
*   Total parameters for output layer = $100 + 5 = 105$.

**Total Trainable Parameters:**
Total = Parameters (Hidden Layer) + Parameters (Output Layer)
Total = $220 + 105 = 325$.

**Answer:** The total number of trainable parameters is 325.

---

**Question 2 (Forward and Backward Pass):**
Given:
$x = \begin{pmatrix} 10 \\ 50 \\ -20 \end{pmatrix}$, $y = 0$
$W^{(1)} = \begin{pmatrix} 0.1 & 0.4 \\ 0.2 & 0.5 \\ 0.3 & 0.6 \end{pmatrix}$ (shape 3x2, so $W^{(1)T}$ will be used or inputs are rows of W)
Let's assume the convention $z^{(1)} = x^T W^{(1)}$ if $x$ is a row vector, or $z^{(1)} = W^{(1)T} x$ if $x$ is a column vector and $W^{(1)}$ is (inputs x hidden_nodes).
The problem states $W_{ij}^{(1)}$ is weight from input $i$ to hidden node $j$. This means $W^{(1)}$ is (input_dim x hidden_dim).
So $z^{(1)} = x^T W^{(1)}$ is not standard. Standard for $x$ as column vector is $z^{(1)} = W^{(1)T} x$ or $z^{(1)} = W_{alt}^{(1)} x$ where $W_{alt}^{(1)}$ is (hidden_dim x input_dim).
Given $W^{(1)}$ shape is (3,2), it implies $W^{(1)}_{input\_feature, hidden\_unit}$.
So, $z^{(1)T} = x^T W^{(1)}$ or $z^{(1)} = W^{(1)T} x$.
Let's use $z^{(l)} = W^{(l)T} h^{(l-1)} + b^{(l)}$ where $h^{(l-1)}$ is column vector.
So for the first layer, $z^{(1)} = W^{(1)T} x + b^{(1)}$.
$W^{(1)T} = \begin{pmatrix} 0.1 & 0.2 & 0.3 \\ 0.4 & 0.5 & 0.6 \end{pmatrix}$ (shape 2x3)
$b^{(1)} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$

$W^{(2)} = \begin{pmatrix} 0.7 \\ 0.8 \end{pmatrix}$ (shape 2x1)
This means $W^{(2)T} = \begin{pmatrix} 0.7 & 0.8 \end{pmatrix}$ (shape 1x2)
$b^{(2)} = \begin{pmatrix} 0 \end{pmatrix}$

Activation functions: $f^{(1)}$ = ReLU, $f^{(2)}$ = Sigmoid.

**a) Forward Pass:**

*   **Hidden Layer Pre-activation ($z^{(1)}$):**
    $z^{(1)} = W^{(1)T} x + b^{(1)}$
    $z^{(1)} = \begin{pmatrix} 0.1 & 0.2 & 0.3 \\ 0.4 & 0.5 & 0.6 \end{pmatrix} \begin{pmatrix} 10 \\ 50 \\ -20 \end{pmatrix} + \begin{pmatrix} 0 \\ 0 \end{pmatrix}$
    $z^{(1)}_1 = (0.1 \times 10) + (0.2 \times 50) + (0.3 \times -20) = 1 + 10 - 6 = 5$
    $z^{(1)}_2 = (0.4 \times 10) + (0.5 \times 50) + (0.6 \times -20) = 4 + 25 - 12 = 17$
    $z^{(1)} = \begin{pmatrix} 5 \\ 17 \end{pmatrix}$

*   **Hidden Layer Activation ($h^{(1)}$):** (ReLU)
    $h^{(1)} = \text{ReLU}(z^{(1)})$
    $h^{(1)}_1 = \max(0, 5) = 5$
    $h^{(1)}_2 = \max(0, 17) = 17$
    $h^{(1)} = \begin{pmatrix} 5 \\ 17 \end{pmatrix}$

*   **Output Layer Pre-activation ($z^{(2)}$):**
    $z^{(2)} = W^{(2)T} h^{(1)} + b^{(2)}$
    $z^{(2)} = \begin{pmatrix} 0.7 & 0.8 \end{pmatrix} \begin{pmatrix} 5 \\ 17 \end{pmatrix} + \begin{pmatrix} 0 \end{pmatrix}$
    $z^{(2)} = (0.7 \times 5) + (0.8 \times 17) = 3.5 + 13.6 = 17.1$
    $z^{(2)} = \begin{pmatrix} 17.1 \end{pmatrix}$

*   **Final Prediction ($\hat{y}$):** (Sigmoid)
    $\hat{y} = \sigma(z^{(2)}) = \frac{1}{1 + e^{-17.1}}$
    $e^{-17.1} \approx 3.37 \times 10^{-8}$
    $\hat{y} \approx \frac{1}{1 + 3.37 \times 10^{-8}} \approx \frac{1}{1.0000000337} \approx 0.999999966$
    For simplicity in manual calculation, let's use a more rounded $\hat{y} \approx 1.0$ if $z^{(2)}$ were very large, but $17.1$ is finite.
    $\hat{y} \approx 0.99999997$ (using a calculator)

**Summary of Forward Pass:**
$z^{(1)} = \begin{pmatrix} 5 \\ 17 \end{pmatrix}$
$h^{(1)} = \begin{pmatrix} 5 \\ 17 \end{pmatrix}$
$z^{(2)} = \begin{pmatrix} 17.1 \end{pmatrix}$
$\hat{y} \approx 0.99999997$

**b) Loss Calculation (MSE):**
$L = \frac{1}{2} (\hat{y} - y)^2$
$L = \frac{1}{2} (0.99999997 - 0)^2 = \frac{1}{2} (0.99999997)^2 \approx \frac{1}{2} (0.99999994) \approx 0.49999997$

**c) Backward Pass:**

*   **Error term for Output Layer ($\delta^{(2)}$):**
    $\delta^{(2)} = (\hat{y} - y) \cdot f^{(2)'}(z^{(2)})$
    $f^{(2)}(z) = \sigma(z)$, so $f^{(2)'}(z) = \sigma(z)(1-\sigma(z)) = \hat{y}(1-\hat{y})$
    $\delta^{(2)} = (\hat{y} - y) \cdot \hat{y}(1-\hat{y})$
    $\delta^{(2)} = (0.99999997 - 0) \cdot 0.99999997(1 - 0.99999997)$
    $\delta^{(2)} = 0.99999997 \cdot 0.99999997 \cdot (0.00000003)$
    $\delta^{(2)} \approx 1 \cdot 1 \cdot (3 \times 10^{-8}) = 3 \times 10^{-8}$
    More precisely: $(0.999999966) \times (0.999999966) \times (1 - 0.999999966) \approx 0.999999932 \times 3.37 \times 10^{-8} \approx 3.369999 \times 10^{-8}$
    Let $\delta^{(2)} \approx 3.37 \times 10^{-8}$

*   **Gradients for $W^{(2)}$:**
    $\frac{\partial L}{\partial W^{(2)}} = \delta^{(2)} h^{(1)T}$ (This is for $W^{(2)}$ being $n_2 \times n_1$. Our $W^{(2)}$ is $n_1 \times n_2 = 2 \times 1$. So $\frac{\partial L}{\partial W^{(2)}}$ should be $2 \times 1$)
    The formula for $z^{(2)} = W^{(2)T}h^{(1)}$ means $\frac{\partial L}{\partial W^{(2)T}} = h^{(1)} \delta^{(2)T}$. So $\frac{\partial L}{\partial W^{(2)}} = (h^{(1)} \delta^{(2)T})^T = \delta^{(2)} h^{(1)T}$ if $\delta^{(2)}$ is a row vector.
    If $z^{(2)} = W^{(2)T}h^{(1)}$ (scalar output), $W^{(2)}$ is $n_1 \times 1$ (a column vector), $h^{(1)}$ is $n_1 \times 1$.
    $\frac{\partial z^{(2)}}{\partial W_k^{(2)}} = h_k^{(1)}$. So $\frac{\partial L}{\partial W_k^{(2)}} = \delta^{(2)} h_k^{(1)}$.
    $\frac{\partial L}{\partial W^{(2)}} = h^{(1)} \delta^{(2)}$ (since $\delta^{(2)}$ is scalar here)
    $\frac{\partial L}{\partial W^{(2)}} = \begin{pmatrix} 5 \\ 17 \end{pmatrix} \cdot (3.37 \times 10^{-8})$
    $\frac{\partial L}{\partial W_{1}^{(2)}} = 5 \times 3.37 \times 10^{-8} = 1.685 \times 10^{-7}$
    $\frac{\partial L}{\partial W_{2}^{(2)}} = 17 \times 3.37 \times 10^{-8} = 5.729 \times 10^{-7}$
    $\frac{\partial L}{\partial W^{(2)}} = \begin{pmatrix} 1.685 \times 10^{-7} \\ 5.729 \times 10^{-7} \end{pmatrix}$

*   **Error term for Hidden Layer ($\delta^{(1)}$):**
    $\delta^{(1)} = (W^{(2)} \delta^{(2)}) \odot f^{(1)'}(z^{(1)})$ (General form $W^{(l+1)T}\delta^{(l+1)}$)
    Here, $W^{(2)}$ is $2 \times 1$. $\delta^{(2)}$ is $1 \times 1$. $W^{(2)}\delta^{(2)}$ is $2 \times 1$.
    $f^{(1)'}(z^{(1)})$ is the derivative of ReLU.
    ReLU$' (z) = 1$ if $z > 0$, else $0$.
    $z^{(1)} = \begin{pmatrix} 5 \\ 17 \end{pmatrix}$, so $f^{(1)'}(z^{(1)}) = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$.
    $\delta^{(1)} = \left( \begin{pmatrix} 0.7 \\ 0.8 \end{pmatrix} \cdot (3.37 \times 10^{-8}) \right) \odot \begin{pmatrix} 1 \\ 1 \end{pmatrix}$
    $\delta^{(1)}_1 = 0.7 \times (3.37 \times 10^{-8}) \times 1 = 2.359 \times 10^{-8}$
    $\delta^{(1)}_2 = 0.8 \times (3.37 \times 10^{-8}) \times 1 = 2.696 \times 10^{-8}$
    $\delta^{(1)} = \begin{pmatrix} 2.359 \times 10^{-8} \\ 2.696 \times 10^{-8} \end{pmatrix}$

*   **Gradients for $W^{(1)}$:**
    $\frac{\partial L}{\partial W^{(1)}} = x \delta^{(1)T}$ (This is if $W^{(1)}$ is $n_0 \times n_1$. Our $W^{(1)}$ is $3 \times 2$)
    The formula $z^{(1)} = W^{(1)T} x$ implies $\frac{\partial L}{\partial W^{(1)T}} = x \delta^{(1)T}$.
    So $\frac{\partial L}{\partial W^{(1)}} = (\delta^{(1)} x^T)^T = x \delta^{(1)T}$ if $\delta^{(1)}$ is row vector.
    If $W^{(1)}$ is $n_0 \times n_1$, then $z^{(1)}_j = \sum_k W_{kj}^{(1)} x_k$. $\frac{\partial z_j^{(1)}}{\partial W_{ki}^{(1)}} = x_k \delta_{ji}$.
    The gradient $\frac{\partial L}{\partial W^{(1)}}$ should have the same shape as $W^{(1)}$, which is $3 \times 2$.
    $\frac{\partial L}{\partial W_{ij}^{(1)}} = x_i \delta_j^{(1)}$.
    $\frac{\partial L}{\partial W^{(1)}} = x \delta^{(1)T} = \begin{pmatrix} 10 \\ 50 \\ -20 \end{pmatrix} \begin{pmatrix} 2.359 \times 10^{-8} & 2.696 \times 10^{-8} \end{pmatrix}$
    $\frac{\partial L}{\partial W_{11}^{(1)}} = 10 \times (2.359 \times 10^{-8}) = 2.359 \times 10^{-7}$
    $\frac{\partial L}{\partial W_{12}^{(1)}} = 10 \times (2.696 \times 10^{-8}) = 2.696 \times 10^{-7}$
    $\frac{\partial L}{\partial W_{21}^{(1)}} = 50 \times (2.359 \times 10^{-8}) = 1.1795 \times 10^{-6}$
    $\frac{\partial L}{\partial W_{22}^{(1)}} = 50 \times (2.696 \times 10^{-8}) = 1.348 \times 10^{-6}$
    $\frac{\partial L}{\partial W_{31}^{(1)}} = -20 \times (2.359 \times 10^{-8}) = -4.718 \times 10^{-7}$
    $\frac{\partial L}{\partial W_{32}^{(1)}} = -20 \times (2.696 \times 10^{-8}) = -5.392 \times 10^{-7}$

    $\frac{\partial L}{\partial W^{(1)}} = \begin{pmatrix}
    2.359 \times 10^{-7} & 2.696 \times 10^{-7} \\
    1.1795 \times 10^{-6} & 1.348 \times 10^{-6} \\
    -4.718 \times 10^{-7} & -5.392 \times 10^{-7}
    \end{pmatrix}$

**Summary of Backward Pass Gradients:**
$\delta^{(2)} \approx 3.37 \times 10^{-8}$
$\frac{\partial L}{\partial W^{(2)}} = \begin{pmatrix} 1.685 \times 10^{-7} \\ 5.729 \times 10^{-7} \end{pmatrix}$
$\delta^{(1)} = \begin{pmatrix} 2.359 \times 10^{-8} \\ 2.696 \times 10^{-8} \end{pmatrix}$
$\frac{\partial L}{\partial W^{(1)}} = \begin{pmatrix}
2.359 \times 10^{-7} & 2.696 \times 10^{-7} \\
1.1795 \times 10^{-6} & 1.348 \times 10^{-6} \\
-4.718 \times 10^{-7} & -5.392 \times 10^{-7}
\end{pmatrix}$
(Biases were zero, so gradients for biases are not explicitly asked but would be $\delta^{(2)}$ and $\delta^{(1)}$ respectively.)
