## 4.1 Feature Projection Mathematical Relations Determination

To design an efficient and effective converter that maps CNN-extracted image features to CLIP's embedding space, we first empirically determine the nature of the projection relationship—specifically, whether it is linear or nonlinear. This experiment compares linear regression, simple nonlinear (MLP) modeling, and kernel canonical correlation analysis (KCCA) across multiple kernels to assess their explanatory capabilities.

### Linear Regression
We conducted linear regression analysis to evaluate whether a simple linear model could adequately capture the mapping between ResNet-18 and ViT-B/32 CLIP embeddings. The linear regression yielded a Mean Squared Error (MSE) of 0.001037 and an R² of -0.4661. R² measures the proportion of variance explained relative to a baseline predictor that always returns the mean; a negative R² thus indicates performance worse than predicting the mean, revealing that the linear projection not only fails to capture the mapping but also generalizes poorly.

### Simple MLP Approximation
A basic Multi-Layer Perceptron (MLP) with two hidden layers sized 1024 and 512, each followed by SiLU activations, was employed to balance representational capacity with computational cost. This architecture was chosen to provide sufficient nonlinear modeling power without excessive overfitting risk. The model achieved an MSE of 0.000546 and an R² of 0.2141. Therefore, the improvement to a positive R² (~0.21) signifies the presence of significant nonlinear components in the feature relationship, validating that nonlinear mappings are more suitable.

### Kernel Canonical Correlation Analysis (KCCA)
To compare various nonlinear mapping capabilities, we implemented KCCA with multiple kernel functions. Kernel hyperparameters—specifically gamma and coef0—were selected via five-fold cross-validation over a logarithmic grid (gamma ∈ {1e-3, 1e-2, 1e-1, 1, 10}, coef0 ∈ {0, 1, 10}). The optimal γ and coef0 were chosen based on maximum canonical correlation on validation splits.

```python
kernel_funcs = {
    'Linear': linear_kernel,
    'Polynomial (d=3)': lambda A, B=None: polynomial_kernel(A, B, degree=3, gamma=1.0, coef0=1),
    # ... higher-degree polynomials ...
    'RBF': lambda A, B=None: rbf_kernel(A, B, gamma=1.0 / X_train.shape[1]),
    'Sigmoid': lambda A, B=None: sigmoid_kernel(A, B, gamma=1.0 / X_train.shape[1], coef0=0)
}
```

| Kernel Type       | Canonical Correlation | Interpretation                                    |
| ----------------- | --------------------- | ------------------------------------------------- |
| Linear            | 0.0318                | Negligible correlation                            |
| RBF               | -0.0756               | Poorly configured or structurally unfit           |
| Sigmoid           | 0.2161                | Weak nonlinear mapping                            |
| Polynomial (d=3)  | 0.7220                | Significant nonlinear relationships               |
| Polynomial (d=5)  | 0.7485                | Enhanced polynomial order effectiveness           |
| Polynomial (d=7)  | 0.7593                | Incremental improvements, diminishing returns     |
| Polynomial (d=9)  | 0.7645                | Approaching saturation                            |
| Polynomial (d=11) | 0.7667                | Peak performance                                  |
| Polynomial (d=13) | 0.7670                | Plateau achieved                                  |
| Polynomial (d=15) | 0.7662                | Slight decrease, indicating potential overfitting |

Therefore, polynomial kernels—by virtue of their capacity to model explicit high-order interactions through adjustable degree parameters—demonstrate the highest canonical correlations (≈0.77), peaking between degrees 11 and 13, which indicates they are particularly well suited to capture the complex nonlinear relationships inherent in feature projections. Other kernels (linear, RBF, sigmoid) fail to identify meaningful structures. Collectively, these experimental results clearly demonstrate that the relationship between CNN features and CLIP embeddings is predominantly nonlinear, characterized specifically by higher-order polynomial interactions. Consequently, these findings directly inform and motivate our proposed Converter design, emphasizing a dual-path structure combined with adaptive gating to efficiently capture complex nonlinear mappings.

---

```tex
% 1. 符號與資料矩陣
r_i = f_R(x_i) \in \mathbb{R}^{d_R},\quad
c_i = f_C(x_i) \in \mathbb{R}^{d_C}

R = \begin{bmatrix}
r_1^\top \\
\vdots \\
r_N^\top
\end{bmatrix}
\in \mathbb{R}^{N\times d_R},\quad
C = \begin{bmatrix}
c_1^\top \\
\vdots \\
c_N^\top
\end{bmatrix}
\in \mathbb{R}^{N\times d_C}

% 2. 線性投射模型
\hat C = R\,W + \mathbf{1}\,b^\top,\quad
W\in\mathbb{R}^{d_R\times d_C},\;b\in\mathbb{R}^{d_C}

E_{\rm linear}
= \min_{W,b}\;\bigl\|R\,W + \mathbf{1}\,b^\top - C\bigr\|_F^2

% 3. 非線性 MLP 投射
\hat C = g(R)
= \sigma\bigl(R\,W^{(1)} + \mathbf{1}\,b^{(1)\top}\bigr)\,W^{(2)} + \mathbf{1}\,b^{(2)\top}

E_{\rm MLP}
= \min_{g\in\mathcal{G}_{\rm MLP}}\;\bigl\|g(R) - C\bigr\|_F^2

% 4. 線性 vs. 核化 CCA
\rho_{\rm linear}
= \max_{u,v}\;\mathrm{corr}(R\,u,\;C\,v),\quad
\rho_{\rm kernel}
= \max_{u,v}\;\mathrm{corr}(\phi(R)\,u,\;\phi(C)\,v)
```