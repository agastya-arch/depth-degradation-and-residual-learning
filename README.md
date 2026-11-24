# Depth Degradation and Residual Connections in MLPs

A small PyTorch experiment inspired by the degradation problem discussed in **Deep Residual Learning for Image Recognition (ResNet)**. The objective is to investigate whether increasing network depth makes optimization harder, and whether identity skip connections can recover the lost performance.

## Results at a glance

| Metric | Shallow MLP | Deep plain MLP | Deep residual MLP |
|---|---:|---:|---:|
| Hidden layers | 3 | 12 | 12 |
| Trainable parameters | 643 | 3,091 | 3,091 |
| Skip connections | 0 | 0 | 5 |
| Initial training loss | 1.114741 | 1.099639 | 1.150591 |
| Last recorded training loss | 0.381229 | 1.098613 | **0.376123** |
| Training accuracy | 83.60% | 33.00% | **84.03%** |
| Test accuracy | **85.22%** | 32.89% | 84.78% |
| Misclassified test points | 133/900 | 604/900 | 137/900 |

Loss values are taken from saved notebook outputs at epochs 0 and 2990; accuracy is evaluated after training. The results represent one recorded run per architecture.

![Training loss comparison](assets/loss_comparison.png)

The plain 12-hidden-layer MLP remains near the uniform-prediction cross-entropy baseline, ln(3) ≈ 1.098612. Adding residual connections to the deep model restores learning without increasing its parameter count. The shallow model remains slightly more accurate on the held-out test set.

## 1. Experimental question

Increasing depth expands the set of representations a network can potentially learn, but does not guarantee that gradient-based optimization will find good parameters. In the degradation problem, a deeper plain network can have higher **training error** than a shallower network. This differs from ordinary overfitting, where training performance is good but test performance deteriorates.

This experiment tests a small-scale analogue of that problem using fully connected networks rather than the convolutional networks of the original ResNet paper.

## 2. Dataset

Three synthetic concentric rings in two dimensions:

- Nominal radii: 1.0, 2.5 and 4.0.
- Angles sampled uniformly from 0 to 2π.
- Radial Gaussian noise: standard deviation 0.65.
- Training set: 3,000 points (1,000 per class).
- Test set: 900 separately generated points (300 per class).
- Input: two Cartesian coordinates; target: one of three class labels.
- Dataset generator seed: 42. The notebooks use a separate PyTorch random generator for the data and shuffle the samples.

Each point is constructed as `x = r * cos(theta)` and `y = r * sin(theta)`, where `r` is the nominal radius plus noise. The class distributions overlap, making perfect classification unrealistic. Extra depth cannot remove ambiguity inherent in the dataset.

## 3. Model architectures

### A. Shallow MLP

`2 → 16 → 16 → 16 → 3`

Three hidden linear layers with ReLU after each, followed by a linear output layer. This is the baseline.

### B. Deep plain MLP

`2 → [16 × 12 hidden layers] → 3`

Twelve hidden linear layers with ReLU after each. The forward pass is ordinary `model(X)`.

### C. Deep residual MLP

`2 → 16 → [residual block × 5] → 16 → 3`

The architecture retains all twelve hidden linear layers: one input projection, ten layers inside five residual blocks, and one remaining hidden layer. Each block has two 16-to-16 linear transformations. The output layer produces three logits. The underlying `nn.Sequential` modules are the same as in the deep plain MLP.

## 4. How residual learning works

A plain block computes a transformation of its input. A residual block instead learns a correction to its input:

$$F(h)=W_2\,\mathrm{ReLU}(W_1h+b_1)+b_2$$

$$H(h)=\mathrm{ReLU}(h+F(h))$$

The shortcut bypasses the two linear layers and is added **before** the final ReLU. Input and transformed output have the same shape `(batch_size, 16)`, so elementwise addition is possible. When the residual branch is zero and the incoming activation is nonnegative, the block can act as an identity mapping.

For the pre-activation sum `z = h + F(h)`, its Jacobian is:

$$\frac{\partial z}{\partial h}=I+\frac{\partial F}{\partial h}$$

The identity term provides a direct route for gradients through the addition. The final ReLU also affects the actual gradient, so residual connections do not guarantee that vanishing or exploding gradients disappear.

### Implementation without custom classes

The residual notebook uses an ordinary Python function and indexes the modules in `nn.Sequential`. The first block is implemented as follows:

```python
a = model[0](x)
input_layer = model[1](a)
a = model[2](input_layer)
a = model[3](a)
a = model[4](a)
a = a + input_layer
a = model[5](a)
```

The same pattern is repeated for the other four blocks, saving the input of **each** block separately. All learnable parameters remain registered in `model`, so `model.parameters()` works with SGD and PyTorch autograd differentiates through the additions automatically.

**Important:** use `forward(X)` instead of `model(X)` throughout residual training, testing and decision-boundary prediction. Calling the sequential model directly bypasses the shortcuts. The custom function must process its argument, not a global training tensor.

## 5. Training configuration

| Setting | Value |
|---|---|
| Framework | PyTorch |
| Optimizer | SGD (no momentum) |
| Learning rate | 0.01 |
| Epochs | 3,000 full-batch updates |
| Loss | CrossEntropyLoss |
| Activation | ReLU |
| Hidden width | 16 |
| Model initialization seed specified in notebooks | 42 |
| Loss logging interval | 10 epochs |

The model outputs raw logits; `CrossEntropyLoss` handles log-softmax internally. Each iteration performs a forward pass, calculates loss, clears gradients, backpropagates and updates parameters. Predictions use `argmax` over the three logits.

## 6. Convergence and decision boundaries

First logged epoch at which the training loss crosses each threshold:

| Threshold | Shallow | Deep plain | Deep residual |
|---|---:|---:|---:|
| Below 0.9 | 1080 | Not reached | 240 |
| Below 0.5 | 1490 | Not reached | 560 |
| Below 0.4 | 1890 | Not reached | 900 |

The residual model reaches these thresholds faster in epoch count. This does not establish faster wall-clock training.

| Shallow | Deep plain | Deep residual |
|---|---|---|
| ![Shallow decision boundary](assets/shallow_boundary.png) | ![Plain decision boundary](assets/deep_plain_boundary.png) | ![Residual decision boundary](assets/deep_residual_boundary.png) |

The shaded regions indicate predicted classes, colored points indicate true classes, and black crosses indicate misclassified test samples.

## 7. Interpretation

- Under the recorded SGD configuration, the deep plain model fails to learn useful classification boundaries, despite having more parameters than the shallow model.
- The residual model recovers approximately the shallow model's accuracy and reaches low training loss in fewer epochs.
- For this simple, noisy dataset, the extra depth is not needed for better test accuracy. The shallow model uses substantially fewer parameters and is the more compact solution.
- The experiment illustrates improved trainability, not a guarantee that residual networks always outperform plain or shallow models.

## 8. Limitations and future work

This is an educational experiment, not an exact replication of ResNet or a novel research result. The recorded results come from single runs. The notebooks specify fixed seeds and matching data-generation code, but the saved outputs have not been independently rerun to verify the exact initial weights and data used. Training loss is logged every ten epochs rather than after the final update.

Vanishing gradients are a possible explanation for the deep plain network's failure, but their role has not been measured. Future investigations could compare layer-wise gradient norms, repeat across multiple initialization seeds, measure wall-clock convergence, and test alternative initialization schemes or optimizers. These extensions are intentionally left for later.

## 9. Running the notebooks

Install the dependencies and launch Jupyter:

```bash
pip install -r requirements.txt
jupyter notebook
```

Restart the kernel and run each notebook from top to bottom in order to reset the seeds and avoid stale variables. The notebooks are standalone. Package versions are not pinned; exact numerical reproducibility across environments is not guaranteed.

| Notebook | Purpose |
|---|---|
| [01_Shallow_MLP.ipynb](01_Shallow_MLP.ipynb) | Shallow baseline |
| [02_Deep_Plain_MLP.ipynb](02_Deep_Plain_MLP.ipynb) | Deep plain network and degradation |
| [03_Deep_Residual_MLP.ipynb](03_Deep_Residual_MLP.ipynb) | Residual network with manual forward pass |

The `assets/` directory contains the original notebook decision-boundary images, a combined loss plot and a CSV of the saved loss logs (300 entries per model). The original notebooks are included unchanged.

## Reference

He, K., Zhang, X., Ren, S., and Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385).
