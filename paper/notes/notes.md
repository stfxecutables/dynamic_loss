# Contents

- [Updated Methods](#updated-methods)
- [Corrections to Loss Description [IMPORTANT!]](#corrections-to-loss-description-important)
  - [Issue #1 - Dynamic Loss Equation](#issue-1---dynamic-loss-equation)
    - [Issue #1 - Missing Negative](#issue-1---missing-negative)
    - [Issue #1 - Missing Simplification #1](#issue-1---missing-simplification)
    - [Issue #1 - Missing Simplification #2](#issue-1---missing-simplification)
  - [Issue #2 - Incorrect Description of Weight Updates](#issue-2---incorrect-description-of-weight-updates)
    - [Issue #2 - A Note on Classic Cross-Entopy Loss](#issue-2---a-note-on-classic-cross-entopy-loss)
    - [Issue #2 - The Dynamic Loss](#issue-2---the-dynamic-loss)
    - [Issue #2 - Simulations of Dynamic Loss Gradient Behaviour](#issue-2---simulations-of-dynamic-loss-gradient-behaviour)
      - [Issue #2 - Simulation Results](#issue-2---simulation-results)
    - [Nans in Training with Dynamic Loss](#nans-in-training-with-dynamic-loss)
- [Models](#models)
  - [Base Learners](#base-learners)
  - [Super Learners](#super-learners)
  - [Static Aggregation / Fusion Methods](#static-aggregation--fusion-methods)
- [Splitting Procedure](#splitting-procedure)
- [Training and Evaluation of Base Learners](#training-and-evaluation-of-base-learners)
- [Results](#results)
  - [Tables](#tables)
    - [Table 1: Ensemble Accuracies](#table-1-ensemble-accuracies)
    - [Table 2: Accuracy-Threshold Correlations](#table-2-accuracy-threshold-correlations)
    - [Table 3: Base Ensemble Performance](#table-3-base-ensemble-performance)
- [References](#references)







## Updated Methods

New source code is available on GitHub [@dm-bergerStfxecutablesDynamicLoss2022].

## Corrections to Loss Description [IMPORTANT!]

There are some errors in the current manuscript dynamic loss description.

### Issue #1 - Dynamic Loss Equation

The previous version of the manuscript defined the dynamic loss $\mathcal{L}_{\text{dyn}}$
to be, for a classification problem with $c$ classes, and free threshold parameter $\tau$:

$$
\mathcal{L}_{\text{dyn}}(\hat{y}, y) = \sum_{i=1}^c \mathcal{L}_{\text{dyn}}^{(i)}(\hat{y}_i, y_i)
$$

where

$$
\mathcal{L}_{\text{dyn}}^{(i)}(\hat{y}_i, y_i) =
\begin{cases}
y_i \cdot \log (0.1 \cdot \hat{y}_i) & \hat{y}_i < \tau \\
y_i \cdot \log (1.0) & \hat{y}_i \ge \tau \\
\end{cases}
$$

and where $\hat{y}_i$s are the **softmaxed predictions** (this is important)
such that $\hat{y}_i \in [0, 1]$ and $\sum \hat{y}_i = 1$.

#### Issue #1 - Missing Negative

The above equation is ***missing an important negative***:

$$
\mathcal{L}_{\text{dyn}}(\hat{y}, y) = -\sum_{i=1}^c \mathcal{L}_{\text{dyn}}^{(i)}(\hat{y}_i, y_i)
$$

#### Issue #1 - Missing Simplification #1

The equation is misleading / confusing as it is
written, since $\log(1.0)$ is zero, and so we really ought to write that the
loss simplfies to:

$$
\mathcal{L}_{\text{dyn}}^{(i)}(\hat{y}_i, y_i) =
\begin{cases}
y_i \cdot \log (0.1 \cdot \hat{y}_i) & \hat{y}_i < \tau \\
0 & \hat{y}_i \ge \tau \\
\end{cases}
$$

This ends up being important, because, as we shall see, if we define:

$$
\mathcal{L}_{\text{dyn}}^{(i)}(\hat{y}_i, y_i; \theta) =
\begin{cases}
y_i \cdot \log (0.1 \cdot \hat{y}_i) & \hat{y}_i < \tau \\
\theta & \hat{y}_i \ge \tau \\
\end{cases}
$$

then

$$
\nabla \mathcal{L}_{\text{dyn}}^{(i)}(\hat{y}_i, y_i; \theta) =
\nabla \mathcal{L}_{\text{dyn}}^{(i)}(\hat{y}_i, y_i; \theta^{\prime}) \quad \text{for all} \quad  \theta, \theta^{\prime} > 0
$$

because the derivative of a constant function is zero, regardless of the value
of the constant. I.e. ***the choice of fill value $\theta$ has no impact on the
gradients, and since the gradients are all that actually matter for the loss
function, the choice of fill value is also irrelevant***.


#### Issue #1 - Missing Simplification #2

Consider any function $f: \mathbb{R}^n \mapsto \mathbb{R}$, and any constant
$\gamma \in \mathbb{R}$. By the properties of the logarithm, then for $x \in
\mathbb{R}^n$, and for element-wise logarithm $\log$:

$$
\begin{align}
\nabla_x \left( \log (\gamma \cdot f) \right) &= \nabla_x (\log(\gamma) + \log(f)) \\
&= \nabla_x (\log(\gamma)) + \nabla_x (\log(f)) \\
&= \nabla_x (\log(f)) \\
\end{align}
$$

See also [logarithmic
differentiation](https://en.wikipedia.org/wiki/Logarithmic_differentiation).
Basically, if you are going to take a logarithm, then multiplication by a
constant does not affect the derivative.

Ignoring floating point issues, this means it
doesn't matter if we make our scaling factor 0.1, 69, or 12093.92: the
gradients are the same under the threshold, i.e. when $\hat{y}_i = x_i < \tau$:

\begin{align}


\mathcal{L}_{\text{dyn}}^{(i)}(x_i, y_i) &= y_i \cdot \log (0.1 \cdot x_i) & \qquad x_i < \tau \\

\nabla_x \left( \mathcal{L}_{\text{dyn}}^{(i)}(x_i, y_i) \right)

&= \nabla_x \big( y_i \cdot \log (0.1 \cdot x_i) \big) \\

&= y_i \nabla_x \log (0.1 \cdot x_i) \\

&= y_i \nabla_x \log (x_i) \\

&= \nabla_x \big( y_i \cdot \log (x_i) \big) \\

&= \nabla_x \left( \mathcal{L}_{\text{CE}}^{(i)}(x_i, y_i) \right) \\

\end{align}

Since the derivative is *all that matters* in a
loss function (actual values are irrelevant), then we actually have:

$$
\nabla\mathcal{L}_{\text{dyn}}^{(i)}(\hat{y}_i, y_i) =
\begin{cases}
\nabla \mathcal{L}^{(i)}_{\text{CE}}(\hat{y}_i, y_i) & \hat{y}_i < \tau \\
0 & \hat{y}_i \ge \tau \\
\end{cases}
$$

i.e. **the dynamic loss results in different model behaviour only when
softmaxed predictions are above the threshold: otherwise, the model trains
identically as to when using the cross-entropy loss**^[I saw this when making
the scaling factor ($\gamma$) a trainable parameter, and in the [loss demo
script](https://github.com/stfxecutables/dynamic_loss/blob/master/scripts/loss_demonstration.py).
When made trainable, the scaling factor never changed (gradients to it were
zero), and even as a constant, the choice of value has no impact on gradients.]. That
is, the 0.1 and 1.0 values end up being nothing but obscurantisms / distractions,
and the dynamic loss could be identically (and more clearly and elegantly) defined as:

$$
\mathcal{L}_{\text{dyn}}^{(i)}(\hat{y}_i, y_i) =
\begin{cases}
\mathcal{L}^{(i)}_{\text{CE}}(\hat{y}_i, y_i) & \hat{y}_i < \tau \\
0 & \hat{y}_i \ge \tau \\
\end{cases}
$$

where $\mathcal{L}^{(i)}_{\text{CE}}$ is the usual component of the cross-entropy loss.



### Issue #2 - Incorrect Description of Weight Updates

The following description accompanies the original equation (emphasis mine, on
incorrect parts):

> Any score above this threshold is updated according to equation 1 ($\hat{y}_i > \tau$),
> while any value below the threshold is downscaled from its
> original value according to equation 1 ($\hat{y}_i \le \tau$). Thus, the ***approach
> shares some similarities to a leaky rectified linear unit (ReLU) layer*** [...]

The main purpose of the LeakyReLU is to fix the problem of vanishing gradients
left of zero by replacing the constant mapping to zero with a linear mapping to
0.1 times the input. **Since the dynamic loss maps to a constant value, it has
more in common with the plain ReLU, in that the dynamic loss thresholding
_destroys_ gradients**. That is the following description:

> This custom loss function supports the creation of soft-max scores that are
> more indicative of classifier prediction reliability, thus potentially
> assisting in making incorrect and uncertain predictions more distinguishable
> from correct predictions. As an illustrative example, assuming a threshold of
> 0.7, ***if the soft-max value were above 0.7, and the prediction was correct,
> the weights would be adjusted very slightly, and if the prediction were
> incorrect, the weights would be adjusted more severely. In the reverse case,
> if the soft-max value were below the example threshold of 0.7, and the
> prediction was correct, the weights would be changed more severely. Finally,
> if the soft-max value were below 0.7 and the prediction was incorrect, the
> weights would not be changed substantially, since we would have considered
> the prediction as unreliable anyway***.

is mostly incorrect. In actual fact, weight updates from the dynamic loss
are identical to those from the cross-entropy loss in most cases, and when
updates are not identical, this is because the dynamic loss causes no updates
whatsoever: the dynamic loss is rather a "hard" loss which destroys gradients
above the threshold $\tau$.

#### Issue #2 - A Note on Classic Cross-Entopy Loss

The definition of the cross entropy loss (see [PyTorch
documentation](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html))
for a sample is such that:

$$
\mathcal{L}(\hat{\mathbfit{y}}, \mathbfit{y}) = -\sum_i^C y_i \log \sigma(\hat{y}_i)
$$

i.e. $\mathbfit{y}$ is assumed to be one-hot above. Letting $\mathbfit{x} = \mathbfit{\hat{y}}$:

$$
\begin{align}
\nabla_{\mathbfit{x}} \mathcal{L}(\mathbfit{x}, \mathbfit{y}) &= \nabla_{\mathbfit{x}} \left(-\sum_i^C y_i \log \sigma_i(\mathbfit{x})\right) \\
&= - \sum_i^C \nabla_{\mathbfit{x}} \left(y_i \log \sigma_i(\mathbfit{x})\right) \\
&= - \sum_i^C \big( (\nabla_{\mathbfit{x}} y_i)  \log \sigma_i(\mathbfit{x}) + y_i \nabla_{\mathbfit{x}} \log \sigma_i(\mathbfit{x}) \big)\\
&= - \sum_i^C \big(  y_i \nabla_{\mathbfit{x}} \log \sigma_i(\mathbfit{x}) \big)\\
\end{align}
$$

Note that if the correct prediction for a sample is $c \in \mathbb{Z}$, then in
the above equation, $y_i = 0$ if $i \ne c$, and 1 otherwise, so

$$
\nabla_{\mathbfit{x}} \mathcal{L}(\mathbfit{x}, \mathbfit{y})
= - \sum_{i = c} \big(  y_i \nabla_{\mathbfit{x}} \log \sigma_i(\mathbfit{x}) \big)
$$

and thus **weight updates for the classic cross-entropy loss come only from
gradients of the softmax components corresponding to the _correct_ class
predictions, $\sigma_i(\mathbfit{x})$**, where $i = c$ for corect class $c$.

#### Issue #2 - The Dynamic Loss


The smallest possible softmax (probability) value that can be observed given
$C$ classes if of course $1/C$, since otherwise the probabilities would not sum
to greater than zero. The largest possible softmax value is 1, which is not
helpful, but is there a bound on the *second* largest possible softmax value,
supposing the largest value is beyond some threshold $\tau$?

There is.

Specifically, if $\tau$ > 0.5, and $C \ge 2$ (both of which are true in our
manuscript), then there are in fact only two mutually-exclusive cases
for the softmax value distributions, due to the requirement that softmax values
must sum to 1:

1. **"Unconfident"**: all softmax values for a sample are $\le \tau$
2. **"Confident"**: one softmax value for a sample is $> \tau$, and all others are $< \tau$

We have shown above that dynamic loss gradients are identical to cross-entropy
gradients in case (1), since constant multiplication has no effect on gradients
under the logarithm, and since the dynamic loss in this case is just a constant
multiplication of the cross-entropy loss.  Thus we need only consider what
happens in case (2).

**When the prediction is incorrect and _over_-confident**, i.e. there is a softmax value $> \tau$ on the
*wrong class*, then gradients are identical to those from the cross-entropy
loss, since softmax values on the incorrect class do not contribute to
gradients in the cross-entropy loss, and since the dynamic loss also sets these
gradients to zero.

**This leaves only the case where the prediction is both confident ($> \tau$) and
correct**. In this case, *the gradient of the cross-entropy loss is non-zero, but
the gradient of the dynamic loss is zero, by definition*. This is
the only case where the loss functions differ. We can see this in simulations
below.

#### Issue #2 - Simulations of Dynamic Loss Gradient Behaviour

This simulation ([repo script](https://github.com/stfxecutables/dynamic_loss/blob/master/scripts/loss_demonstration.py)) shows that the
dynamic loss differs from the cross-entropy loss only in that it *prevents* the
model from updating from correct predictions above the threshold $\tau$.

In this demo, we define an additional "soft dynamic loss", which is defined (in
various equivalent forms, as per above) as:


\begin{aligned}
\mathcal{L}_{\text{soft}}^{(i)}(\hat{y}_i, y_i) &=
\begin{cases}
y_i \cdot \log (0.1 \cdot \hat{y}_i) & \hat{y}_i < \tau \\
y_i \cdot \log (\hat{y}_i^{0.1}) & \hat{y}_i \ge \tau \\
\end{cases} \\

&=
\begin{cases}
y_i \cdot \log (0.1 \cdot \hat{y}_i) & \hat{y}_i < \tau \\
0.1 \cdot y_i \cdot \log (\hat{y}_i) & \hat{y}_i \ge \tau \\
\end{cases} \\

&=
\begin{cases}
\mathcal{L}_{\text{CE}}^{(i)}(\hat{y}_i, y_i) & \hat{y}_i < \tau \\
0.1 \cdot \mathcal{L}_{\text{CE}}^{(i)}(\hat{y}_i, y_i) & \hat{y}_i \ge \tau \\

\end{cases} \\
\end{aligned}

The last line shows how this truly emulates a LeakyReLU, as gradients are
identical to those from the cross-entropy loss under the threshold $\tau$, and
multiplied by 0.1 when over the threshold (compared to the "hard" dynamic loss,
which would zero gradients on these values).





##### Issue #2 - Simulation Results


When a prediction is **correct** and **confident** (above threshold), weights
are NOT updated for that sample with the dynamic loss. In this case, the soft
dynamic loss behaves more like the previous manuscript description, updating
weights less. **This is the only case where the dynamic loss results in different
behaviour from the cross-entropy loss**.


```
================================================================================
CORRECT, CONFIDENT (above threshold) prediction (seed=16)
================================================================================

Softmaxed inputs to loss function from linear layer
[[0.069 0.921 0.01 ]]

Correct target: 1
Prediction:     1
Threshold:     0.7

Gradients on softmaxed variable:
  Cross-entropy loss [[ 0.      -1.08573  0.     ]]
  Dynamic loss       [[0. 0. 0.]]
  Soft dynamic loss  [[ 0.      -0.10857  0.     ]]

Gradients on linear layer weights:
  Cross-entropy loss
    [[ 0.079 -0.05   0.262  0.396]
    [-0.091  0.058 -0.301 -0.456]
    [ 0.012 -0.008  0.04   0.06 ]]
  Dynamic loss
    [[ 0. -0.  0.  0.]
    [ 0. -0.  0.  0.]
    [ 0. -0.  0.  0.]]
  Soft dynamic loss
    [[ 0.0079 -0.005   0.0262  0.0396]
    [-0.0091  0.0058 -0.0301 -0.0456]
    [ 0.0012 -0.0008  0.004   0.006 ]]
```

When a sample prediction is **correct** but **unconfident** (below threshold), weights are
updated identically to as in the cross-entropy loss.

```
================================================================================
CORRECT, DOUBTFUL (below threshold) prediction (seed=3)
================================================================================

Softmaxed inputs to loss function from linear layer
[[0.222 0.568 0.21 ]]

Correct target: 1
Prediction:     1
Threshold:     0.7

Gradients on softmaxed variable:
  Cross-entropy loss [[ 0.     -1.7619  0.    ]]
  Dynamic loss       [[ 0.     -1.7619  0.    ]]
  Soft dynamic loss  [[ 0.     -1.7619  0.    ]]

Gradients on linear layer weights:
Cross-entropy loss
    [[ 0.103 -0.165  0.302  0.008]
    [-0.2    0.322 -0.588 -0.016]
    [ 0.097 -0.156  0.286  0.008]]
  Dynamic loss
    [[ 0.103 -0.165  0.302  0.008]
    [-0.2    0.322 -0.588 -0.016]
    [ 0.097 -0.156  0.286  0.008]]
  Soft dynamic loss
    [[ 0.103  -0.1653  0.3024  0.0081]
    [-0.2003  0.3217 -0.5883 -0.0157]
    [ 0.0974 -0.1564  0.286   0.0076]]
```

When a sample prediction is ***in*correct** but **over-confident (above threshold)**
about a prediction, weights are updated identically to as they would be with
the cross-entropy loss.

```
================================================================================
INCORRECT, OVERCONFIDENT (above threshold) prediction (seed=6)
================================================================================

Softmaxed inputs to loss function from linear layer
[[0.02  0.073 0.907]]

Correct target: 1
Prediction:     2
Threshold:     0.7

Gradients on softmaxed variable:
  Cross-entropy loss [[  0.      -13.70468   0.     ]]
  Dynamic loss       [[  0.      -13.70468   0.     ]]
  Soft dynamic loss  [[  0.      -13.70468   0.     ]]

Gradients on linear layer weights:
  Cross-entropy loss
    [[ 3.300e-02 -4.000e-03 -1.160e-01  2.000e-03]
    [-1.498e+00  1.800e-01  5.339e+00 -8.100e-02]
    [ 1.465e+00 -1.760e-01 -5.223e+00  7.900e-02]]
  Dynamic loss
    [[ 3.300e-02 -4.000e-03 -1.160e-01  2.000e-03]
    [-1.498e+00  1.800e-01  5.339e+00 -8.100e-02]
    [ 1.465e+00 -1.760e-01 -5.223e+00  7.900e-02]]
  Soft dynamic loss
    [[ 3.2600e-02 -3.9000e-03 -1.1600e-01  1.8000e-03]
    [-1.4979e+00  1.8030e-01  5.3392e+00 -8.0900e-02]
    [ 1.4653e+00 -1.7640e-01 -5.2232e+00  7.9100e-02]]
```

When a sample prediction is ***in*correct** but **under-confident (below threshold)**
about a prediction, weights are updated identically to as they would be with
the cross-entropy loss.

```
================================================================================
INCORRECT, DOUBTFUL (below threshold) prediction (seed=0)
================================================================================

Softmaxed inputs to loss function from linear layer
[[0.106 0.587 0.307]]

Correct target: 0
Prediction:     1
Threshold:     0.7

Gradients on softmaxed variable:
  Cross-entropy loss [[-9.42081  0.       0.     ]]
  Dynamic loss       [[-9.42081  0.       0.     ]]
  Soft dynamic loss  [[-9.42081  0.       0.     ]]

Gradients on linear layer weights:
  Cross-entropy loss
    [[-1.391  3.52  -0.515 -1.455]
    [ 0.914 -2.312  0.338  0.956]
    [ 0.477 -1.208  0.177  0.499]]
  Dynamic loss
    [[-1.391  3.52  -0.515 -1.455]
    [ 0.914 -2.312  0.338  0.956]
    [ 0.477 -1.208  0.177  0.499]]
  Soft dynamic loss
    [[-1.3908  3.5196 -0.5148 -1.4555]
    [ 0.9135 -2.3119  0.3382  0.956 ]
    [ 0.4772 -1.2077  0.1767  0.4994]]
```

Thus, the intuitive explanation of the dynamic loss is simply that it prevents
learning on confidently, correctly predicted samples (and thus, perhaps,
prevents overfitting to "easy" samples). This suggests the dynamic loss has
more in common with **label smoothing** [@mullerWhenDoesLabel2020], where
highly confident predictions (large softmax values) are randomly penalized.

#### NaNs in Training with Dynamic Loss

With a high dynamic loss threshold in $\{0.8, 0.9\}$, it was occasionally
possible for the loss to devolve entirely to NaNs. This happened only for
CIFAR-100 1/50 times when the threshold was 0.8, and 2/50 times when the
threshold was 0.9. In addition, all models using the dynamic loss required
a learning rate $1/10^{\text{th}}$ of the learning rate found via tuning with
the usual cross-entropy loss.

Nevertheless, all models trained with the dynamic loss seemed to benefit from
a learning rate $1/10^{\text{th}}$ of the magnitude of the learning rate found
to be best when using the classical cross-entropy loss, in that larger learning
rates *extremly* freqeuently resulted in Nan losses.

This is likely a numerical stability issue, in that the dynamic loss prevents
the use of the more numerically-stable
[`CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html),
and requires the manual computation of softmax and logarithm values. The
"zeroing" of gradients from the dynamic loss means that naive logarithms will
be computed for values very lose to or equal to zero, resulting in negative
infinity and/or NaN values in some unlucky cases. When implementing my own
trainable version of the dynamic loss, I had to add a stability epsilon to
prevent such issues ([source](https://github.com/stfxecutables/dynamic_loss/blob/master/src/dynamic_loss.py#L139)).

## Models

### Base Learners

We train WideResNet-16-8 (WR-16-8) [@zagoruykoWideResidualNetworks2017] base
learners on CIFAR-10, CIFAR-100, and FashionMNIST datasets.

### Super Learners

Suppose we have an ensemble of $E$ trained base learners, and the raw
(un-softmaxed) predictions on $N$ samples for each of these learners, and where
the dataset has $C$ classes. If there are $N_{\text{train}}$ training samples
(a subset of which may have been used for training each base learner), and
$N_{\text{test}}$ testing samples (which were never seen by base-learners),
then this yields a super training set $\hat{\mathbf{Y}}_{\text{train}} =
[\hat{\mathbf{y}}^{(1)}_{\text{train}}, \dots,
\hat{\mathbf{y}}^{(E)}_{\text{train}} ]$, where
$\hat{\mathbf{y}}^{(i)}_{\text{train}} \in \mathbb{R}^{ N_{\text{train}} \times
C }$ is the matrix of un-softmaxed predictions for the $N_{\text{train}}$
training samples of ensemble $i$, and thus $\hat{\mathbf{Y}}_{\text{train}} \in
\mathbb{R}^{N_{\text{train}} \times C \times E }$. Likewise, there is a
super-evaluation set $\hat{\mathbf{Y}}_{\text{test}}$ with $N_{\text{test}}$
samples.

A "super-training sample" $\hat{\mathbf{Y}}_i$ is thus a matrix of predictions
in $\mathbb{R}^{C \times E}$. This can be flattened into a vector $\mathbf{v}_i$ of length $D
= C \times E$ and fed into a linear deep learning model $f$ such that $f(\mathbf{v}_i)
\in \mathbb{R}^C$ is the final class prediction.

We train two super or "meta" learner models, which operate on the raw
(un-softmaxed) predictions of all base learners.

The "Weighted" model (Tables 1-2) is a simple linear model which is just
weighted combination of the components of $\mathbf{v}_i$, i.e. $f(\mathbf{v}) =
\mathbf{A}\cdot\mathbf{v} + \mathbf{b}$. This is equivalent to a
multilayer-perceptron (MLP) with a single linear layer and no activations (see
the paper [source code](https://github.com/stfxecutables/dynamic_loss/blob/master/src/models.py#L336-L349)).

The "MLP" model (Tables 1-2) is a modern MLP architecture with 6 hidden layers,
and which incorporates both batch normalization and dropout layers
[see @martinezSimpleEffectiveBaseline2017, or the [source code for this
paper](https://github.com/stfxecutables/dynamic_loss/blob/master/src/models.py#L313-L333)].

Both super-learner models were implemented and trained using PyTorch
[@paszkePyTorchImperativeStyle2019], and trained with a learning rate of $3
\times 10^{-4}$ and using the AdamW optimizer
[@loshchilovDecoupledWeightDecay2019] without weight decay (brief investigation
showed different weight decays had negligible impacts on final results). The
simple linear model was trained for 10 epochs, and the MLP for 20 epochs, and
final performances were evaluated on $\hat{\mathbf{Y}}_{\text{test}}$ using the
weights from training epoch with the best validation accuracy on a 10% subset
of $\hat{\mathbf{Y}}_{\text{train}}$ (early stopping).

### Static Aggregation / Fusion Methods

We also compared the impact of the dynamic loss on two classic model fusion
methods: voting and averaging ("Vote" and "Average" in Tables 1 and 2,
respectively). For voting, the ensemble prediction was taken to be the modally-predicted
class. For averaging, the raw (un-softmaxed) predictions were averaged across ensembles
to create a single raw average prediction, and these predictions where then softmaxed
and argmaxed to obtain the final average aggregate predictions.
Code for these implementations is available
[here](https://github.com/stfxecutables/dynamic_loss/blob/master/src/ensemble.py#L64-L98).



## Splitting Procedure

Each dataset has official training data $\mathbfit{x}$ with labels
$\mathbfit{y}$ and test data $(\mathbfit{x}_{\text{test}},
\mathbfit{y}_{\text{test}})$. We construct 50 different base training datasets
$\mathbfit{x}_{\text{train}}^{(i)}$ via *k*-fold with *k*=50 and bootstrap
resampling: first, $\mathbfit{x}_{\text{fold}}^{(i)}$ is the *i*th training
*k*-fold partition from $\mathbfit{x}$, and then
$\mathbfit{x}_{\text{fold}}^{(i)}$ is resampled with replacement to create
$\mathbfit{x}_{\text{base}}^{(i)}$,
which has the same number of samples as $\mathbfit{x}_{\text{fold}}^{(i)}$.
This base-learner training subset is then finally split into $\mathbfit{x}_{\text{train}}^{(i)}$
and $\mathbfit{x}_{\text{val}}^{(i)}$, where $\mathbfit{x}_{\text{val}}^{(i)}$ is 20% of
the samples of $\mathbfit{x}_{\text{base}}^{(i)}$ heldout for internal validation during
training of the base learner, and where $\mathbfit{x}_{\text{train}}^{(i)}$ is the remaining
80%. In total, since 50-fold uses 98% of samples in a training fold, and a bootstrap resample is
expected to contain approximately 63% of the original samples, the the final
$\mathbfit{x}_{\text{train}}^{(i)}$ set can be expected to contain 0.80 * 0.98 * 0.63 = 49.4%
distinct samples from the original training data $\mathbfit{x}$. The corresponding labels
$\mathbfit{y}_{\text{train}}^{(i)}$ are selected identically.

Note also that since
the internal validation sample $\mathbfit{x}_{\text{val}}^{(i)}$ may contain some samples also
present in $\mathbfit{x}_{\text{train}}^{(i)}$, due to resampling, we do not condition
training, stopping, or any decisions on any internal validation metrics, and the validation
set ultimately functions solely to futher increase variation in the final base-learner
training sets.

## Training and Evaluation of Base Learners

Each base WR-16-8 learner $i$ is trained on each dataset's training data
$(\mathbfit{x}_{\text{val}}^{(i)}, \mathbfit{y}_{\text{train}}^{(i)})$ for 50
epochs, using the AdamW optimizer [@loshchilovDecoupledWeightDecay2019] with a
weight decay of 0.05, initial learning rate of 0.1, and using a batch size of 1024.
The learning rate was warmed up linearly starting from $10^{-5}$ for 5 epochs, until
reaching the initial learning rate, and then decayed to a minimum learning rate
of $10^{-9}$ via cosine annealing [@loshchilovSGDRStochasticGradient2017].


The initial learning rate and weight decay described above were found via grid
search on CIFAR-100 only, over the learning rates $\{0.001, 0.01, 0.05, 0.1\}$
and weight decays $\{10^{-4}, 5 \times 10^{-4}, 0.001, 0.005, 0.01, 0.05,
0.1\}$, and otherwise using identical training parameters as described above.
This search was done using the classical cross-entropy loss: base
models were trained with the dynamic loss were trained with an initial learning rate
of 0.01, $1/10^{\text{th}}$ of the learning rate of those with the classic
cross-entropy loss, since larger rates resulted in NaNs.

Upon completing training, predictions $\hat{\mathbfit{y}}$ were made on the
full original training data $\mathbfit{x}$, as were predictions
$\hat{\mathbfit{y}}_{\text{test}}$ as on the full test set
$\mathbfit{x}_{\text{test}}$. These raw (un-softmaxed) predictions
$\hat{\mathbfit{y}}$ were saved and set aside for later training of super/meta
learners, and the validation samples $\hat{\mathbfit{y}}_{\text{test}}$ were
set aside for testing of those same super/meta-learners.

Each base model was trained and evaluated on a single V100 GPU, with each base
model taking approximately 45 minutes to train. Each model was trained using
either no dynamic loss threshold, or a dynamic loss threshold in $\{0.6, 0.7,
0.8, 0.9\}$. Thus, each  base training subset
$\mathbfit{x}_{\text{train}}^{(i)}$ yielded a total of 5 different models. The
total training and evaluation budget was thus 5 * 45min / base model * 3 datasets * 50 base
models / dataset $\approx$ 24 GPU days.




## Results

There was an exceptionally clear relationship between the dynamic loss threshold and final ensemble performance.


### Tables

#### Table 1: Ensemble Accuracies

|          Data | CIFAR10     |          |              |         | CIFAR100    |          |              |         | FashionMNIST |          |              |         |
|--------------:|-------------|----------|--------------|---------|-------------|----------|--------------|---------|--------------|----------|--------------|---------|
|    **Fusion** | **Average** | **Vote** | **Weighted** | **MLP** | **Average** | **Vote** | **Weighted** | **MLP** | **Average**  | **Vote** | **Weighted** | **MLP** |
| **Threshold** |             |          |              |         |             |          |              |         |              |          |              |         |
|      **None** | 0.8840      | 0.8814   | 0.8741       | 0.9052  | 0.6563      | 0.6588   | 0.6759       | 0.6958  | 0.9374       | 0.9371   | 0.9278       | 0.9406  |
|       **0.6** | 0.9029      | 0.9034   | 0.8747       | 0.9165  | 0.7101      | 0.7094   | 0.7062       | 0.7298  | 0.9431       | 0.9431   | 0.9236       | 0.9463  |
|       **0.7** | 0.9091      | 0.9096   | 0.8951       | 0.9173  | 0.7118      | 0.7100   | 0.7131       | 0.7295  | 0.9452       | 0.9455   | 0.9307       | 0.9465  |
|       **0.8** | 0.9098      | 0.9114   | 0.8956       | 0.9217  | 0.7148      | 0.7145   | 0.7084       | 0.7335  | 0.9457       | 0.9459   | 0.9328       | 0.9473  |
|       **0.9** | 0.9148      | 0.9158   | 0.9085       | 0.9228  | 0.7201      | 0.7210   | 0.7107       | 0.7364  | 0.9461       | 0.9467   | 0.9398       | 0.9479  |

**Table 1**: Accuracies of WideResNet-16-8 ensembles, by dynamic loss threshold and fusion strategies.

<hr>

#### Table 2: Accuracy-Threshold Correlations

|              | CIFAR10 | CIFAR100 | FashionMNIST |
|-------------:|---------|----------|--------------|
|  **Average** | 0.975   | 0.999    | 0.979        |
|      **MLP** | 0.965   | 0.998    | 0.996        |
|     **Vote** | 0.981   | 0.998    | 0.975        |
| **Weighted** | 0.685   | 0.985    | 0.421        |

**Table 2**:  Pearson correlations between dynamic loss threshold and ensemble performance, by dataset and fusion method.

<hr>

#### Table 3: Base Ensemble Performance

|              |               | mean     | std      | min    | max    |
|-------------:|---------------|----------|----------|--------|--------|
|     **Data** | **Threshold** |          |          |        |        |
|      CIFAR10 | -1.0          | 0.847316 | 0.014557 | 0.8102 | 0.8747 |
|              | 0.6           | 0.871064 | 0.008301 | 0.8463 | 0.8885 |
|              | 0.7           | 0.878064 | 0.006873 | 0.8640 | 0.8899 |
|              | 0.8           | 0.881754 | 0.006667 | 0.8582 | 0.8968 |
|              | 0.9           | 0.884318 | 0.008179 | 0.8638 | 0.8980 |
|     CIFAR100 | -1.0          | 0.574636 | 0.021818 | 0.4973 | 0.6108 |
|              | 0.6           | 0.633200 | 0.009395 | 0.6100 | 0.6489 |
|              | 0.7           | 0.636844 | 0.008167 | 0.6137 | 0.6533 |
|              | 0.8           | 0.640447 | 0.008345 | 0.6173 | 0.6584 |
|              | 0.9           | 0.645869 | 0.006691 | 0.6276 | 0.6588 |
| FashionMNIST | -1.0          | 0.930534 | 0.001941 | 0.9261 | 0.9350 |
|              | 0.6           | 0.935220 | 0.001925 | 0.9294 | 0.9387 |
|              | 0.7           | 0.937574 | 0.001824 | 0.9336 | 0.9418 |
|              | 0.8           | 0.938620 | 0.001485 | 0.9356 | 0.9414 |
|              | 0.9           | 0.939708 | 0.001945 | 0.9343 | 0.9428 |

**Table 3**: Base learner accuracy statistics by dataset and dynamic loss threshold.

<hr>

# References