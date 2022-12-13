# Contents

- [Updated Methods](#updated-methods)
- [Corrections to Loss Description [IMPORTANT!]](#corrections-to-loss-description-important)
  - [Issue #1 - Misleading and Unclear Formula](#issue-1---misleading-and-unclear-formula)
  - [Issue #2 - Incorrect Description of Gradients](#issue-2---incorrect-description-of-gradients)
  - [Models](#models)
    - [Base Learners](#base-learners)
    - [Super Learners](#super-learners)
    - [Static Aggregation / Fusion Methods](#static-aggregation--fusion-methods)
  - [Splitting Procedure](#splitting-procedure)
  - [Training and Evaluation of Base Learners](#training-and-evaluation-of-base-learners)
- [Results](#results)
  - [Tables](#tables)
- [References](#references)


## Updated Methods

New source code is available on GitHub [@dm-bergerStfxecutablesDynamicLoss2022].

## Corrections to Loss Description [IMPORTANT!]

There are a number of errors in the current manuscript dynamic loss description.

### Issue #1 - Misleading and Unclear Formula

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

and where $\hat{y}_i$s are the softmaxed predictions such that $\hat{y}_i \in
[0, 1]$ and $\sum \hat{y}_i = 1$. This equation is ***missing an important
negative***, and is misleading / confusing as it is written, since $\log(1.0)$
is zero, and so we really ought to write that the loss simplfies to:

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
of the constant. I.e. ***the choice of fill value has no impact on the
gradients, and since the gradients are all that actually matter for the loss
function, the choice of fill value is also irrelevant***.

In fact, there is technically another issue, because since $\log(ab) = \log(a) + \log(b)$, then:


$$
\begin{align}
-y_i \cdot \log (0.1 \cdot \hat{y}_i) &= y_i \cdot \left( \log (0.1) \cdot \log(\hat{y}_i) \right) \\
&= \log (0.1) \left( y_i \cdot  \log(\hat{y}_i) \right) \\
&\approx 2.3 \left( y_i \cdot  \log(\hat{y}_i) \right)
\end{align}
$$

So what we *actually* ultimately have, if we denote the classic cross-entropy loss components as $\mathcal{L}^{(i)}_{\text{CE}}$, is

$$
\mathcal{L}_{\text{dyn}}^{(i)}(\hat{y}_i, y_i) =
\begin{cases}
2.3 \cdot \mathcal{L}^{(i)}_{\text{CE}}(\hat{y}_i, y_i) & \hat{y}_i < \tau \\
0 & \hat{y}_i \ge \tau \\
\end{cases}
$$

Since $\nabla(c \cdot f) = c \cdot \nabla f$, this makes it more clear that the
**dynamic loss is the cross-entropy loss, but with gradients doubled under the
threshold $\tau$, and zeroed otherwise**. I.e., the dynamic loss *increases*
learning from low-confidence predictions, and *prevents* learning on confident
predictions.

### Issue #2 - Incorrect Description of Gradients

The following description accompanies the original equation (emphasis mine, on
incorrect parts):

> Any score above this threshold is updated according to equation 1 ($\hat{y}_i > \tau$),
> while any value below the threshold is downscaled from its
> original value according to equation 1 ($\hat{y}_i \le \tau$). Thus, the ***approach
> shares some similarities to a leaky rectified linear unit (ReLU) layer*** [...]

In fact, the purpose of the LeakyReLU is to retain (diminished) gradients left
of zero by replacing the constant mapping to zero with a linear mapping to 0.1
times the input. **Since the dynamic loss maps to a constant value, it has
more in common with the plain ReLU, in that the effect of purpose of the
dynamic loss thresholding is to _destroy_ gradients**. That is the following:

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

is mostly incorrect.

You can see this is not the case in from the [demo script in the repo](https://github.com/stfxecutables/dynamic_loss/blob/master/scripts/loss_demonstration.py).
In this demo, we define an additional "soft dynamic loss", which is defined as:


\begin{aligned}
\mathcal{L}_{\text{soft}}^{(i)}(\hat{y}_i, y_i) &=
\begin{cases}
y_i \cdot \log (0.1 \cdot \hat{y}_i) & \hat{y}_i < \tau \\
y_i \cdot \log (\hat{y}_i^{0.1}) & \hat{y}_i \ge \tau \\
\end{cases} \\

\\
&=
\begin{cases}
y_i \cdot \log (0.1 \cdot \hat{y}_i) & \hat{y}_i < \tau \\
0.1 \cdot y_i \cdot \log (\hat{y}_i) & \hat{y}_i \ge \tau \\
\end{cases}
\end{aligned}

This truly emulates a LeakyReLU, as gradients are doubled under the threshold,
and multiplied by 0.1 when over the threshold. Here as some samples from the
script:

**Both predictions incorrect**

```
Softmaxed inputs to loss function (threshold=0.3)
[[0.587 0.413]
 [0.025 0.975]]

Correct targets:
[[0]
 [0]]

Preds after argmax:
[[0]
 [1]]

Gradients on raw linear outputs:
Cross-entropy loss
[[-0.20656  0.20656]
 [-0.48766  0.48766]]
Dynamic loss
[[ 0.       0.     ]
 [-0.48766  0.48766]]
Soft dynamic loss
[[-0.02066  0.02066]
 [-0.48766  0.48766]]

Gradients on softmaxed variable:
Cross-entropy loss
[[ -0.8519   0.    ]
 [-20.2611   0.    ]]
Dynamic loss
[[  0.       0.    ]
 [-20.2611   0.    ]]
Soft dynamic loss
[[ -0.0852   0.    ]
 [-20.2611   0.    ]]
```

### Models

#### Base Learners

We train WideResNet-16-8 (WR-16-8) [@zagoruykoWideResidualNetworks2017] base
learners on CIFAR-10, CIFAR-100, and FashionMNIST datasets.

#### Super Learners

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

#### Static Aggregation / Fusion Methods

We also compared the impact of the dynamic loss on two classic model fusion
methods: voting and averaging ("Vote" and "Average" in Tables 1 and 2,
respectively). For voting, the ensemble prediction was taken to be the modally-predicted
class. For averaging, the raw (un-softmaxed) predictions were averaged across ensembles
to create a single raw average prediction, and these predictions where then softmaxed
and argmaxed to obtain the final average aggregate predictions.
Code for these implementations is available
[here](https://github.com/stfxecutables/dynamic_loss/blob/master/src/ensemble.py#L64-L98).



### Splitting Procedure

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
set functions practicall solely to futher increase variation in the final base-learner
training sets.

### Training and Evaluation of Base Learners

Each base WR-16-8 learner $i$ is trained on each dataset's training data
$(\mathbfit{x}_{\text{val}}^{(i)}, \mathbfit{y}_{\text{train}}^{(i)})$ for 50
epochs, using the AdamW optimizer [@loshchilovDecoupledWeightDecay2019] with a
weight decay of 0.05, initial learning rate of 0.1, and using a batch size of 1024.
The learning rate was warmed up linearly starting from $10^{-5}$ for 5 epochs, until
reaching the initial learning rate, and then decayed to a minimum learning rate
of $10^{-9}$ via cosine annealing [@loshchilovSGDRStochasticGradient2017].

The initial learning rate and weight decay were found via grid search on
CIFAR-100 only, over the learning rates $\{0.001, 0.01, 0.05, 0.1\}$ and weight
decays $\{10^{-4}, 5 \times 10^{-4}, 0.001, 0.005, 0.01, 0.05, 0.1\}$, and
otherwise using identical training parameters as described above.

Upon completing training, predictions $\hat{\mathbfit{y}}$ were made on the
full original training data $\mathbfit{x}$, as well as on the full test set
$\mathbfit{x}_{\text{test}}$, $\hat{\mathbfit{y}}_{\text{test}}$. These raw
(un-softmaxed) predictions were saved and set aside for later training
$\hat{\mathbfit{y}}$, and validation $\hat{\mathbfit{y}}_{\text{test}}$ of
super/meta-learners.

Each base model was trained and evaluated on a single V100 GPU, with each base
model taking approximately 45 minutes to train. Each model was trained using
either no dynamic loss threshold, or a dynamic loss threshold in $\{0.6, 0.7,
0.8, 0.8\}$. Thus, each  base training subset
$\mathbfit{x}_{\text{train}}^{(i)}$ yielded a total of 5 different models. The
total training budget was thus 5 * 45min / base model * 3 datasets * 50 base
models / dataset $\approx$ 24 GPU days.




## Results


### Tables

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


|              | CIFAR10 | CIFAR100 | FashionMNIST |
|-------------:|---------|----------|--------------|
|  **Average** | 0.975   | 0.999    | 0.979        |
|      **MLP** | 0.965   | 0.998    | 0.996        |
|     **Vote** | 0.981   | 0.998    | 0.975        |
| **Weighted** | 0.685   | 0.985    | 0.421        |

**Table 2**:  Pearson correlations between dynamic loss threshold and ensemble performance, by dataset and fusion method.

# References