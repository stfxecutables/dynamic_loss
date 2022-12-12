## Updated Methods

We train a WideResNet-16-8 (WR-16-8) [@zagoruykoWideResidualNetworks2017] on CIFAR-10,
CIFAR-100, and FashionMNIST datasets.

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

### Training of Base Learners

Each base WR-16-8 learner $i$ is trained on the data
$(\mathbfit{x}_{\text{val}}^{(i)}, \mathbfit{y}_{\text{train}}^{(i)})$ for 50
epochs, using the AdamW optimizer [@loshchilovDecoupledWeightDecay2019] with a
weight decay of 0.05 , initial learning rate of 0.1, using a batch size of 1024.
The learning rate was warmed up linearly starting from $10^{-5}$ for 5 epochs, until
reaching the initial learning rate, and then decayed to a minimum learning rate
of $10^{-9}$ via cosine annealing [@loshchilovSGDRStochasticGradient2017].





## Tables

|          Data | CIFAR10     |          |              |         | CIFAR100    |          |              |         | FashionMNIST |          |              |         |
|--------------:|-------------|----------|--------------|---------|-------------|----------|--------------|---------|--------------|----------|--------------|---------|
|    **Fusion** | **Average** | **Vote** | **Weighted** | **MLP** | **Average** | **Vote** | **Weighted** | **MLP** | **Average**  | **Vote** | **Weighted** | **MLP** |
| **Threshold** |             |          |              |         |             |          |              |         |              |          |              |         |
|      **None** | 0.8840      | 0.8814   | 0.8729       | 0.9048  | 0.6563      | 0.6588   | 0.6721       | 0.6978  | 0.9374       | 0.9371   | 0.9313       | 0.9413  |
|       **0.6** | 0.9029      | 0.9034   | 0.8832       | 0.9166  | 0.7101      | 0.7094   | 0.7052       | 0.7298  | 0.9431       | 0.9431   | 0.9244       | 0.9452  |
|       **0.7** | 0.9091      | 0.9096   | 0.8923       | 0.9176  | 0.7118      | 0.7100   | 0.7128       | 0.7309  | 0.9452       | 0.9455   | 0.9315       | 0.9463  |
|       **0.8** | 0.9098      | 0.9114   | 0.8967       | 0.9228  | 0.7148      | 0.7145   | 0.7089       | 0.7319  | 0.9457       | 0.9459   | 0.9359       | 0.9469  |
|       **0.9** | 0.9148      | 0.9158   | 0.9006       | 0.9239  | 0.7201      | 0.7210   | 0.7138       | 0.7365  | 0.9461       | 0.9467   | 0.9404       | 0.9493  |

**Table 1**: Accuracies of WideResNet-16-8 ensembles, by dynamic loss threshold and fusion strategies.


| Data       | CIFAR10  | CIFAR100 | FashionMNIST |
|------------|----------|----------|--------------|
| **Fusion** |          |          |              |
| Average    | 0.974619 | 0.999399 | 0.979239     |
| MLP        | 0.956017 | 0.998031 | 0.919808     |
| Vote       | 0.980550 | 0.997606 | 0.975035     |
| Weighted   | 0.886108 | 0.990659 | 0.269259     |

**Table 2**:  Pearson correlations between dynamic loss threshold and ensemble performance, by dataset and fusion method.

# References