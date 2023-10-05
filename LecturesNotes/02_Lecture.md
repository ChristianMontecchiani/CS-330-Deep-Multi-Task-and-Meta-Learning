# Multi Task Learning 

### Overwiew 
- [Problem Statement](#problem-statement)
- [Models, Objective, Optimization](#model)
- [Challenges](#challenge)
- [Case study of real world multi task learning](#case-study)

#### Notation
- $x$ is the input of the NN.
- $y$ is the output of the NN.
- $\theta$ are the parameters of the NN.
- $f$ represent the NN and $f_{\theta}(y|x)$ is the a distribution over $y$ given the input $x$ parameterized by $\theta$


### Problem Statement
**Multi-task learning** is a technique used to train neural networks to perform multiple tasks at the same time by sharing some of the model's parameters across tasks. This is different from **single-task supervised learning**, where we work with a dataset $\mathcal{D}\{(x, y)_{k}\}$, and define a loss function $\min_{\theta}  \mathcal{L}$ to find a neural network that performs well on the given task.

A formal task definition consists of three components: $\mathcal{T} = \{p_{i}(x), p_{i}(y|x), \mathcal{L}_{i}\}$
1. A distribution over input $p_{i}(x)$.
2. A distribution of outputs given inputs $p_{i}(y|x)$.
3. A loss function $\mathcal{L}_{i}$.

For multi-task learning, various scenarios include:
- **Multi task classification**: Where the loss remains the same across all tasks, for example, handwriting recognition across different languages. 

- **Multi label learning**: In cases where the loss $\mathcal{L}$ and $p_{i}(x)$ are consistent across all tasks, as seen in face attribute recognition with different label distributions.

- **Loss varying**: When there are different types of labels, either continuous or discrete.

To enable a neural network to distinguish tasks, a task descriptor $z_{i}$ is provided as input to the network. Consequently, the neural network becomes $f_{\theta}(y|x, z_{i})$. Task descriptors can take various forms, including one-hot encoding or natural language processing strings (e.g., summary, explanation). 
The objective for this setup is to minimize $\min_{\theta} \sum_{i=1}^{T} \mathcal{L}_{i}(\theta, \mathcal{D}_{i})$, where $T$ is the number of tasks.

### Model

##### How should the model be condition on $z_{i}$?
One way to condition the model on $z_{i}$ s by assuming $z_{i}$ to be the one-hot task index. In this approach, 
$T$ separate neural networks with distinct parameters exist, and the final output combines the 
$z_{i}$ vector with the outputs of the $T$ neural networks. Each neural network is trained independently. This obviously an extreme case.

Another extreme case is to share all the parameters for all the task and just by concatenating $z_{i}$ to one of the layer. And in this case all parameters are shared across the task, except the paramteres of $z_{i}$.


### Architectures
An alternative is to split the parameters on $\theta^{sh}$ shared parameters and task specific parameters $\theta^{i}$. Then the objective is: $\min_{\theta^{sh}, \theta^{1},\dots, \theta^{T}} \sum_{i=1}^{T} \mathcal{L}_{i}(\{\theta^{sh}, \theta^{i}\}, \mathcal{D}_{i})$.

### Objective
Vanilla Multi Task Learning (MLT) is: $\min_{\theta} \sum_{i=1}^{T} \mathcal{L}_{i}(\theta, \mathcal{D}_{i})$. Often a weighted sum over the task differently, so it becomes: 
$\min_{\theta} \sum_{i=1}^{T} w_{i}\mathcal{L}_{i}(\theta, \mathcal{D}_{i})$

The problem become how to choose $w_{i}$? 
- Manually based on importance or priority.
- Dynamically changing weights.
- Heuristics-based approaches.

### Optimizing the objective
To optimize the objective, follow these steps: 
1. Sample mini-batch of tasks $\mathcal{B} \sim \mathcal{T}_{i}$
2. Sample mini-batch datapoints for each task $\mathcal{D}_{i}^{b} \sim \mathcal{D}_{i}$
3. Compute loss on the mini-batch $\mathcal{L}(\theta, \mathcal{B} = \sum_{\mathcal{T}_{k} \sim \mathcal{B}} \mathcal{L}_{k} (\theta, \mathcal{D}_{k}^{b}))$
4. Backpropagate loss to compute gradient $\nabla_{\theta} \mathcal{L}$
5. Apply gradient with your favourite NN optimizer (ex. Adam).


## Challenge 
1. **Negative transfer**: Independent networks may work better due to cross-task interference or differing convergence rates among tasks. Larger neural networks might be needed to accommodate multiple tasks, or consider implementing multi-head neural networks.

2. **Overfitting**: Sharing parameters can be a solution if overfitting occurs on certain tasks.

3. **What if you have a lot of tasks?**: When dealing with numerous tasks, there is no closed-form solution for measuring task similarity. Analyzing tasks based on gradients and optimization processes during training can help decide which tasks to train together.


## Case study
Making recommendations for YouTube.