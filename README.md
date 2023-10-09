# Deep Multi-Task Learning and Meta Learning :robot:

## Stanford CS330-2023

This repo contains homework assignment solutions for the Stanford [CS 330](https://cs330.stanford.edu/) 
(Deep Multi-Task and Meta Learning) class offered in Autumn 2023.  A brief summary of key concepts covered in different 
assignments is summarized below.

## :memo: Homework 0: [Multitask Training for Recommender Systems](HW0/CS330_Homework0.pdf)
In this assignment, we will implement a multi-task movie recommender system based on the classic **Matrix Factorization** and **Neural Collaborative Filtering** algorithms. In particular, we will build a model based on the _BellKor solution_ to the Netflix Grand Prize challenge and extend it to predict both likely user-movie interactions and potential scores. A multi-task neural network architecture is  implemented and the effect of parameter sharing and loss weighting on model performance is explored.

The main goal of these exercises is to get familiar with multi-task architectures, the training pipeline, and coding in PyTorch.

## :memo: Homework 1: [Data Processing and Black-Box Meta-Learning](HW1/CS330_Homework_1.pdf)
Goal of this assignment is to understand meta-learning for **few shot classification**. Following are the key undertakings:
- Learn how to process and partition data for meta learning problems, where training is done over a distribution of training tasks 
- Implement and train _memory augmented neural networks_ (MANN), a black-box meta-learner that uses a recurrent neural network [1].
- Analyze the learning performance for different size problems.
- Experiment with model parameters and explore how they improve performance.
The analysis is performed using **Omniglot** [2], a dataset with 1623 characters from 50 languages and each character
has 20 images.


## References
[1] Santoro, A., Bartunov, S., Botvinick, M., Wierstra, D., & Lillicrap, T. (2016, June). Meta-learning with memory-augmented neural networks. In International conference on machine learning (pp. 1842-1850). PMLR.

[2] Lake, B. M., Salakhutdinov, R., & Tenenbaum, J. B. (2015). Human-level concept learning through probabilistic program induction. Science, 350(6266), 1332-1338.

