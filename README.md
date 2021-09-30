# Deep-reinforcement-learning-for-multi-class-imbalanced-classification
Implementation of [Deep reinforcement learning for imbalanced classification](https://arxiv.org/abs/1901.01379) and its extended version to multi-class imbalanced classification.

### Differences with the original paper

* [Double DQN](https://arxiv.org/abs/1509.06461) and [Dueling DQN](https://arxiv.org/abs/1511.06581) are applied.
* The reward function on the paper is extended to multi-class imbalanced data. 
* It has been implemented to easily test various multi-class imbalanced settings of Cifar-10 dataset.

### Test environment
* Python 3.7
* Tensorflow 1.14

### Examples of how to run
You can check example codes for some major configurations in ``demo.sh``.
```
$ ./demo.sh
```

### Experiment results
The values of train parameters from the [original paper](https://arxiv.org/abs/1901.01379) are used.

| Dataset | Imbalance ratio | F-measure 
|  ---    |  ---            | ---   
| Cifar-10(1) | 4% | 0.901 
| | 2% | 0.879 
| | 1% | 0.862 
| | 0.5% | 0.784  
| | |     
| Cifar-10(2) | 4% | 0.887  
| | 2% | 0.855 
| | 1% | 0.806 
| | 0.5% | 0.708 



