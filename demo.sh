# Imbalance ratio can be calculated by 'Ratio of the amount of data in major and minor classes before data subsampling' X 'minority_subsample_rate'.
# You can tune the parameter 'minority_subsample_rate' after checking the printed imbalance ratio when executing the file.

# Dataset: Cifar-10 (1), imbalance ratio: 0.04 (The base configuration)
# python train.py

# Dataset: Cifar-10 (1), imbalance ratio: 0.02
# python train.py --minority_subsample_rate 0.08

# Dataset: Cifar-10 (1), imbalance ratio: 0.01
# python train.py --minority_subsample_rate 0.04

# Dataset: Cifar-10 (2), imbalance ratio: 0.04
# python train.py --new_class '{0:[7], 1:[8, 9]}' --minority_subsample_rate 0.08

# Dataset: Cifar-10 (2), imbalance ratio: 0.005
# python train.py --new_class '{0:[7], 1:[8, 9]}' --minority_subsample_rate 0.01

# Multi Class (example1), imbalance ratio: (class0: 0.04, class1: 0.04)
# python train.py --new_class '{0:[0], 1:[1], 2:[3,4,5,6]}' --minor_classes [0,1]

# Multi Class (example2), imbalance ratio: (class0: 0.02, class1: 0.02, class3: 0.04)
# python train.py --new_class '{0:[0], 1:[1], 2:[3,4,5,6], 3:[7,8]}' --minor_classes [0,1,3] --minority_subsample_rate 0.08