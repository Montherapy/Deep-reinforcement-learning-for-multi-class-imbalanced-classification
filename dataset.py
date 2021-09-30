import tensorflow as tf
import numpy as np


class Dataset:
    def __init__(self, config):
        self.config = config
        self.n_class = len(self.config.new_class)
        self.label_connector = {}
        self.get_label_changer()
        self.set_data(self.config.minority_subsample_rate)
        self.get_rho()

    def get_label_changer(self):
        new_class_setting = self.config.new_class  # old classes to new class
        for new_label in new_class_setting.keys():
            for old_label in new_class_setting[new_label]:
                self.label_connector[old_label] = new_label

        for new_label, old_label in new_class_setting.items():
            print("\nNew label {} = old label".format(new_label), *old_label)

    def set_data(self, minority_subsample_rate=1):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # 1. Delete non-used class
        survived_class = list(self.label_connector.keys())
        self.x_train = x_train[np.isin(y_train, survived_class).squeeze()]
        self.x_test = x_test[np.isin(y_test, survived_class).squeeze()]
        y_train = y_train[np.isin(y_train, survived_class).squeeze()].reshape(-1, 1)
        y_test = y_test[np.isin(y_test, survived_class).squeeze()].reshape(-1, 1)

        # 2. Change from old label to new label
        self.y_train, self.y_test = -np.ones_like(y_train), -np.ones_like(y_test)
        for i, old in enumerate(y_train.squeeze()):
            self.y_train[i, 0] = self.label_connector[old]
        for i, old in enumerate(y_test.squeeze()):
            self.y_test[i, 0] = self.label_connector[old]

        # 3. Minor class subsampling for decrease imbalance ratio
        if minority_subsample_rate < 1:
            # decrease the number of minority class
            nums_cls = self.get_class_num()
            delete_indices = set()
            for minor_cl in self.config.minor_classes:
                num_cl = nums_cls[minor_cl]
                idx_cl = np.where(self.y_train == minor_cl)[0]
                delete_idx = np.random.choice(idx_cl, int(num_cl * (1 - minority_subsample_rate)), replace=False)
                delete_indices.update(delete_idx)
            survived_indices = set(range(len(self.y_train))).difference(delete_indices)
            self.y_train = self.y_train[list(survived_indices)]
            self.x_train = self.x_train[list(survived_indices)]
        print("\nNumber of each class.")
        for cl_idx, cl_num in enumerate(self.get_class_num()):
            print("\t- Class {} : {}".format(cl_idx, cl_num))
        print("\nImbalance ratio compared to major class.")
        for cl_idx, cl_ratio in enumerate(self.get_class_num() / max(self.get_class_num())):
            print("\t- Class {} : {:.3f}".format(cl_idx, cl_ratio))

        # 4. Data normalization
        image_mean = np.array([self.x_train[..., i].mean() for i in range(3)])
        self.x_train = (self.x_train - image_mean) / 255
        self.x_test = (self.x_test - image_mean) / 255

    def get_class_num(self):
        # get number of all classes
        _, nums_cls = np.unique(self.y_train, return_counts=True)
        return nums_cls

    def get_rho(self):
        """
        In the two-class dataset problem, this paper has proven that the best performance is achieved when the reciprocal of the ratio of the number of data is used as the reward function.
        In this code, the result of this paper is extended to multi-class by creating a reward function with the reciprocal of the number of data for each class.
        """
        nums_cls = self.get_class_num()
        raw_reward_set = 1 / nums_cls
        self.reward_set = np.round(raw_reward_set / np.linalg.norm(raw_reward_set), 6)
        print("\nReward for each class.")
        for cl_idx, cl_reward in enumerate(self.reward_set):
            print("\t- Class {} : {:.6f}".format(cl_idx, cl_reward))