import tensorflow as tf
import numpy as np
import os
from sklearn import metrics
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Agent:
    def __init__(self, network, dataset, memory, config):
        self.net = network
        self.dataset = dataset
        self.memory = memory
        self.config = config
        self.epsilon = 1.

        self.saver = tf.train.Saver()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)

    def get_action(self, state, is_train=True):
        if np.random.random() < self.epsilon and is_train:
            # random action
            action = [np.random.randint(self.net.n_class)]
        else:
            q = self.sess.run(self.net.q_mnet, feed_dict={self.net.state: state})
            action = np.argmax(q, axis=1)
        return action

    def get_reward_and_terminal(self, label, action):
        terminal = 0
        if action == label:
            reward = self.dataset.reward_set[label]
        else:
            reward = - self.dataset.reward_set[label]
            # End of an episode if the agent misjudgement about a minority class
            if label in self.config.minor_classes:
                terminal = 1
        return [reward], [terminal]

    def update_epsilon(self, train_step):
        epsilon_range = self.config.epsilon_range
        epsilon_polynomial_decay_step = self.config.epsilon_polynomial_decay_step
        self.epsilon = np.clip(
            (epsilon_range[0] - epsilon_range[1]) / epsilon_polynomial_decay_step * train_step + epsilon_range[1],
            *epsilon_range)

    def save_model(self, step):
        save_folder = self.config.save_folder
        if save_folder:
            save_path = os.path.join(save_folder, 'model' + '-step_%d' % step)
            self.saver.save(self.sess, save_path)

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.net.update_target, feed_dict={self.net.target_soft_update: 1.})
        print("start training")
        if self.config.restore_model_path:
            self.saver.restore(self.sess, self.config.restore_model_path)
        eval_s_list = np.array_split(self.dataset.x_test, 10)
        eval_y_list = list(np.squeeze(self.dataset.y_test))

        at = np.random.randint(len(self.dataset.x_train))
        s = self.dataset.x_train[at]  # (32, 32, 3)
        y = self.dataset.y_train[at]  # (1)
        train_step = 0
        pred_list, label_list = [], []  # for evaluation
        while train_step < self.config.train_step:
            a = self.get_action(s[np.newaxis, ...])  # (1)
            r, t = self.get_reward_and_terminal(*y, *a)  # (1), (1)
            at_ = np.random.randint(len(self.dataset.x_train))
            s_ = self.dataset.x_train[at_]  # (32, 32, 3)
            y_ = self.dataset.y_train[at_]  # (1)
            self.memory.stack([s, a, r, s_, t])

            if self.memory.check_train():
                sample_s, sample_a, sample_r, sample_s_, sample_t = self.memory.sample_batch(self.config.batch)
                q_mnet, q_tnet = self.sess.run([self.net.q_mnet, self.net.q_tnet],
                                               feed_dict={self.net.state: sample_s_})
                a_wrt_qmnet = np.argmax(q_mnet, axis=1)[:, np.newaxis]  # (batch, 1)
                max_q_ = np.take_along_axis(q_tnet, a_wrt_qmnet, axis=1)  # (batch, 1)
                self.sess.run(self.net.train_op, feed_dict={self.net.state: sample_s, self.net.action: sample_a,
                                                            self.net.reward: sample_r, self.net.terminal: sample_t,
                                                            self.net.target_q: max_q_,
                                                            self.net.learning_rate: self.config.learning_rate})
                pred_list.append(*a)
                label_list.append(*y)
                train_step += 1
                self.update_epsilon(train_step)

                if train_step % self.config.target_update_step == 0:
                    self.sess.run(self.net.update_target,
                                  feed_dict={self.net.target_soft_update: self.config.target_soft_update})

                if train_step % self.config.evaluation_term == 0:
                    # validation dataset
                    eval_a_list = []
                    for eval_s in eval_s_list:
                        eval_a = self.get_action(eval_s, False)  # we take greedy policy for validation dataset.
                        eval_a_list.extend(list(eval_a))
                    self.evaluate(label_list, pred_list, eval_y_list, eval_a_list, train_step, self.config.show_phase)
                    label_list.clear(), pred_list.clear()
                # save
                if train_step % self.config.save_term == 0:
                    self.save_model(train_step)
            at, s, y = at_, s_, y_

    def evaluate(self, train_label, train_prediction, val_label, val_prediction, step, show_phase="Both"):
        # Calculate f1 score of each class and weighted macro average
        print("train_step : {}, epsilon : {:.3f}".format(step, self.epsilon))
        if show_phase == "Both":
            phase = ["Train Data.", "Validation Data."]
            labels = [train_label, val_label]
            predictions = [train_prediction, val_prediction]
        elif show_phase == "Train":
            phase = ["Train Data."]
            labels = [train_label]
            predictions = [train_prediction]
        elif show_phase == "Validation":
            phase = ["Validation Data."]
            labels = [val_label]
            predictions = [val_prediction]
        for idx, (label, prediction) in enumerate(zip(labels, predictions)):
            f1_all_cls = metrics.f1_score(label, prediction, average=None)
            f1_macro_avg = metrics.f1_score(label, prediction, average='weighted')
            print("\t\t {:<20} f1-score of ".format(phase[idx]), end="")
            for i, f1 in enumerate(f1_all_cls):
                print("class {} : {:.3f}".format(i, f1), end=", ")
            print("weighted macro avg : {:.3f}".format(f1_macro_avg))