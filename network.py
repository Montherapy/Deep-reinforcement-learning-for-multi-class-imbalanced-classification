import tensorflow as tf


class QNetwork:
    def __init__(self, config):
        size = 32
        channel = 3
        self.config = config
        self.n_class = len(self.config.new_class)
        tf.reset_default_graph()
        with tf.variable_scope('input'):
            self.state = tf.placeholder(shape=[None, size, size, channel], dtype=tf.float32)
            self.learning_rate = tf.placeholder(dtype=tf.float32)
            self.target_q = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            self.reward = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            self.action = tf.placeholder(shape=[None, 1], dtype=tf.int32)
            self.terminal = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            self.target_soft_update = tf.placeholder(dtype=tf.float32)
        with tf.variable_scope('main_net'):
            self.q_mnet = self._build_network()
        with tf.variable_scope('target_net'):
            self.q_tnet = self._build_network()

        main_variables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="main_net")
        target_variables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_net")

        self.update_target = [t.assign((1 - self.target_soft_update) * t + self.target_soft_update * m)
                              for t, m in zip(target_variables, main_variables)]
        self.q_wrt_a = tf.expand_dims(tf.gather_nd(self.q_mnet, self.action, batch_dims=1), axis=1)
        self.target = self.reward + (1 - self.terminal) * self.config.gamma * self.target_q
        self.loss = tf.losses.huber_loss(self.target, self.q_wrt_a)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=main_variables)

    def _build_network(self):
        x = tf.layers.conv2d(self.state, 32, 5, strides=2, activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 32, 5, strides=2, activation=tf.nn.relu)
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, 256, activation=tf.nn.relu)
        a = tf.layers.dense(x, self.n_class)
        v = tf.layers.dense(x, 1)
        q = v + a - tf.reduce_mean(a, axis=1, keepdims=True)
        return q