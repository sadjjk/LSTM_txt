import os
import time

import tensorflow as tf
import numpy as np


def pick_top_n(preds, top_n=5):
    # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    p = np.squeeze(preds)

    # 返回数组值从小到大的索引
    p[np.argsort(p)[:-top_n]] = 0

    # 归一化 转化为概率值
    p = p / np.sum(p)

    # 随机选取一字符
    c = np.random.choice(len(p), 1, p=p)[0]

    return c


class CharRNN:
    def __init__(self, vocab_size, num_seqs=64, num_steps=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False, train_keep_prob=0.5,
                 use_embedding=False, embedding_size=128):

        if sampling is True:
            num_seqs, num_steps = 1, 1
        else:
            num_seqs, num_steps = num_seqs, num_steps

        self.vocab_size = vocab_size
        self.num_seqs = num_seqs
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size

        tf.reset_default_graph()

        self.build_inputs()
        self.build_lstm()
        self.build_loss()
        self.build_optimizer()

        self.saver = tf.train.Saver()

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=(self.num_seqs, self.num_steps), name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=(self.num_seqs, self.num_steps), name='targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            # 是否使用embedding层,若矩阵过于疏松,如中文,建议使用
            # 如果是仅有英文字母, one_hot即可
            if self.use_embedding:
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])
                    self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)
            else:
                self.lstm_inputs = tf.one_hot(self.inputs, self.vocab_size)

    def build_lstm(self):
        # 单层cell
        def get_a_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop

        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
                                               )
            self.initial_state = cell.zero_state(self.num_seqs, tf.float32)

            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_inputs, initial_state=self.initial_state)

            x = tf.reshape(self.lstm_outputs, [-1, self.lstm_size])

            with tf.variable_scope('softmax'):

                softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.vocab_size], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(self.vocab_size))

            self.logits = tf.matmul(x, softmax_w) + softmax_b
            self.proba_prediction = tf.nn.softmax(self.logits, name='predictions')

    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets, self.vocab_size)
            y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)

    def build_optimizer(self):
        # 梯度裁剪 防止梯度爆炸
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))

    def train(self, batch_generator, save_path, log_every_n):

        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())

            step = 0
            new_state = sess.run(self.initial_state)
            for x, y in batch_generator:
                step += 1
                start = time.time()
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: self.train_keep_prob,
                        self.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([self.loss,
                                                     self.final_state,
                                                     self.optimizer],
                                                    feed_dict=feed)
                end = time.time()

                if step % log_every_n == 0:
                    print('step:{}...'.format(step),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))

            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)

    def sample(self, n_samples, prime_string, vocab_size):
        samples = [c for c in prime_string]

        sess = self.session
        new_state = sess.run(self.initial_state)

        # print(samples)
        # 最后一个字符预测 如果为空c为，返回长度
        try:
            c = samples[-1]
        except IndexError:
            c = vocab_size
        x = np.zeros((1, 1))
        # 输入单个字符
        x[0, 0] = c
        feed = {self.inputs: x,
                self.keep_prob: 1.,
                self.initial_state: new_state}
        preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                    feed_dict=feed)
        c = pick_top_n(preds, vocab_size)
        # 添加字符到samples中
        samples.append(c)

        # 不断生成字符，直到达到指定数目
        for i in range(n_samples):
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)

            c = pick_top_n(preds, vocab_size)
            samples.append(c)

        return np.array(samples)

    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))
