import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import time
import esposallesData

class SeqLearn():
    def __init__(self, n_classes, datasets):
        self.trainImg, self.seqLen_train, self.trainLabel, self.validationImg, self.seqLen_validation, self.validationLabel,  self.testImg, self.seqLen_test, self.testLabel = datasets
        self.n_examples = len(self.trainImg)
        self.n_examples_t = len(self.validationImg)
        self.n_features = esposallesData.IMG_HEIGHT
        if esposallesData.TEXTLINE:
            self.batch_size = 8
        else:
            self.batch_size = 64
        self.n_classes = n_classes
        self.n_hidden = 32
        self.n_layers = 1
        self.learning_rate = 1e-3
        self.n_epochs = 100
        self.n_batches_per_epoch = int(self.n_examples/self.batch_size)
        self.n_batches_per_epoch_t = int(self.n_examples_t/self.batch_size)
        self.summary = tf.Summary()
        self.summary_writer = tf.summary.FileWriter('ler_epoch_tensorboard')
        self.model()

    def sparse_tuple_from(self, sequences, dtype=np.int32):
        # Create a sparse representention of x.
        # Args:
        # sequences: a list of lists of type dtype where each element is a sequence
        # Returns:
        # A tuple with (indices, values, shape)
        indices = []
        values = []
        for n, seq in enumerate(sequences):
            indices.extend(zip([n]*len(seq), range(len(seq))))
            values.extend(seq)
        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)
        return indices, values, shape

    def model(self):
        self.x = tf.placeholder(tf.float32, [None, self.n_features, None]) # (batch_size, n_features, time_steps)
        self.y = tf.sparse_placeholder(tf.int32)
        self.seqLen = tf.placeholder(tf.int32, [None])

        # <CNN>
        batch_s = tf.shape(self.x)[0]
        conv = tf.reshape(self.x, shape=[batch_s, self.n_features, -1, 1])
        w_conv = tf.Variable(tf.random_normal([5, 5, 1, 32]))
        b_conv = tf.Variable(tf.constant(0., shape=[32]))
        conv = tf.nn.conv2d(conv, w_conv, strides=[1, 1, 1, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, b_conv)
        conv = tf.nn.relu(conv)
        # conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # shapeConv = tf.shape(conv) # (batch_size, features/2, time_step/2, channels==32)
        xx = tf.transpose(conv, (2, 0, 1, 3)) # (time/2, batch, features/2, channels==32)
        xx = tf.reshape(xx, [-1, batch_s, self.n_features*32]) # (time/2, batch, features/2 * 32)
        # </CNN>

        lstm_fw_cell = rnn.BasicLSTMCell(self.n_hidden)
        lstm_bw_cell = rnn.BasicLSTMCell(self.n_hidden)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, xx, self.seqLen, dtype=tf.float32, time_major=True)
        outputs = tf.concat(outputs, 2) # (time_step, batch, features*2)

        outputs = tf.reshape(outputs, [-1, self.n_hidden*2])
        weight2 = tf.Variable(tf.random_normal([self.n_hidden*2, self.n_classes+1]))
        bias2 = tf.Variable(tf.constant(0., shape=[self.n_classes+1]))
        pred = tf.matmul(outputs, weight2) + bias2
        self.pred = tf.reshape(pred, [-1, batch_s, self.n_classes+1])

        loss = tf.nn.ctc_loss(self.y, self.pred, self.seqLen)
        self.cost = tf.reduce_mean(loss)
        #self.loss_summary = tf.summary.scalar('loss', self.cost) ###
        #self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.cost)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        self.decoded, log_prob = tf.nn.ctc_greedy_decoder(self.pred, self.seqLen)
        self.decoded_long, log_prob = tf.nn.ctc_greedy_decoder(self.pred, self.seqLen, merge_repeated=False)
        self.mistake_num = tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.y, normalize=False)
        #self.error_rate_summary = tf.summary.scalar('error rate', self.label_error_rate) ###
        #self.merged_summary = tf.summary.merge([self.loss_summary, self.error_rate_summary]) ###

    def train(self, test_flag=True):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for epoch in range(self.n_epochs+1):
                train_cost = mistake_num = y_label_len = 0
                start = time.time()
                for batch in range(self.n_batches_per_epoch):
                    batch_x = np.array(self.trainImg[batch*self.batch_size: (batch+1)*self.batch_size])
                    batch_train_seqLen = self.seqLen_train[batch*self.batch_size: (batch+1)*self.batch_size]
                    batch_y = self.trainLabel[batch*self.batch_size: (batch+1)*self.batch_size]
                    label_len = sum([len(i) for i in batch_y])
                    batch_y = self.sparse_tuple_from(batch_y)
                    feed = {self.x: batch_x, self.y: batch_y, self.seqLen: batch_train_seqLen}
                    batch_cost, _, prediction = sess.run([self.cost, self.optimizer, self.pred], feed_dict=feed)
                    train_cost += batch_cost * self.batch_size
                    mistake_num += sess.run(self.mistake_num, feed_dict=feed).sum()
                    y_label_len += label_len
                train_cost /= self.n_examples
                train_cer = mistake_num / y_label_len
                self.summary.value.add(tag='train_cer', simple_value=train_cer)
                print('epoch {}/{}, train_cost={:.3f}, train_cer={:.3f}, time={:.3f}'.format(epoch, self.n_epochs, train_cost, train_cer, time.time()-start))
                with open('train_cer.log', 'a') as f:
                    f.write(str(train_cer))
                    f.write(' ')

                if test_flag:
                    mistake_num_t = y_label_len_t = 0
                    start_t = time.time()
                    for bat in range(self.n_batches_per_epoch_t):
                        batch_x_t = np.array(self.validationImg[bat*self.batch_size: (bat+1)*self.batch_size])
                        batch_validation_seqLen = self.seqLen_validation[bat*self.batch_size: (bat+1)*self.batch_size]
                        batch_y_t = self.validationLabel[bat*self.batch_size: (bat+1)*self.batch_size]
                        label_len_t = sum([len(i) for i in batch_y_t])
                        batch_y_t = self.sparse_tuple_from(batch_y_t)
                        feed_t = {self.x: batch_x_t, self.y: batch_y_t, self.seqLen: batch_validation_seqLen}
                        mistake_num_t += sess.run(self.mistake_num, feed_dict=feed_t).sum()
                        y_label_len_t += label_len_t
                    test_cer = mistake_num_t / y_label_len_t
                    self.summary.value.add(tag='test_cer', simple_value=test_cer)
                    print('###TEST### character error rate: {:.3f}, time={:.3f}'.format(test_cer, time.time()-start_t))
                    with open('test_cer.log', 'a') as f:
                        f.write(str(test_cer))
                        f.write(' ')

                self.summary_writer.add_summary(self.summary, epoch)


if __name__ == '__main__':
    #labelNum, (trainImg, seqLen_train, trainLabel), (validationImg, seqLen_validation, validationLabel), (testImg, seqLen_test, testLabel) = esposallesData.getData(1280, 10, 4)
    labelNum, (trainImg, seqLen_train, trainLabel), (validationImg, seqLen_validation, validationLabel), (testImg, seqLen_test, testLabel) = esposallesData.getData(None, None, None)
    model = SeqLearn(labelNum, [trainImg, seqLen_train, trainLabel, validationImg, seqLen_validation, validationLabel, testImg, seqLen_test, testLabel])
    model.train()
