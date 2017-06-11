from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, random
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs


class Config:

    # parameters that can be passed in
    height = 200
    width = 52

    n_classes = 5

    # local parameters

    conv_strides = (1, 1, 1, 1)
    pool_ksize = (1, 2, 2, 1)
    pool_strides = (1, 2, 2, 1)

    patch_size = 3 # 3x3 filter for convolution
    n_channels = 16
    n_pools = 2

    reduced = int(height * width / (4 ** n_pools))
    fc_input_size = reduced * n_channels
    fc_units = 128 # 64 # 1024

    n_epochs = 100
    batch_size = 10
    early_term_criterion = 1.0e-6
    n_epochs_per_ckpt = 10
    n_epochs_per_print = 10

    dtype = tf.float32
    decay_rate = 0.95 # decay per epoch
    max_grad_norm = 10.0

    # hyperparameters
    starter_lr = 0.001
    dropout = 0.1
    l2_reg = 0.001


class ConvolutionNeuralNetwork(object):

    def __init__(self, train_dir=None, fc_units=None):
        # set random seed
        random.seed(101)
        np.random.seed(101)
        tf.set_random_seed(101)
        tf.reset_default_graph()

        # update train_dir
        if train_dir is None:
            train_dir = "data/train.cnn"

        if fc_units is not None:
            Config.fc_units = fc_units
            train_dir = train_dir + "." + str(fc_units) + ".fc_units"

        print("fc units", Config.fc_units)
        print("train_dir", train_dir)

        # print config
        print("n_classes =", Config.n_classes)
        print("height =", Config.height)
        print("width =", Config.width)
        print("starter_lr =", Config.starter_lr)
        print("dropout =", Config.dropout)
        print("l2_reg =", Config.l2_reg)

        # create neural network
        self.model = Model()

        # save your model parameters/checkpoints here
        self.train_dir = train_dir
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        # start tf session
        self.session = tf.Session()

        # initialize model
        ckpt = tf.train.get_checkpoint_state(self.train_dir)
        v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
        if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
            print("read parameters from %s" % ckpt.model_checkpoint_path)
            self.model.saver.restore(self.session, ckpt.model_checkpoint_path)
        else:
            print("created model with fresh parameters")
            self.session.run(tf.global_variables_initializer())
        print("n_params: %d" % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))

    def close():
        if self.session is not None:
            self.session.close()
            self.session = None

    def fit(self, X, y):
        self.train(X, y)

    def train(self, X_img, y):
        assert y.shape == (X_img.shape[0],)
        n_samples = y.shape[0]

        # initialize exponential decay_rate
        self.model.global_step = 0
        self.model.decay_steps = Config.n_epochs

        # calculate checkpoint frequencies
        n_batches = int(n_samples / Config.batch_size)

        # split in minibatch of size batch_size
        for epoch in xrange(Config.n_epochs):
            total_loss, total_grad_norm, batch_count = 0.0, 0.0, 0

            n_batches = int(n_samples / Config.batch_size)
            for iter in xrange(n_batches):
                start_idx = iter * Config.batch_size
                end_idx = min(start_idx + Config.batch_size, n_samples)
                #print("batch size", end_idx - start_idx)

                # prepare input data and label data
                input_feed = self.model.set_input_feed(X_img[start_idx:end_idx], y[start_idx:end_idx], Config.dropout)
                output_feed = [self.model.train_op, self.model.loss, self.model.grad_norm]

                # train this batch
                _, loss, grad_norm = self.session.run(output_feed, input_feed)

                # update cumulative stats
                total_loss += loss
                total_grad_norm += grad_norm
                batch_count += 1

            # increment global step
            self.model.global_step += 1

            # checkpoint the model for each epoch
            if (epoch+1) % Config.n_epochs_per_ckpt == 0:
                save_path = self.model.saver.save(self.session, "%s/model_epoch_%d.ckpt" % (self.train_dir, epoch))
                
            # compute epoch loss
            epoch_loss = total_loss / float(batch_count)
            epoch_grad_norm = total_grad_norm / float(batch_count)
            if (epoch+1) % Config.n_epochs_per_print == 0:
                print("epoch = %d, loss = %6.4f, grad_norm = %6.4f" % (epoch, epoch_loss, epoch_grad_norm))

            # check for early termination
            if epoch_grad_norm < Config.early_term_criterion:
                return
        return

    def predict(self, X_img, y=None):
        n_samples = X_img.shape[0]
        y_output = np.zeros(n_samples, dtype=np.int)

        batch_size = Config.batch_size
        for start_idx in xrange(0, n_samples, batch_size):
            # calculate end_idx
            end_idx = min(start_idx + batch_size, n_samples)
            
            # prepare input data and label data
            input_feed = self.model.set_input_feed(X_img[start_idx:end_idx])
            output_feed = [self.model.evals, self.model.softmax]

            # run returns a numpy ndarray
            evals, _ = self.session.run(output_feed, input_feed) # (batch_size)
            y_output[start_idx:end_idx] = evals
        
        if y is not None:
            accuracy = np.sum(y_output == y) / float(n_samples)
            return y_output, accuracy
        else:
            return y_output
    
    def score(self, X, y):
        _, accuracy = self.predict(X, y)
        return accuracy


class Model(object):
    
    def __init__(self, name="CNN.Model"):
        self.name = name

        # set up weights and biases
        with vs.variable_scope(self.name):

            self.W_conv11 = tf.get_variable("W_conv11", shape=(Config.patch_size, Config.patch_size, 1, Config.n_channels), \
                        dtype=Config.dtype, initializer=tf.contrib.layers.xavier_initializer())

            self.W_conv21 = tf.get_variable("W_conv21", shape=(Config.patch_size, Config.patch_size, Config.n_channels, Config.n_channels), \
                        dtype=Config.dtype, initializer=tf.contrib.layers.xavier_initializer())
 
            self.W_fc = tf.get_variable("W_fc", shape=(Config.fc_input_size, Config.fc_units), \
                        dtype=Config.dtype, initializer=tf.contrib.layers.xavier_initializer())

            self.W_out = tf.get_variable("W_out", shape=(Config.fc_units, Config.n_classes), \
                        dtype=Config.dtype, initializer=tf.contrib.layers.xavier_initializer())

            self.b_conv11 = tf.get_variable("b_conv11", shape=(Config.n_channels,), \
                        dtype=Config.dtype, initializer=tf.constant_initializer(0))

            self.b_conv21 = tf.get_variable("b_conv21", shape=(Config.n_channels,), \
                        dtype=Config.dtype, initializer=tf.constant_initializer(0))

            self.b_fc = tf.get_variable("b_fc", shape=(Config.fc_units,), \
                        dtype=Config.dtype, initializer=tf.constant_initializer(0))
            
            self.b_out = tf.get_variable("b_out", shape=(Config.n_classes,), \
                        dtype=Config.dtype, initializer=tf.constant_initializer(0))

        # placeholders
        self.image_placeholder = tf.placeholder(Config.dtype, shape=(None, Config.height * Config.width))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None))
        self.dropout_placeholder = tf.placeholder(Config.dtype)

        # graph
        with tf.variable_scope(self.name, initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        self.saver = tf.train.Saver()

    def setup_embeddings(self):
        self.x = tf.reshape(self.image_placeholder, shape=(-1, Config.height, Config.width, 1))
        return

    def setup_system(self):

        conv1 = tf.nn.conv2d(self.x, self.W_conv11, strides=Config.conv_strides, padding='SAME')
        conv1 = tf.nn.relu(conv1 + self.b_conv11)
        conv1 = tf.nn.max_pool(conv1, ksize=Config.pool_ksize, strides=Config.pool_strides, padding='SAME')
        
        conv2 = tf.nn.conv2d(conv1, self.W_conv21, strides=Config.conv_strides, padding='SAME')
        conv2 = tf.nn.relu(conv2) + self.b_conv21
        conv2 = tf.nn.max_pool(conv2, ksize=Config.pool_ksize, strides=Config.pool_strides, padding='SAME')

        fc = tf.reshape(conv2, (-1, Config.fc_input_size))
        fc = tf.matmul(fc, self.W_fc) + self.b_fc
        fc = tf.nn.relu(fc)

        fc = tf.nn.dropout(fc, self.dropout_placeholder)

        self.preds = tf.matmul(fc, self.W_out) + self.b_out

        # for evaluation
        self.softmax = tf.nn.softmax(self.preds) # (batch_size, n_classes)
        self.evals = tf.argmax(self.softmax, 1) # (batch_size)

    def setup_loss(self):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(self.preds, self.labels_placeholder)
        self.loss = tf.reduce_mean(losses)
        l2_cost = 0.0
        for var in tf.trainable_variables():
            if len(var.get_shape()) > 1:
                l2_cost += tf.nn.l2_loss(var)
        self.loss += Config.l2_reg * l2_cost

        # lr = starter_lr * decay_rate ^ (global_step / decay_steps)
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.decay_steps = tf.Variable(100, trainable=False, dtype=tf.int32) # temporary value
        self.lr = tf.train.exponential_decay(Config.starter_lr, self.global_step, self.decay_steps, Config.decay_rate, staircase=True)

        # create optimizer
        optimizer = tf.train.AdamOptimizer(self.lr)
        
        # gradient clipping
        grads_and_vars = optimizer.compute_gradients(self.loss, tf.trainable_variables())
        grads = [x[0] for x in grads_and_vars]
        if Config.max_grad_norm > 0.0:
            grads, _ = tf.clip_by_global_norm(grads, Config.max_grad_norm)
        self.grad_norm = tf.global_norm(grads)
        grads_and_vars = [(grads[i], x[1]) for i, x in enumerate(grads_and_vars)]
        self.train_op = optimizer.apply_gradients(grads_and_vars)
   
    def set_input_feed(self, image_batch, y_batch=None, dropout=0.0):
        n_samples = image_batch.shape[0]
        if y_batch is not None:
            assert y_batch.shape == (n_samples,)
        
        input_feed = {}
        input_feed[self.image_placeholder] = image_batch
        input_feed[self.dropout_placeholder] = 1.0 - dropout

        if y_batch is not None:
            input_feed[self.labels_placeholder] = y_batch
        
        return input_feed

