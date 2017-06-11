from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, random
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs


class Config:
    random_seed = 101
    attention = False # attention does work well

    seq_length = 200 ########
    embed_size1 = 52 #### 26  
    n_inputs = 1
    embed_size2 = None
    embed_size3 = None
    n_classes = 5  
    state_size = 50 ############# seq_length
 
    n_epochs = 100
    batch_size = 10
    early_term_criterion = 1.0e-6
    n_epochs_per_ckpt = 10

    dtype = tf.float64
    decay_rate = 0.95 # decay per epoch
    max_grad_norm = 10.0

    # hyperparameters
    starter_lr =  0.0001 #0.001
    dropout = 0.1 # 0.10
    l2_reg = 0.001


class LstmNeuralNetwork(object):

    def __init__(self, vector=None, state_size=None, train_dir=None, seed=101):
        # set random seed
        random.seed(Config.random_seed)
        np.random.seed(Config.random_seed)
        tf.set_random_seed(Config.random_seed)

        # update train_dir
        if train_dir is None:
            train_dir = "data/train"

        # update config
        if vector is not None:
            if len(vector) == 2:
                Config.n_inputs = 1
                Config.seq_length, Config.embed_size1 = vector

            elif len(vector) == 3:
                Config.n_inputs = 2
                Config.seq_length, Config.embed_size1, Config.embed_size2 = vector
            
            elif len(vector) == 4:
                Config.n_inputs = 3
                Config.seq_length, Config.embed_size1, Config.embed_size2, Config.embed_size3 = vector
        
        # update state size
        if state_size is not None:
            Config.state_size = state_size

        # print config
        print("seq_length =", Config.seq_length)
        print("embed_size1 =", Config.embed_size1)
        print("embed_size2 =", Config.embed_size2)
        print("embed_size3 =", Config.embed_size3)
        print("n_classes =", Config.n_classes)
        print("state_size =", Config.state_size)
        print("starter_lr =", Config.starter_lr)
        print("dropout =", Config.dropout)
        print("l2_reg =", Config.l2_reg)
        
        # create neural network
        self.model = NeuralNetworkModel(bilstm=True)

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

    def unpack_X(self, X):
        if isinstance(X, tuple) or isinstance(X, list):
            if len(X) == 1:
                X1, X2, X3 = X[0], None, None
            elif len(X) == 2:
                X1, X2, X3 = X[0], X[1], None
            elif len(X) == 3:
                X1, X2, X3 = X[0], X[1], X[2]
            else:
                raise Exception("Too many items in the X tuple", X)
        else:
            X1, X2, X3 = X, None, None
        return X1, X2, X3

    def fit(self, X, y):
        # check input
        n_samples = y.shape[0]
        X1, X2, X3 = self.unpack_X(X)
        assert y.shape == (X1.shape[0],)

        # initialize exponential decay_rate
        self.model.global_step = 0
        self.model.decay_steps = Config.n_epochs

        # calculate checkpoint frequencies
        n_batches = int(n_samples / Config.batch_size)

        # split in minibatch of size batch_size
        for epoch in xrange(Config.n_epochs):
            total_loss, total_grad_norm, batch_count = 0.0, 0.0, 0

            for start_idx in xrange(0, n_samples, Config.batch_size):
                end_idx = min(start_idx + Config.batch_size, n_samples)

                # prepare input data and label data
                y_batch = y[start_idx:end_idx]
                X1_batch = X1[start_idx:end_idx]
                X2_batch = X2[start_idx:end_idx] if X2 is not None else None
                X3_batch = X3[start_idx:end_idx] if X3 is not None else None
                input_feed = self.model.set_input_feed(X1_batch, y_batch, Config.dropout, X2_batch=X2_batch, X3_batch=X3_batch)
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
            print("epoch = %d, loss = %6.4f, grad_norm = %6.4f" % (epoch, epoch_loss, epoch_grad_norm))

            # check for early termination
            if epoch_grad_norm < Config.early_term_criterion:
                print("EARLY TERMINATION")
                return
        print("Training is done")

    def predict(self, X):
        X1, X2, X3 = self.unpack_X(X)
        n_samples = X1.shape[0]
        y_output = np.zeros(n_samples, dtype=np.int)

        batch_size = Config.batch_size
        for start_idx in xrange(0, n_samples, batch_size):
            # calculate end_idx
            end_idx = min(start_idx + batch_size, n_samples)
            
            # prepare input data and label data
            X1_batch = X1[start_idx:end_idx]
            X2_batch = X2[start_idx:end_idx] if X2 is not None else None
            X3_batch = X3[start_idx:end_idx] if X3 is not None else None
            input_feed = self.model.set_input_feed(X1_batch, X2_batch=X2_batch, X3_batch=X3_batch)
            output_feed = [self.model.evals, self.model.softmax]

            # run returns a numpy ndarray
            evals, _ = self.session.run(output_feed, input_feed) # (batch_size)
            y_output[start_idx:end_idx] = evals
        
        self.X = X
        self.y = y_output
        return y_output

    def score(self, X, y):
        n_samples = y.shape[0]
        X1, X2, X3 = self.unpack_X(X)
        assert y.shape == (X1.shape[0],)

        if self.X is not X:
            self.predict(X)
        
        accuracy = np.sum(self.y == y) / float(n_samples)
        return accuracy


class NeuralNetworkModel(object):
    
    def __init__(self, name="NeuralNework", bilstm=True):
        self.name = name
        if bilstm:
            self.encoder = Encoder2(self.name)
            self.decoder = Decoder2(self.name)
        else:
            self.encoder = Encoder(self.name)
            self.decoder = Decoder(self.name)

        # placeholders
        self.input_X1_placeholder = tf.placeholder(Config.dtype, shape=(None, Config.seq_length * Config.embed_size1))
        self.input_X2_placeholder, self.input_X3_placeholder = None, None
        if Config.n_inputs >= 2:
            self.input_X2_placeholder = tf.placeholder(Config.dtype, shape=(None, Config.seq_length * Config.embed_size2))
        if Config.n_inputs >= 3:
            self.input_X3_placeholder = tf.placeholder(Config.dtype, shape=(None, Config.seq_length * Config.embed_size3))

        self.input_seq_length_placeholder = tf.placeholder(tf.int32, shape=(None))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None))
        self.dropout_placeholder = tf.placeholder(Config.dtype)

        # graph
        with tf.variable_scope(self.name, initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        self.saver = tf.train.Saver()

    def setup_embeddings(self):
        self.x1 = tf.reshape(self.input_X1_placeholder, [-1, Config.seq_length, Config.embed_size1]) # (batch_size, seq_length, embed_size1)
        self.x2, self.x3 = None, None
        if self.input_X2_placeholder is not None:
            self.x2 = tf.reshape(self.input_X2_placeholder, [-1, Config.seq_length, Config.embed_size2]) # (batch_size, seq_length, embed_size2)
        if self.input_X3_placeholder is not None:
            self.x3 = tf.reshape(self.input_X3_placeholder, [-1, Config.seq_length, Config.embed_size3]) # (batch_size, seq_length, embed_size3)

        self.seq_length = self.input_seq_length_placeholder

    def setup_system(self):
        # connect components together
        encoded = self.encoder.encode(self.x1, self.seq_length, x2=self.x2, x3=self.x3)
        self.preds = self.decoder.decode(encoded, self.seq_length, self.dropout_placeholder) # (batch_size, n_classes)

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
        self.lr = tf.train.exponential_decay(Config.starter_lr,
                            self.global_step, self.decay_steps,
                            Config.decay_rate, staircase=True)

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
   
    def set_input_feed(self, X1_batch, y_batch=None, dropout=0.0, X2_batch=None, X3_batch=None):
        seq_length = np.array([Config.seq_length for _ in xrange(X1_batch.shape[0])])
        
        input_feed = {}
        input_feed[self.input_X1_placeholder] = X1_batch
        input_feed[self.input_seq_length_placeholder] = seq_length
        input_feed[self.dropout_placeholder] = 1.0 - dropout
        
        if y_batch is not None:
            input_feed[self.labels_placeholder] = y_batch

        if X2_batch is not None:
            input_feed[self.input_X2_placeholder] = X2_batch
        if X3_batch is not None:
            input_feed[self.input_X3_placeholder] = X3_batch
        
        return input_feed



class Encoder2(object):

    def __init__(self, name="algo"):
        self.name = name + ".Encoder2"
        self.cell1 = Lstm2(Config.state_size, name=self.name)
        if Config.n_inputs >= 2:
            self.cell2 = Lstm2(Config.state_size, name=self.name+"2")
        if Config.n_inputs >= 3:
            self.cell3 = Lstm2(Config.state_size, name=self.name+"3")

    def encode(self, x1, seq_length, x2=None, x3=None):
        output_fw, output_bw, state_fw, state_bw = self.cell1.run(x1, seq_length)
        outputs = [output_fw, output_bw]
        if Config.attention:
            vect1 = tf.concat_v2((output_fw, output_bw), 2)
        if x2 is not None:
            assert Config.n_inputs >= 2
            output_fw, output_bw, state_fw, state_bw = self.cell2.run(x2, seq_length)
            outputs.extend([output_fw, output_bw])
            if Config.attention:
                vect2 = tf.concat_v2((output_fw, output_bw), 2)
                outputs.append(self.attention(vect2, vect1))
        if x3 is not None:
            assert Config.n_inputs >= 3
            output_fw, output_bw, state_fw, state_bw = self.cell3.run(x3, seq_length)
            outputs.extend([output_fw, output_bw])

        encoded = tf.concat_v2(outputs, 2) # (batch_size, seq_length, 2 * state_size)
        return encoded

    def attention(self, Q, P):
        # affinity matrix
        Q_T = tf.transpose(Q, [0, 2, 1]) # (batch_size, n_units, q_max_length)
        L = tf.matmul(P, Q_T) # (batch_size, p_max_length, q_max_length)

        # attention weights of Q (row-wise normalization)
        A_Q = tf.nn.softmax(L) # (batch_size, p_max_length, q_max_length)

        # attention weights of P (column-wise normalization)
        A_P = tf.nn.softmax(tf.transpose(L, [0, 2, 1]))# (batch_size, q_max_length, p_max_length)

        # summary
        S_Q = tf.matmul(tf.transpose(A_Q, [0, 2, 1]), P) # (batch_size, q_max_length, n_units)
        new_Q_states = tf.concat_v2([Q, S_Q], 2) # (batch_size, q_max_length, 2 * n_units)
        S_P = tf.matmul(tf.transpose(A_P, [0, 2, 1]), new_Q_states) # (batch_size, p_max_length, 2 * n_units)
        return S_P # (batch_size, p_max_length, 3 * n_units)


class Decoder2(object):

    def __init__(self, name):
        self.name = name + ".Decoder2"
        self.cell = Lstm2(Config.state_size, name=self.name)

        with vs.variable_scope(self.name):
            self.W = tf.get_variable("affine.weight", shape=(Config.seq_length * 2 * Config.state_size, Config.n_classes),
                                        dtype=Config.dtype, initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.get_variable("affine.bias", shape=(Config.n_classes),
                                        dtype=Config.dtype, initializer=tf.constant_initializer(0))

    def decode(self, encoded, seq_length, dropout=1.0):
        output_fw, output_bw, state_fw, state_bw = self.cell.run(encoded, seq_length)
        x = tf.concat_v2([output_fw, output_bw], 2) # (batch_size, seq_length, 2 * state_size)
        x = tf.reshape(x, [-1, Config.seq_length * 2 * Config.state_size])

        # output layer
        out_drop = tf.nn.dropout(x, dropout) # (batch_size, seq_length * 2 * state_size)
        preds = tf.matmul(out_drop, self.W) + self.b # (batch_size, n_classes)        
        return preds



class Lstm2(object): # bidirectional LSTM

    def __init__(self, n_units, name="algo"):
        self.name = name + ".Lstm2"
        with vs.variable_scope(self.name):
            self.cell_fw = tf.nn.rnn_cell.LSTMCell(n_units, state_is_tuple=True)
            self.cell_bw = tf.nn.rnn_cell.LSTMCell(n_units, state_is_tuple=True)

    def run(self, x, seq_length, init_state_fw=None, init_state_bw=None):
        with vs.variable_scope(self.name):
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                                    self.cell_fw, self.cell_bw, x, dtype=Config.dtype,
                                    sequence_length=seq_length,
                                    initial_state_fw=init_state_fw,
                                    initial_state_bw=init_state_bw)

        output_fw, output_bw = outputs
        state_fw, state_bw = states
        return (output_fw, output_bw, state_fw, state_bw)


