import datetime
import pathlib
import sys
from collections import Counter

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import init_ops
from tensorflow.python.ops.rnn_cell_impl import _Linear, LSTMStateTuple, LSTMCell, GRUCell
from tensorflow.python.ops import variable_scope as vs
from utils import *
from tensorflow.contrib import seq2seq


class RLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    """Basic LSTM recurrent network cell.

    The implementation is based on: http://arxiv.org/abs/1409.2329.

    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.

    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.

    For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
    that follows.
    """

    def __init__(self, num_units, forget_bias=1.0, state_is_tuple=True, activation=None, reuse=None, dense=None):
        """Initialize the basic LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
            Must set to `0.0` manually when restoring from CudnnLSTM-trained
            checkpoints.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.  Default: `tanh`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.

          When restoring from CudnnLSTM-trained checkpoints, must use
          CudnnCompatibleLSTMCell instead.
        """
        super(tf.nn.rnn_cell.BasicLSTMCell, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            tf.logging.warn("%s: Using a concatenated state is slower and will soon be "
                            "deprecated.  Use state_is_tuple=True.", self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or tf.nn.tanh
        self._linear = None
        self._dense = dense

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).

        Args:
          inputs: `2-D` tensor with shape `[batch_size x input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
            `True`.  Otherwise, a `Tensor` shaped
            `[batch_size x 2 * self.state_size]`.

        Returns:
          A pair containing the new hidden state, and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """
        sigmoid = tf.sigmoid

        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

        if self._linear is None:
            self._linear = _Linear([inputs, h], 4 * self._num_units, True)

        i, j, f, o = tf.split(value=self._linear([inputs, h]), num_or_size_splits=4, axis=1)

        new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
        new_h_1 = self._activation(new_c) * sigmoid(o)

        w_h, b_h = self.weight_bias([self._num_units, self._num_units], [self._num_units])
        new_h_2 = sigmoid(tf.matmul(h, w_h) + b_h)

        new_h = new_h_1 + new_h_2

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = tf.concat([new_c, new_h], 1)
        return new_h, new_state

    def weight_bias(self, W_shape, b_shape, bias_init=0.1):
        """Fully connected highway layer adopted from
           https://github.com/fomorians/highway-fcn/blob/master/main.py
        """
        W = tf.get_variable("weight_h_2", shape=W_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable("bias_h_2", shape=b_shape)
        return W, b

class RSLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    """Basic LSTM recurrent network cell.

    The implementation is based on: http://arxiv.org/abs/1409.2329.

    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.

    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.

    For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
    that follows.
    """

    def __init__(self, num_units, forget_bias=1.0, state_is_tuple=True, activation=None, reuse=None, dense=None,
                 file_name='tweet', type='enc', component=1, partition=1):
        """Initialize the basic LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
            Must set to `0.0` manually when restoring from CudnnLSTM-trained
            checkpoints.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.  Default: `tanh`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.

          When restoring from CudnnLSTM-trained checkpoints, must use
          CudnnCompatibleLSTMCell instead.
        """
        super(tf.nn.rnn_cell.BasicLSTMCell, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            tf.logging.warn("%s: Using a concatenated state is slower and will soon be "
                            "deprecated.  Use state_is_tuple=True.", self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or tf.nn.tanh
        self._linear = None
        self._dense = dense
        self._step = 0
        self._file_name = file_name
        self._type = type
        self._component = component
        self._partition = partition

        if not os.path.exists('./weight/' + self._file_name):
            os.makedirs('./weight/' + self._file_name)
        if not os.path.exists('./weight/' + self._file_name + '/' + str(self._partition)):
            os.makedirs('./weight/' + self._file_name + '/' + str(self._partition))
        if not os.path.exists('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component)):
            os.makedirs('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component))
        if not os.path.exists('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component) + '/' + self._type):
            os.makedirs('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component) + '/' + self._type)

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).

        Args:
          inputs: `2-D` tensor with shape `[batch_size x input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
            `True`.  Otherwise, a `Tensor` shaped
            `[batch_size x 2 * self.state_size]`.

        Returns:
          A pair containing the new hidden state, and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """
        sigmoid = tf.sigmoid
        self._step = self._step + 1

        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

        if self._linear is None:
            self._linear = _Linear([inputs, h], 4 * self._num_units, True)

        i, j, f, o = tf.split(value=self._linear([inputs, h]), num_or_size_splits=4, axis=1)

        new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
        new_h_1 = self._activation(new_c) * sigmoid(o)

        w_h, b_h = self.weight_bias([self._num_units, self._num_units], [self._num_units])
        new_h_2 = sigmoid(tf.matmul(h, w_h) + b_h)
        masked_w1, masked_w2 = self.masked_weight(_load=False)

        new_h = new_h_1 * masked_w1 + new_h_2 * masked_w2

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = tf.concat([new_c, new_h], 1)
        return new_h, new_state


    def weight_bias(self, W_shape, b_shape, bias_init=0.1):
        """Fully connected highway layer adopted from
           https://github.com/fomorians/highway-fcn/blob/master/main.py
        """
        W = tf.get_variable("weight_h_2", shape=W_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable("bias_h_2", shape=b_shape)
        return W, b


    def masked_weight(self, _load=False):
        if _load==False:
            masked_W1 = np.random.randint(2, size=1)
            if masked_W1 == 0:
                masked_W2 = 1
            else:
                masked_W2 = np.random.randint(2, size=1)

            np.save('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component) + '/'
                    + self._type + '/W1_' + str(self._step), masked_W1)
            np.save('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component) + '/'
                    + self._type + '/W2_' + str(self._step), masked_W2)
        else:
            masked_W1 = np.load('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component)
                                + '/' + str(self._type) + '/W1_' + str(self._step) + '.npy')
            masked_W2 = np.load('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component)
                                + '/' + str(self._type) + '/W2_' + str(self._step) + '.npy')

        tf_mask_W1 = tf.constant(masked_W1, dtype=tf.float32)
        tf_mask_W2 = tf.constant(masked_W2, dtype=tf.float32)
        return tf_mask_W1, tf_mask_W2


class RKLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    """Basic LSTM recurrent network cell.

    The implementation is based on: http://arxiv.org/abs/1409.2329.

    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.

    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.

    For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
    that follows.
    """

    def __init__(self, num_units, forget_bias=1.0, state_is_tuple=True, activation=None, reuse=None, dense=None,
                 file_name='tweet', type='enc', component=1, partition=1, skip_size=3):
        """Initialize the basic LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
            Must set to `0.0` manually when restoring from CudnnLSTM-trained
            checkpoints.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.  Default: `tanh`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.

          When restoring from CudnnLSTM-trained checkpoints, must use
          CudnnCompatibleLSTMCell instead.
        """
        super(tf.nn.rnn_cell.BasicLSTMCell, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            tf.logging.warn("%s: Using a concatenated state is slower and will soon be "
                            "deprecated.  Use state_is_tuple=True.", self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or tf.nn.tanh
        self._linear = None
        self._dense = dense
        self._step = 0
        self._file_name = file_name
        self._type = type
        self._component = component
        self._partition = partition
        self._skip_size = skip_size

        if not os.path.exists('./weight/' + self._file_name):
            os.makedirs('./weight/' + self._file_name)
        if not os.path.exists('./weight/' + self._file_name + '/' + str(self._partition)):
            os.makedirs('./weight/' + self._file_name + '/' + str(self._partition))
        if not os.path.exists('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component)):
            os.makedirs('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component))
        if not os.path.exists('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component) + '/' + self._type):
            os.makedirs('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component) + '/' + self._type)

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).

        Args:
          inputs: `2-D` tensor with shape `[batch_size x input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
            `True`.  Otherwise, a `Tensor` shaped
            `[batch_size x 2 * self.state_size]`.

        Returns:
          A pair containing the new hidden state, and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """
        sigmoid = tf.sigmoid
        self._step = self._step + 1

        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

        if self._linear is None:
            self._linear = _Linear([inputs, h], 4 * self._num_units, True)

        i, j, f, o = tf.split(value=self._linear([inputs, h]), num_or_size_splits=4, axis=1)

        new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
        new_h_cnt = self._activation(new_c) * sigmoid(o)

        if self._step % self._skip_size == 0:
            w_h_skip, b_h_skip = self.weight_bias([self._num_units, self._num_units], [self._num_units])
            new_h_skip = sigmoid(tf.matmul(h, w_h_skip) + b_h_skip)
            masked_w1, masked_w2 = self.masked_weight(_load=False)
            new_h = new_h_cnt * masked_w1 + new_h_skip * masked_w2

        else:
            new_h = new_h_cnt

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = tf.concat([new_c, new_h], 1)
        return new_h, new_state


    def weight_bias(self, W_shape, b_shape, bias_init=0.1):
        """Fully connected highway layer adopted from
           https://github.com/fomorians/highway-fcn/blob/master/main.py
        """
        W = tf.get_variable("weight_h_2", shape=W_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable("bias_h_2", shape=b_shape)
        return W, b


    def masked_weight(self, _load=False):
        if _load==False:
            masked_W1 = np.random.randint(2, size=1)
            if masked_W1 == 0:
                masked_W2 = 1
            else:
                masked_W2 = np.random.randint(2, size=1)

            np.save('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component) + '/'
                    + self._type + '/W1_' + str(self._step), masked_W1)
            np.save('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component) + '/'
                    + self._type + '/W2_' + str(self._step), masked_W2)
        else:
            masked_W1 = np.load('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component)
                                + '/' + str(self._type) + '/W1_' + str(self._step) + '.npy')
            masked_W2 = np.load('./weight/' + self._file_name + '/' + str(self._partition) + '/' + str(self._component)
                                + '/' + str(self._type) + '/W2_' + str(self._step) + '.npy')

        tf_mask_W1 = tf.constant(masked_W1, dtype=tf.float32)
        tf_mask_W2 = tf.constant(masked_W2, dtype=tf.float32)
        return tf_mask_W1, tf_mask_W2


class RGRUCell(tf.nn.rnn_cell.GRUCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

    Args:
      num_units: int, The number of units in the GRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
    """

    def __init__(self, num_units, activation=None, reuse=None, kernel_initializer=None, bias_initializer=None):
        super(tf.nn.rnn_cell.GRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or tf.nn.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""

        sigmoid = tf.sigmoid

        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates"):  # Reset gate and update gate.
                self._gate_linear = _Linear([inputs, state], 2 * self._num_units, True, bias_initializer=bias_ones, kernel_initializer=self._kernel_initializer)

        value = sigmoid(self._gate_linear([inputs, state]))
        r, u = tf.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear([inputs, r_state], self._num_units, True, bias_initializer=self._bias_initializer, kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        new_h = (u * state + (1 - u) * c)
        return new_h, new_h


    def weight_bias(self, W_shape, b_shape, bias_init=0.1):
        """Fully connected highway layer adopted from
           https://github.com/fomorians/highway-fcn/blob/master/main.py
        """
        W = tf.get_variable("weight_h_2", shape=W_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable("bias_h_2", shape=b_shape)
        return W, b


def Model(_j, _abnormal_data, _abnormal_label, _hidden_num, _elem_num, _file_name, _partition):
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        # placeholder list
        p_input = tf.placeholder(tf.float32, shape=(batch_num, _abnormal_data.shape[1], _abnormal_data.shape[2]))

        # create RNN cell
        if cell_type == 0:
            enc_cell = tf.nn.rnn_cell.BasicRNNCell(_hidden_num)
            dec_cell = tf.nn.rnn_cell.BasicRNNCell(_hidden_num)
        if cell_type == 1:
            pure_enc_cell = LSTMCell(_hidden_num)
            pure_dec_cell = LSTMCell(_hidden_num)
            residual_enc_cell = RLSTMCell(_hidden_num)
            residual_dec_cell = RLSTMCell(_hidden_num)
            enc_cell = RKLSTMCell(_hidden_num, file_name=_file_name, component=_j, partition=_partition, type='enc')
            dec_cell = RKLSTMCell(_hidden_num, file_name=_file_name, component=_j, partition=_partition, type='dec')
        if cell_type == 2:
            pure_enc_cell = GRUCell(_hidden_num)
            pure_dec_cell = GRUCell(_hidden_num)
            enc_cell = RGRUCell(_hidden_num)
            dec_cell = RGRUCell(_hidden_num)

        projection_layer = tf.layers.Dense(units=_elem_num, use_bias=True)

        # with tf.device('/device:GPU:0'):
        with tf.variable_scope('encoder'):
            if _j == 0:
                (enc_output, enc_state) = tf.nn.dynamic_rnn(pure_enc_cell, p_input, dtype=tf.float32)
            elif _j == 1:
                (enc_output, enc_state) = tf.nn.dynamic_rnn(residual_enc_cell, p_input, dtype=tf.float32)
            else:
                (enc_output, enc_state) = tf.nn.dynamic_rnn(enc_cell, p_input, dtype=tf.float32)

        # with tf.device('/device:GPU:1'):
        with tf.variable_scope('decoder') as vs:
            # dec_weight_ = tf.Variable(tf.truncated_normal([hidden_num, elem_num], dtype=tf.float32), name='dec_weight')
            # dec_bias_ = tf.Variable(tf.constant(0.1, shape=[elem_num], dtype=tf.float32), name='dec_bias')
            if decode_without_input:
                dec_input = tf.zeros([p_input.shape[0], p_input.shape[1], p_input.shape[2]], dtype=tf.float32)

                if _j == 0:
                    (dec_outputs, dec_state_) = tf.nn.dynamic_rnn(pure_dec_cell, dec_input, initial_state=enc_state, dtype=tf.float32)
                elif _j == 1:
                    (dec_outputs, dec_state_) = tf.nn.dynamic_rnn(residual_dec_cell, dec_input, initial_state=enc_state, dtype=tf.float32)
                else:
                    (dec_outputs, dec_state_) = tf.nn.dynamic_rnn(dec_cell, dec_input, initial_state=enc_state, dtype=tf.float32)

                if reverse:
                    dec_outputs = dec_outputs[::-1]

            else:
                dec_input = tf.zeros([p_input.shape[0], p_input.shape[2]], dtype=tf.float32)
                inference_helper = tf.contrib.seq2seq.InferenceHelper(
                    sample_fn=lambda outputs: outputs,
                    sample_shape=[_elem_num],
                    sample_dtype=tf.float32,
                    start_inputs=dec_input,
                    end_fn=lambda sample_ids: False)

                if _j == 0:
                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(pure_dec_cell, inference_helper, enc_state, output_layer=projection_layer)
                elif _j == 1:
                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(residual_dec_cell, inference_helper, enc_state, output_layer=projection_layer)
                else:
                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, inference_helper, enc_state, output_layer=projection_layer)

                dec_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder, impute_finished=True, maximum_iterations=p_input.shape[1])

                if reverse:
                    dec_outputs = dec_outputs[::-1]

        loss = tf.reduce_mean(tf.square(p_input - dec_outputs))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
    return g, p_input, dec_outputs, loss, optimizer, saver


def RunModel(_abnormal_data, _abnormal_label, _hidden_num, _elem_num, _file_name, _partition):
    error = []
    for j in range(ensemble_space):
        graph, p_input, dec_outputs, loss, optimizer, saver = Model(j, _abnormal_data, _abnormal_label, _hidden_num, _elem_num, _file_name, _partition)
        config = tf.ConfigProto()

        # config.gpu_options.allow_growth = True
        # config.allow_soft_placement=True

        # Add ops to save and restore all the variables.
        with tf.Session(graph=graph, config=config) as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(iteration):
                """Random sequences.
                  Every sequence has size batch_num * step_num * elem_num 
                  Each step number increases 1 by 1.
                  An initial number of each sequence is in the range from 0 to 19.
                  (ex. [8. 9. 10. 11. 12. 13. 14. 15])
                """

                (loss_val, _) = sess.run([loss, optimizer], {p_input: _abnormal_data})
            if (j%5==0):
                print('iter %d:' % (j*iteration), loss_val)

            if save_model:
                save_path = saver.save(sess, './saved_model/' + pathlib.Path(file_name).parts[0] + '/masked_skip_rnn_seq2seq_' + str(cell_type) + '_' + os.path.basename(file_name) + '.ckpt')
                print("Model saved in path: %s" % save_path)

            (input_, output_) = sess.run([p_input, dec_outputs], {p_input: _abnormal_data})
            error.append(SquareErrorDataPoints(input_, output_[0]))

    ensemble_errors = np.asarray(error)
    anomaly_score = CalculateFinalAnomalyScore(ensemble_errors)
    zscore = Z_Score(anomaly_score)
    y_pred = CreateLabelBasedOnZscore(zscore, 3)
    if not partition:
        score_pred_label = np.c_[ensemble_errors, y_pred, _abnormal_label]
        np.savetxt('./saved_result/' + pathlib.Path(file_name).parts[0] + '/masked_skip_rnn_seq2seq_' + os.path.basename(file_name) + '_score.txt', score_pred_label, delimiter=',')  # X is an array

    precision, recall, f1 = CalculatePrecisionRecallF1Metrics(_abnormal_label, y_pred)
    if not partition:
        PrintPrecisionRecallF1Metrics(precision, recall, f1)

    fpr, tpr, roc_auc = CalculateROCAUCMetrics(_abnormal_label, anomaly_score)
    # PlotROCAUC(fpr, tpr, roc_auc)
    if not partition:
        print('roc_auc=' + str(roc_auc))

    precision_curve, recall_curve, average_precision = CalculatePrecisionRecallCurve(_abnormal_label, anomaly_score)
    # PlotPrecisionRecallCurve(precision_curve, recall_curve, average_precision)
    if not partition:
        print('pr_auc=' + str(average_precision))

    cks = CalculateCohenKappaMetrics(_abnormal_label, y_pred)
    if not partition:
        print('cks=' + str(cks))

    return anomaly_score, precision, recall, f1, roc_auc, average_precision, cks

if __name__ == '__main__':
    batch_num = 1
    hidden_num = 8
    # step_num = 8
    iteration = 30
    learning_rate = 1e-3
    multivariate = True

    reverse = True
    decode_without_input = False

    partition = True
    save_model = False

    # cell type 0 => BasicRNN, 1 => LSTM, 2 => GRU
    cell_type = 1
    ensemble_space = 20

    for n in range(1, 8):
        dataset = 7
        # 1-ionosphere,2-Musk2,3-ISOLET,4-MF-3,5-Arrhythmia,6-MF-5,7-MF-7
        if dataset == 1:
            k_partition = 40
            elem_num = 34
            _file_name = r"data/ionosphere.txt"
            print('当前数据集是：{0}'.format(_file_name))
            t1 = datetime.datetime.now()
            print('从当前时间开始:{0}'.format(t1))
            abnormal_data = np.loadtxt(_file_name, delimiter=",", usecols=np.arange(0, 34))
            abnormal_label = np.loadtxt(_file_name, delimiter=",", usecols=(-1,))
            abnormal_label[abnormal_label == 1] = 1
            abnormal_label[abnormal_label == 0] = -1
            if multivariate:
                abnormal_data = np.expand_dims(abnormal_data, axis=0)
            if partition:
                splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data, abnormal_label, _part_number=k_partition)
                final_error = []
                for i in range(k_partition):
                    print("一共{0}块，这是第{1}块".format(k_partition, i))
                    error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(splitted_data[i], splitted_label[i], hidden_num, elem_num,_file_name=os.path.splitext(os.path.basename(_file_name))[0],_partition=i)
                    final_error.append(error_partition)
                # print('-----------------------------------------')
                final_error = np.concatenate(final_error).ravel()
                final_zscore = Z_Score(final_error)
                print('abnormal_label:{0}'.format(abnormal_label))
                print('anomaly_label:{0}'.format(Counter(abnormal_label)))

                zscore_abs = np.fabs(final_zscore)
                result_temp = []
                temp_list = [5, 10, 30, 60, 90, 120, 130, 140, 150, 200, 300, 340]
                for m in temp_list:
                    count = 0
                    index = np.argpartition(zscore_abs, -m)[-m:]
                    for each_index in index:
                        if abnormal_label[each_index] == -1:
                            count += 1
                    result_temp.append(count)
                print('result_temp:{0}'.format(result_temp))
            else:
                error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label, hidden_num, elem_num)
            t2 = datetime.datetime.now()
            print('从当前时间结束:{0}'.format(t2))
            print('一共用时：{0}'.format(t2 - t1))
            print('########################################')

        if dataset == 2:
            k_partition = 80
            elem_num = 166
            _file_name = r"data/clean2.data"
            print('当前数据集是：{0}'.format(_file_name))
            t1 = datetime.datetime.now()
            print('从当前时间开始:{0}'.format(t1))
            X = pd.read_csv(_file_name, header=None, index_col=None, skiprows=0, sep=',')
            abnormal_data = X.iloc[:, 2:168].values
            abnormal_label = X.iloc[:, 168].values
            abnormal_label[abnormal_label == 1] = -1
            abnormal_label[abnormal_label == 0] = 1
            if multivariate:
                abnormal_data = np.expand_dims(abnormal_data, axis=0)
            if partition:
                splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data, abnormal_label, _part_number=k_partition)
                final_error = []
                for i in range(k_partition):
                    print("一共{0}块，这是第{1}块".format(k_partition, i))
                    error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(splitted_data[i], splitted_label[i], hidden_num, elem_num,_file_name=os.path.splitext(os.path.basename(_file_name))[0],_partition=i)
                    final_error.append(error_partition)
                # print('-----------------------------------------')
                final_error = np.concatenate(final_error).ravel()
                final_zscore = Z_Score(final_error)
                print('abnormal_label:{0}'.format(abnormal_label))
                print('anomaly_label:{0}'.format(Counter(abnormal_label)))

                zscore_abs = np.fabs(final_zscore)
                result_temp = []
                temp_list = [1000, 2000, 3000, 4000, 5000, 6000, 6598]
                for m in temp_list:
                    count = 0
                    index = np.argpartition(zscore_abs, -m)[-m:]
                    for each_index in index:
                        if abnormal_label[each_index] == -1:
                            count += 1
                    result_temp.append(count)
                print('result_temp:{0}'.format(result_temp))
            else:
                error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label, hidden_num, elem_num)
            t2 = datetime.datetime.now()
            print('从当前时间结束:{0}'.format(t2))
            print('一共用时：{0}'.format(t2 - t1))
            print('########################################')

        if dataset == 3:
            k_partition = 120
            elem_num = 617
            _file_name = r"data/ISOLET-23/data_23.dat"
            print('当前数据集是：{0}'.format(_file_name))
            t1 = datetime.datetime.now()
            print('从当前时间开始:{0}'.format(t1))
            X = pd.read_csv(_file_name, header=None, index_col=None, skiprows=0, sep=',')
            abnormal_data = X.iloc[:, 0:167].values

            y_loc = r"data/ISOLET-23/classid_23.dat"
            y = pd.read_csv(y_loc, header=None, index_col=None, skiprows=0, sep=',')
            abnormal_label = y.iloc[:, 0].values
            abnormal_label[abnormal_label != 23] = 1
            abnormal_label[abnormal_label == 23] = -1

            if multivariate:
                abnormal_data = np.expand_dims(abnormal_data, axis=0)
            if partition:
                splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data,
                                                                         abnormal_label,
                                                                         _part_number=k_partition)
                final_error = []
                for i in range(k_partition):
                    print("一共{0}块，这是第{1}块".format(k_partition, i))
                    error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(
                        splitted_data[i], splitted_label[i], hidden_num, elem_num, _file_name=os.path.splitext(os.path.basename(_file_name))[0],_partition=i)
                    final_error.append(error_partition)
                # print('-----------------------------------------')
                final_error = np.concatenate(final_error).ravel()
                final_zscore = Z_Score(final_error)
                print('abnormal_label:{0}'.format(abnormal_label))
                print('anomaly_label:{0}'.format(Counter(abnormal_label)))

                zscore_abs = np.fabs(final_zscore)
                result_temp = []
                temp_list = [5, 10, 15, 20, 30, 50, 60, 80, 100, 150]
                for m in temp_list:
                    count = 0
                    index = np.argpartition(zscore_abs, -m)[-m:]
                    for each_index in index:
                        if abnormal_label[each_index] == -1:
                            count += 1
                    result_temp.append(count)
                print('result_temp:{0}'.format(result_temp))
            else:
                error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label, hidden_num, elem_num)

            t2=datetime.datetime.now()
            print('从当前时间结束:{0}'.format(t2))
            print('一共用时：{0}'.format(t2 - t1))
            print('########################################')

        if dataset == 4:
            k_partition=80
            elem_num = 649
            _file_name = r"data/MF-3/data_3.dat"
            print('当前数据集是：{0}'.format(_file_name))
            t1 = datetime.datetime.now()
            print('从当前时间开始:{0}'.format(t1))
            X = pd.read_csv(_file_name, header=None, index_col=None, skiprows=0, sep=',')
            abnormal_data = X.iloc[:, :649].values

            y_loc = r"data/MF-3/classid_3.dat"
            y = pd.read_csv(y_loc, header=None, index_col=None, skiprows=0, sep=',')
            abnormal_label = y.iloc[:, 0].values
            abnormal_label[abnormal_label != 3] = 1
            abnormal_label[abnormal_label == 3] = -1

            if multivariate:
                abnormal_data = np.expand_dims(abnormal_data, axis=0)
            if partition:
                splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data,
                                                                         abnormal_label,
                                                                         _part_number=k_partition)
                final_error = []
                for i in range(k_partition):
                    print("一共{0}块，这是第{1}块".format(k_partition, i))
                    error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(
                        splitted_data[i], splitted_label[i], hidden_num, elem_num, _file_name=os.path.splitext(os.path.basename(_file_name))[0],_partition=i)
                    final_error.append(error_partition)
                # print('-----------------------------------------')
                final_error = np.concatenate(final_error).ravel()
                final_zscore = Z_Score(final_error)
                print('abnormal_label:{0}'.format(abnormal_label))
                print('anomaly_label:{0}'.format(Counter(abnormal_label)))

                zscore_abs = np.fabs(final_zscore)
                result_temp = []
                temp_list = [20, 30, 50, 60, 90, 100, 150]
                for m in temp_list:
                    count = 0
                    index = np.argpartition(zscore_abs, -m)[-m:]
                    for each_index in index:
                        if abnormal_label[each_index] == -1:
                            count += 1
                    result_temp.append(count)
                print('result_temp:{0}'.format(result_temp))
            else:
                error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data,
                                                                              abnormal_label,
                                                                              hidden_num, elem_num)
            t2 = datetime.datetime.now()
            print('从当前时间结束:{0}'.format(t2))
            print('一共用时：{0}'.format(t2 - t1))
            print('########################################')


        if dataset == 5:
            k_partition = 80
            elem_num = 260
            _file_name = r"data/Arrhythmia_withoutdupl_05_v03.dat"
            print('当前数据集是：{0}'.format(_file_name))
            t1 = datetime.datetime.now()
            print('从当前时间开始:{0}'.format(t1))
            X = pd.read_csv(_file_name, header=None, index_col=None, skiprows=0, sep=' ')
            abnormal_data = X.iloc[:, :260].values
            abnormal_label = X.iloc[:, 260].values
            abnormal_label[abnormal_label == 1] = -1
            abnormal_label[abnormal_label == 0] = 1

            if multivariate:
                abnormal_data = np.expand_dims(abnormal_data, axis=0)
            if partition:
                splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data, abnormal_label,
                                                                         _part_number=k_partition)
                final_error = []
                for i in range(k_partition):
                    print("一共{0}块，这是第{1}块".format(k_partition, i))
                    error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, pr_cks = RunModel(
                        splitted_data[i], splitted_label[i], _file_name=os.path.splitext(os.path.basename(_file_name))[0],_partition=i)
                    final_error.append(error_partition)
                # print('-----------------------------------------')
                final_error = np.concatenate(final_error).ravel()
                final_zscore = Z_Score(final_error)
                print('abnormal_label:{0}'.format(abnormal_label))
                print('anomaly_label:{0}'.format(Counter(abnormal_label)))

                zscore_abs = np.fabs(final_zscore)
                result_temp = []
                temp_list = [5, 10, 15, 25, 30, 35, 45, 50, 55, 60, 80, 90, 100, 110, 120, 140, 150, 160, 170, 180, 190,
                             200]
                for m in temp_list:
                    count = 0
                    index = np.argpartition(zscore_abs, -m)[-m:]
                    for each_index in index:
                        if abnormal_label[each_index] == -1:
                            count += 1
                    result_temp.append(count)
                print('result_temp:{0}'.format(result_temp))
            else:
                error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label,
                                                                              hidden_num, elem_num)

            t2 = datetime.datetime.now()
            print('从当前时间结束:{0}'.format(t2))
            print('一共用时：{0}'.format(t2 - t1))
            print('########################################')

        if dataset == 6:
            k_partition = 80
            elem_num = 649
            _file_name = r"data/MF-5/data_5.dat"
            print('当前数据集是：{0}'.format(_file_name))
            t1 = datetime.datetime.now()
            print('从当前时间开始:{0}'.format(t1))
            X = pd.read_csv(_file_name, header=None, index_col=None, skiprows=0, sep=',')
            abnormal_data = X.iloc[:, :649].values
            y_loc = r"data/MF-5/classid_5.dat"
            y = pd.read_csv(y_loc, header=None, index_col=None, skiprows=0, sep=',')
            abnormal_label = y.iloc[:, 0].values
            abnormal_label[abnormal_label != 5] = 1
            abnormal_label[abnormal_label == 5] = -1

            if multivariate:
                abnormal_data = np.expand_dims(abnormal_data, axis=0)
            if partition:
                splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data, abnormal_label,
                                                                         _part_number=k_partition)
                final_error = []
                for i in range(k_partition):
                    print("一共{0}块，这是第{1}块".format(k_partition, i))
                    error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(
                        splitted_data[i], splitted_label[i], hidden_num, elem_num,_file_name=os.path.splitext(os.path.basename(_file_name))[0],_partition=i)
                    final_error.append(error_partition)
                # print('-----------------------------------------')
                final_error = np.concatenate(final_error).ravel()
                final_zscore = Z_Score(final_error)
                print('abnormal_label:{0}'.format(abnormal_label))
                print('anomaly_label:{0}'.format(Counter(abnormal_label)))

                zscore_abs = np.fabs(final_zscore)
                result_temp = []
                temp_list = [20, 30, 50, 60, 70, 100, 150]
                for m in temp_list:
                    count = 0
                    index = np.argpartition(zscore_abs, -m)[-m:]
                    for each_index in index:
                        if abnormal_label[each_index] == -1:
                            count += 1
                    result_temp.append(count)
                print('result_temp:{0}'.format(result_temp))
            else:
                error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label,
                                                                              hidden_num, elem_num)
            t2 = datetime.datetime.now()
            print('从当前时间结束:{0}'.format(t2))
            print('一共用时：{0}'.format(t2 - t1))
            print('########################################')

        if dataset == 7:
            k_partition=80
            elem_num = 649
            _file_name = r"data/MF-7/data_7.dat"
            print('当前数据集是：{0}'.format(_file_name))
            t1 = datetime.datetime.now()
            print('从当前时间开始:{0}'.format(t1))
            X = pd.read_csv(_file_name, header=None, index_col=None, skiprows=0, sep=',')
            abnormal_data = X.iloc[:, :649].values
            y_loc = r"data/MF-7/classid_7.dat"
            y = pd.read_csv(y_loc, header=None, index_col=None, skiprows=0, sep=',')
            abnormal_label = y.iloc[:, 0].values
            abnormal_label[abnormal_label != 7] = 1
            abnormal_label[abnormal_label == 7] = -1

            if multivariate:
                abnormal_data = np.expand_dims(abnormal_data, axis=0)
            if partition:
                splitted_data, splitted_label = PartitionTimeSeriesKPart(abnormal_data,
                                                                         abnormal_label,
                                                                         _part_number=k_partition)
                final_error = []
                for i in range(k_partition):
                    print("一共{0}块，这是第{1}块".format(k_partition, i))
                    error_partition, precision_partition, recall_partition, f1_partition, roc_auc_partition, pr_auc_partition, cks = RunModel(
                        splitted_data[i], splitted_label[i], hidden_num, elem_num,
                        _file_name=os.path.splitext(os.path.basename(_file_name))[0],_partition=i)
                    final_error.append(error_partition)
                # print('-----------------------------------------')
                final_error = np.concatenate(final_error).ravel()
                final_zscore = Z_Score(final_error)
                print('abnormal_label:{0}'.format(abnormal_label))
                print('anomaly_label:{0}'.format(Counter(abnormal_label)))

                zscore_abs = np.fabs(final_zscore)
                result_temp = []
                temp_list = [20, 30, 50, 60, 90, 100, 150]
                for m in temp_list:
                    count = 0
                    index = np.argpartition(zscore_abs, -m)[-m:]
                    for each_index in index:
                        if abnormal_label[each_index] == -1:
                            count += 1
                    result_temp.append(count)
                print('result_temp:{0}'.format(result_temp))
            else:
                error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data,
                                                                              abnormal_label,
                                                                              hidden_num, elem_num)
            t2 = datetime.datetime.now()
            print('从当前时间结束:{0}'.format(t2))
            print('一共用时：{0}'.format(t2 - t1))
            print('########################################')