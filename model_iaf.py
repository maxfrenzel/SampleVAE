import tensorflow as tf


def create_variable(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


def create_bias_variable(name, shape):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value=0.001, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)


def upsample(value, name, factor=[2, 2]):
    size = [int(value.shape[1] * factor[0]), int(value.shape[2] * factor[1])]
    with tf.name_scope(name):
        out = tf.image.resize_bilinear(value, size=size, align_corners=None, name=None)
        return out


def upsample2(value, name, output_shape):
    size = [int(output_shape[1]), int(output_shape[2])]
    with tf.name_scope(name):
        out = tf.image.resize_bilinear(value, size=size, align_corners=None, name=None)
        return out


def two_d_conv(value, filter_, pool_kernel=[2, 2], name='two_d_conv'):
    out = tf.nn.conv2d(value, filter_, strides=[1, 1, 1, 1], padding='SAME')
    out = tf.contrib.layers.max_pool2d(out, pool_kernel)

    return out


def two_d_deconv(value, filter_, deconv_shape, pool_kernel=[2, 2], name='two_d_conv'):
    out = upsample2(value, 'unpool', deconv_shape)
    # print(out)
    out = tf.nn.conv2d_transpose(out, filter_, output_shape=deconv_shape, strides=[1, 1, 1, 1], padding='SAME')
    # print(out)

    return out

# KL divergence between posterior with autoregressive flow and prior
def kl_divergence(sigma, epsilon, z_K, param, batch_mean=True):
    # logprob of posterior
    log_q_z0 = -0.5 * tf.square(epsilon)

    # logprob of prior
    log_p_zK = 0.5 * tf.square(z_K)

    # Terms from each flow layer
    flow_loss = 0
    for l in range(param['iaf_flow_length'] + 1):
        # Make sure it can't take log(0) or log(neg)
        flow_loss -= tf.log(sigma[l] + 1e-10)

    kl_divs = tf.identity(log_q_z0 + flow_loss + log_p_zK)
    kl_divs_reduced = tf.reduce_sum(kl_divs, axis=1)

    if batch_mean:
        return tf.reduce_mean(kl_divs, axis=0), tf.reduce_mean(kl_divs_reduced)
    else:
        return kl_divs, kl_divs_reduced


class VAEModel(object):

    def __init__(self,
                 param,
                 batch_size,
                 num_categories=0,
                 num_classes=[],
                 activation=tf.nn.elu,
                 activation_conv=tf.nn.elu,
                 activation_nf=tf.nn.elu,
                 keep_prob=1.0,
                 encode=False):

        self.param = param
        self.batch_size = batch_size
        self.num_categories = num_categories
        self.num_classes = num_classes
        self.activation = activation
        self.activation_conv = activation_conv
        self.activation_nf = activation_nf
        self.encode = encode
        self.layers_enc = len(param['conv_channels'])
        self.layers_dec = self.layers_enc
        # TODO: Conv out shape cannot be hardcoded here
        if self.param['sample_sec'] == 2:
            self.conv_out_shape = [7, 7]
        elif self.param['sample_sec'] == 15:
            self.conv_out_shape = [7, 57]
        else:
            raise Exception(f"No convolution out-shape pre-defined for {self.param['sample_sec']} sample length!")
        self.conv_out_units = self.conv_out_shape[0] * self.conv_out_shape[1] * param['conv_channels'][-1]
        self.cells_hidden = param['cells_hidden']
        self.dim_latent = param['dim_latent']
        if "dim_latent_cat" in self.param.keys():
            for n_dims in self.param['dim_latent_cat']:
                if n_dims > 0:
                    self.dim_latent += n_dims
        if 'rnn_decoder' in param.keys():
            self.rnn_decoder = param['rnn_decoder']
        else:
            self.rnn_decoder = False
        if 'rnn_highway' in param.keys():
            self.rnn_highway = param['rnn_highway']
        else:
            self.rnn_highway = False
        self.keep_prob = keep_prob

        self.variables = self._create_variables()

    def _create_variables(self):
        '''This function creates all variables used by the network.
        This allows us to share them between multiple calls to the loss
        function and generation function.'''

        var = dict()

        with tf.variable_scope('VAE'):

            with tf.variable_scope("Encoder"):

                var['encoder_conv'] = list()
                with tf.variable_scope('conv_stack'):

                    for l in range(self.layers_enc):

                        with tf.variable_scope('layer{}'.format(l)):
                            current = dict()

                            if l == 0:
                                channels_in = 1
                            else:
                                channels_in = self.param['conv_channels'][l - 1]
                            channels_out = self.param['conv_channels'][l]

                            current['filter'] = create_variable("filter",
                                                                [3, 3, channels_in, channels_out])
                            #                             current['bias'] = create_bias_variable("bias",
                            #                                               [channels_out])
                            var['encoder_conv'].append(current)

                with tf.variable_scope('fully_connected'):

                    layer = dict()

                    num_cells_hidden = self.cells_hidden

                    if "cells_hidden_cat" in self.param.keys():
                        for n_cells in self.param['cells_hidden_cat']:
                            if n_cells > 0:
                                num_cells_hidden += n_cells

                    layer['W_z0'] = create_variable("W_z0",
                                                    shape=[self.conv_out_units, num_cells_hidden])
                    layer['b_z0'] = create_bias_variable("b_z0",
                                                         shape=[1, num_cells_hidden])

                    layer['W_mu'] = create_variable("W_mu",
                                                    shape=[self.cells_hidden, self.param['dim_latent']])
                    layer['W_logvar'] = create_variable("W_logvar",
                                                        shape=[self.cells_hidden, self.param['dim_latent']])
                    layer['b_mu'] = create_bias_variable("b_mu",
                                                         shape=[1, self.param['dim_latent']])
                    layer['b_logvar'] = create_bias_variable("b_logvar",
                                                             shape=[1, self.param['dim_latent']])

                    # Weights for latent space dimensions conditioned on categories
                    if 'dim_latent_cat' in self.param.keys():
                        for k, n_dims in enumerate(self.param['dim_latent_cat']):
                            layer[f'W_mu_{k}'] = create_variable(f"W_mu_{k}",
                                                            shape=[self.param['cells_hidden_cat'][k], n_dims])
                            layer[f'W_logvar_{k}'] = create_variable(f"W_logvar_{k}",
                                                                shape=[self.param['cells_hidden_cat'][k], n_dims])
                            layer[f'b_mu_{k}'] = create_bias_variable(f"b_mu_{k}",
                                                                 shape=[1, n_dims])
                            layer[f'b_logvar_{k}'] = create_bias_variable(f"b_logvar_{k}",
                                                                     shape=[1, n_dims])

                    var['encoder_fc'] = layer

            with tf.variable_scope("Classifier"):

                var['classifier'] = list()

                for c in range(self.num_categories):

                    with tf.variable_scope('category{}'.format(c)):
                        category_layers = list()

                        # Hidden layers
                        for l in range(len(self.param['predictor_units'][c]) + 1):
                            with tf.variable_scope('layer{}'.format(l)):

                                layer = dict()

                                if l == 0:
                                    if 'cells_hidden_cat' in self.param.keys() and self.param['cells_hidden_cat'][c] > 0:
                                        units_in = self.param['cells_hidden_cat'][c]
                                    else:
                                        units_in = self.cells_hidden
                                    units_out = self.param['predictor_units'][c][l]
                                # On final layer, match number of classes
                                elif l == len(self.param['predictor_units'][c]):
                                    units_in = self.param['predictor_units'][c][l - 1]
                                    units_out = self.num_classes[c]
                                else:
                                    units_in = self.param['predictor_units'][c][l - 1]
                                    units_out = self.param['predictor_units'][c][l]

                                layer['W'] = create_variable("W",
                                                             shape=[units_in, units_out])
                                layer['b'] = create_bias_variable("b",
                                                                  shape=[1, units_out])

                                category_layers.append(layer)

                        var['classifier'].append(category_layers)

            with tf.variable_scope("IAF"):

                var['iaf_flows'] = list()
                for l in range(self.param['iaf_flow_length']):

                    with tf.variable_scope('layer{}'.format(l)):

                        layer = dict()

                        # Hidden state
                        layer['W_flow'] = create_variable("W_flow",
                                                        shape=[self.conv_out_units, self.dim_latent])
                        layer['b_flow'] = create_bias_variable("b_flow",
                                                             shape=[1, self.dim_latent])

                        flow_variables = list()
                        # Flow parameters from hidden state (m and s parameters for IAF)
                        for j in range(self.dim_latent):
                            with tf.variable_scope('flow_layer{}'.format(j)):

                                flow_layer = dict()

                                # Set correct dimensionality
                                units_to_hidden_iaf = self.param['dim_autoregressive_nl']

                                flow_layer['W_flow_params_nl'] = create_variable("W_flow_params_nl",
                                                                  shape=[self.dim_latent + j, units_to_hidden_iaf])
                                flow_layer['b_flow_params_nl'] = create_bias_variable("b_flow_params_nl",
                                                                       shape=[1, units_to_hidden_iaf])

                                flow_layer['W_flow_params'] = create_variable("W_flow_params",
                                                                                 shape=[units_to_hidden_iaf,
                                                                                        2])
                                flow_layer['b_flow_params'] = create_bias_variable("b_flow_params",
                                                                                      shape=[1, 2])

                                flow_variables.append(flow_layer)

                        layer['flow_vars'] = flow_variables

                        var['iaf_flows'].append(layer)


            with tf.variable_scope("Decoder"):

                with tf.variable_scope('fully_connected'):
                    layer = dict()

                    layer['W_z'] = create_variable("W_z",
                                                   shape=[self.dim_latent, self.conv_out_units])
                    layer['b_z'] = create_bias_variable("b_z",
                                                        shape=[1, self.conv_out_units])

                    var['decoder_fc'] = layer

                var['decoder_deconv'] = list()
                with tf.variable_scope('deconv_stack'):

                    for l in range(self.layers_enc):
                        with tf.variable_scope('layer{}'.format(l)):
                            current = dict()

                            channels_in = self.param['conv_channels'][-1 - l]
                            if l == self.layers_enc - 1:
                                channels_out = 1
                            else:
                                channels_out = self.param['conv_channels'][-l - 2]

                            current['filter'] = create_variable("filter",
                                                                [3, 3, channels_out, channels_in])
                            #                             current['bias'] = create_bias_variable("bias",
                            #                                                 [channels_out])
                            var['decoder_deconv'].append(current)

                if self.rnn_decoder:
                    with tf.variable_scope('bilstm'):

                        # Make sure number of RNN units match deconvolutional decoder output shape
                        rnn_units = self.param['deconv_shape'][-1][1]
                        feature_units = self.param['deconv_shape'][-1][1]

                        with tf.variable_scope('fwd'):
                            cell_fwd = tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_units)
                            cell_fwd = tf.nn.rnn_cell.DropoutWrapper(cell_fwd,
                                                                     input_keep_prob=self.keep_prob)

                            var['cell_fwd'] = cell_fwd

                        with tf.variable_scope('bwd'):
                            cell_bwd = tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_units)
                            cell_bwd = tf.nn.rnn_cell.DropoutWrapper(cell_bwd,
                                                                     input_keep_prob=self.keep_prob)

                            var['cell_bwd'] = cell_bwd

                        with tf.variable_scope('fc'):

                            var['W_rnn'] = create_variable("W_rnn",
                                                           shape=[2*rnn_units, feature_units])
                            var['b_rnn'] = create_bias_variable("b_rnn",
                                                                shape=[1, feature_units])

                        if self.rnn_highway:
                            var['W_hw'] = create_variable("W_hw",
                                                          shape=[feature_units, feature_units])
                            var['b_hw'] = create_bias_variable("b_hw",
                                                               shape=[1, feature_units])



        return var

    def _create_network(self, input_batch, encode=False):

        # -----------------------------------
        # Encoder

        # Do encoder calculation
        encoder_hidden = input_batch
        print('Encoder hidden state 0: ', encoder_hidden)
        for l in range(self.layers_enc):
            # print(encoder_hidden)
            encoder_hidden = two_d_conv(encoder_hidden, self.variables['encoder_conv'][l]['filter'],
                                        self.param['max_pooling'][l])
            encoder_hidden = self.activation_conv(encoder_hidden)

            print(f'Encoder hidden state {l}: ', encoder_hidden)

        encoder_hidden = tf.reshape(encoder_hidden, [-1, self.conv_out_units])

        # Additional non-linearity between encoder hidden state and prediction of mu_0,sigma_0
        mu_logvar_hidden = tf.nn.dropout(self.activation(tf.matmul(encoder_hidden,
                                                                   self.variables['encoder_fc']['W_z0'])
                                                         + self.variables['encoder_fc']['b_z0']),
                                         keep_prob=self.keep_prob)

        # TODO: Properly take care of the case where some values might be -1 (not just here but throughout)
        if "cells_hidden_cat" in self.param.keys():
            mu_logvar_hidden_list = tf.split(mu_logvar_hidden, [self.cells_hidden] + self.param['cells_hidden_cat'], axis=1)
        else:
            mu_logvar_hidden_list = [mu_logvar_hidden]

        # Run classifiers
        y_logits_list, y_prob_list, y_logprob_list = self._class_predictor(predictor_input=mu_logvar_hidden_list,
                                                                           num_categories=self.num_categories)

        # TODO: Check if this makes sense; predicting mu and logvar using linear layer from same input
        encoder_mu = tf.add(tf.matmul(mu_logvar_hidden_list[0], self.variables['encoder_fc']['W_mu']),
                            self.variables['encoder_fc']['b_mu'], name='ZMu')
        encoder_logvar = tf.add(tf.matmul(mu_logvar_hidden_list[0], self.variables['encoder_fc']['W_logvar']),
                                self.variables['encoder_fc']['b_logvar'], name='ZLogVar')

        encoder_mu_list = [encoder_mu]
        encoder_logvar_list = [encoder_logvar]

        if 'dim_latent_cat' in self.param.keys():
            for k, n_dims in enumerate(self.param['dim_latent_cat']):
                encoder_mu_list.append(
                    tf.add(tf.matmul(mu_logvar_hidden_list[k + 1], self.variables['encoder_fc'][f'W_mu_{k}']),
                           self.variables['encoder_fc'][f'b_mu_{k}'], name=f'ZMu_{k}'))
                encoder_logvar_list.append(
                    tf.add(tf.matmul(mu_logvar_hidden_list[k + 1], self.variables['encoder_fc'][f'W_logvar_{k}']),
                           self.variables['encoder_fc'][f'b_logvar_{k}'], name=f'ZLogVar_{k}'))

        # Concatenate unconditioned as well as categorically conditioned parts of latent space into single latent vector
        encoder_mu = tf.concat(encoder_mu_list, axis=1)
        encoder_logvar = tf.concat(encoder_logvar_list, axis=1)

        # print(encoder_mu)

        # Convert log variance into standard deviation
        encoder_std = tf.exp(0.5 * encoder_logvar)

        # Sample epsilon
        epsilon = tf.random_normal(tf.shape(encoder_std), name='epsilon')

        if encode:
            z0 = tf.identity(encoder_mu, name='LatentZ0')
        else:
            z0 = tf.identity(tf.add(encoder_mu, tf.multiply(encoder_std, epsilon),
                                    name='LatentZ0'))

        # -----------------------------------
        # Latent flow

        # Lists to store the latent variables and the flow parameters
        nf_z = [z0]
        nf_sigma = [encoder_std]

        # Do calculations for each flow layer
        for l in range(self.param['iaf_flow_length']):

            W_flow = self.variables['iaf_flows'][l]['W_flow']
            b_flow = self.variables['iaf_flows'][l]['b_flow']

            nf_hidden = self.activation_nf(tf.matmul(encoder_hidden, W_flow) + b_flow)

            # Autoregressive calculation
            m_list = self.dim_latent * [None]
            s_list = self.dim_latent * [None]

            for j, flow_vars in enumerate(self.variables['iaf_flows'][l]['flow_vars']):

                # Go through computation one variable at a time
                if j == 0:
                    hidden_autoregressive = nf_hidden
                else:
                    z_slice = tf.slice(nf_z[-1], [0, 0], [-1, j])
                    hidden_autoregressive = tf.concat(axis=1, values=[nf_hidden, z_slice])

                W_flow_params_nl = flow_vars['W_flow_params_nl']
                b_flow_params_nl = flow_vars['b_flow_params_nl']
                W_flow_params = flow_vars['W_flow_params']
                b_flow_params = flow_vars['b_flow_params']

                # Non-linearity at current autoregressive step
                nf_hidden_nl = self.activation_nf(tf.matmul(hidden_autoregressive,
                                                       W_flow_params_nl) + b_flow_params_nl)

                # Calculate parameters for normalizing flow as linear transform
                ms = tf.matmul(nf_hidden_nl, W_flow_params) + b_flow_params

                # Split into individual components
                # m_list[j], s_list[j] = tf.split_v(value=ms,
                #                    size_splits=[1,1],
                #                    split_dim=1)
                m_list[j], s_list[j] = tf.split(value=ms,
                                                num_or_size_splits=[1, 1],
                                                axis=1)

            # Concatenate autoregressively computed variables
            # Add offset to s to make sure it starts out positive
            # (could have also initialised the bias term to 1)
            # Guarantees that flow initially small
            m = tf.concat(axis=1, values=m_list)
            s = self.param['initial_s_offset'] + tf.concat(axis=1, values=s_list)

            # Calculate sigma ("update gate value") from s
            sigma = tf.nn.sigmoid(s)
            nf_sigma.append(sigma)

            # Perform normalizing flow
            z_current = tf.multiply(sigma, nf_z[-1]) + tf.multiply((1 - sigma), m)

            # Invert order of variables to alternate dependence of autoregressive structure
            z_current = tf.reverse(z_current, axis=[1], name='LatentZ%d' % (l + 1))

            # Add to list of latent variables
            nf_z.append(z_current)

        z = tf.identity(nf_z[-1], name="LatentZ")

        # -----------------------------------
        # Decoder

        # Fully connected
        decoder_hidden = tf.nn.dropout(self.activation(tf.matmul(z, self.variables['decoder_fc']['W_z'])
                                                       + self.variables['decoder_fc']['b_z']),
                                       keep_prob=self.keep_prob)

        # print(decoder_hidden)

        # Reshape
        decoder_hidden = tf.reshape(decoder_hidden, [-1, self.conv_out_shape[0], self.conv_out_shape[1],
                                                     self.param['conv_channels'][-1]])

        for l in range(self.layers_enc):
            # print(decoder_hidden)

            pool_kernel = self.param['max_pooling'][-1 - l]
            decoder_hidden = two_d_deconv(decoder_hidden, self.variables['decoder_deconv'][l]['filter'],
                                          self.param['deconv_shape'][l], pool_kernel)
            if l < self.layers_enc - 1:
                decoder_hidden = self.activation_conv(decoder_hidden)

        # -------
        # If RNN is used, concatenate latent variables to CNN output here, send through bilstm
        if self.rnn_decoder:
            # Tile latent variables along time axis to concatenate to CNN output
            z_tiled = tf.tile(tf.expand_dims(z, axis=2), [1, 1, self.param['deconv_shape'][-1][2]])

            # Remove channel dimension from CNN output and concatenate
            decoder_hidden_no_channel = tf.squeeze(decoder_hidden, axis=3)
            rnn_input = tf.concat([decoder_hidden_no_channel, z_tiled], axis=1)

            # Transpose for RNN input
            rnn_input = tf.transpose(rnn_input, [0, 2, 1])

            # Do RNN calculation
            outputs_rnn_decoder, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.variables['cell_fwd'],
                cell_bw=self.variables['cell_bwd'],
                inputs=rnn_input,
                dtype=tf.float32)

            # Concatenate forward and backward results and apply fully connected layer
            decoder_hidden_rnn = tf.concat(outputs_rnn_decoder, 2)

            decoder_hidden_rnn = tf.reshape(decoder_hidden_rnn,
                                        shape=(-1, 2 * self.variables['cell_fwd'].output_size))
            decoder_hidden_rnn = tf.matmul(decoder_hidden_rnn, self.variables['W_rnn']) + self.variables['b_rnn']

            # Potential highway connection for CNN outputs
            if self.rnn_highway:
                cnn_out = tf.reshape(tf.transpose(decoder_hidden_no_channel, [0, 2, 1]), shape=(-1, self.param['deconv_shape'][-1][1]))
                hw_weights = tf.nn.sigmoid(tf.matmul(cnn_out, self.variables['W_hw']) + self.variables['b_hw'])
                decoder_hidden_rnn = hw_weights * cnn_out + (1 - hw_weights) * decoder_hidden_rnn

            decoder_hidden_rnn = tf.reshape(decoder_hidden_rnn,
                                            shape=(-1, self.param['deconv_shape'][-1][2], self.param['deconv_shape'][-1][1]))

            # Reshape and add channel dimension back in
            decoder_hidden = tf.expand_dims(tf.transpose(decoder_hidden_rnn, [0, 2, 1]), axis=3)

        # -------
        # Final output
        decoder_output = tf.nn.sigmoid(decoder_hidden)

        # print(decoder_output)

        # return decoder_output, encoder_hidden, encoder_logvar, encoder_std
        return decoder_output, encoder_mu, encoder_logvar, encoder_std, epsilon, z, nf_sigma, y_prob_list, y_logprob_list

    def _class_predictor(self,
                         predictor_input,
                         num_categories):

        y_logits_list = []
        y_prob_list = []
        y_logprob_list = []

        # Loop through each categorical variable
        for k in range(num_categories):

            if 'cells_hidden_cat' in self.param.keys() and self.param['cells_hidden_cat'][k] > 0:
                cat_input = predictor_input[k + 1]
            else:
                cat_input = predictor_input[0]

            y_hidden_category = tf.nn.dropout(cat_input, self.keep_prob)

            # Go through each layer
            for l in range(len(self.param['predictor_units'][k])):
                y_hidden_category = tf.nn.dropout(
                    self.activation(tf.matmul(y_hidden_category, self.variables['classifier'][k][l]['W'])
                                    + self.variables['classifier'][k][l]['b']),
                    keep_prob=self.keep_prob)

            # Final logits for class prediction
            logits_category = tf.matmul(y_hidden_category, self.variables['classifier'][k][-1]['W']) + self.variables['classifier'][k][-1]['b']
            y_logits_list.append(logits_category)

            # Class probabilities and logprobs
            y_prob_list.append(tf.nn.softmax(y_logits_list[k], name="ClassProbabilities_%d" % k))
            y_logprob_list.append(tf.log(y_prob_list[k] + 1e-10))

            # TODO: In seq2seq VAE did sampling here and actually passed on this knowledge to z predictor.
            #  Might want to revive this
            # # Sample y using Gumbel softmax or give a hard argmax during inference
            # gumbel_sample = tf.cond(y_argmax,
            #                         lambda: tf.one_hot(tf.argmax(y_prob_list[-1], axis=1),
            #                                            depth=param.num_classes[k]),
            #                         lambda: gumbel.sample_gumbel_softmax(y_logits_list[k], tau))
            #
            # # Combine known labels with sampled ones and specify shape, or pass softmax outputs directly
            # if param.get('pass_softmax') and param.pass_softmax:
            #     y_sample = y_prob_list[k]
            # else:
            #     y_sample = tf.multiply(vecs_known[k], y_one_hot_list[k]) + tf.multiply(vecs_unknown[k], gumbel_sample)
            #
            # y_sampled_list.append(tf.reshape(y_sample, shape=(-1, param.num_classes[k]), name=f"LatentY_{k}"))
            #
            # # Concatenate y to input to z predictor
            # encoder_hidden = tf.concat([encoder_hidden, y_sampled_list[k]], axis=1)

        return y_logits_list, y_prob_list, y_logprob_list


    def loss(self,
             input_batch,
             input_truth,
             batch_size_real=None,
             name='vae',
             beta=1.0,
             test=False):

        with tf.name_scope(name):

            # Unpack category data and convert to one hot
            print('Truth tensor:', input_truth)
            truth_list = tf.unstack(input_truth, axis=1)
            print('Truth list:', truth_list)
            onehot_list = []
            for k, item in enumerate(truth_list):
                onehot_list.append(tf.one_hot(item, depth=self.num_classes[k]))

            output, encoder_mu, encoder_logvar, encoder_std, epsilon, z, nf_sigma, y_prob_list, y_logprob_list = self._create_network(input_batch)

            print("Output size: ", output)

            # If a real batch size is given, cut off padding
            if batch_size_real is not None:
                input_batch = tf.slice(input_batch, [0, 0, 0, 0], [batch_size_real, -1, -1, -1])
                output = tf.slice(output, [0, 0, 0, 0], [batch_size_real, -1, -1, -1])
                epsilon = tf.slice(epsilon, [0, 0], [batch_size_real, -1])
                z = tf.slice(z, [0, 0], [batch_size_real, -1])
                nf_sigma = [tf.slice(x, [0, 0], [batch_size_real, -1]) for x in nf_sigma]
                y_prob_list = [tf.slice(x, [0, 0], [batch_size_real, -1]) for x in y_prob_list]
                y_logprob_list = [tf.slice(x, [0, 0], [batch_size_real, -1]) for x in y_logprob_list]

            _, div = kl_divergence(nf_sigma, epsilon, z, self.param, batch_mean=False)
            loss_latent = tf.identity(div, name='LossLatent')

            loss_reconstruction = tf.identity(-tf.reduce_sum(input_batch * tf.log(1e-8 + output)
                                                             + (1 - input_batch) * tf.log(1e-8 + 1 - output),
                                                             [1,2]), name='LossReconstruction')

            # loss_reconstruction = tf.reduce_mean(tf.pow(input_batch - output, 2))

            loss = tf.reduce_mean(loss_reconstruction + beta*loss_latent, name='Loss')
            # loss = tf.reduce_mean(loss_reconstruction, name='Loss')

            # # Dummy operation for now to use avoiding hanging of truth queue runner
            # loss += tf.to_float(0 * tf.reduce_mean(input_truth))

            # -----------------------------------------------------------------------
            # Compute (semi) supervised loss. If we have no categories this is 0
            y_prior_logprob = 0.0
            y_pred_loss = 0.0
            y_pred_accuracy = 0.0
            kl_y_reduced = 0.0

            accuracy_list = []

            print('One hot list:', onehot_list)
            print('Log prob list:', y_logprob_list)

            # Calculate loss and accuracy for each category
            for k in range(self.num_categories):

                # TODO: For semi-supervised, need to add KL loss here

                category_loss = -tf.reduce_sum(onehot_list[k] * y_logprob_list[k], axis=1)
                y_pred_loss += category_loss

                binary_prediction = tf.math.argmax(y_prob_list[k], dimension=1)
                category_accuracy = tf.reduce_sum(
                    tf.cast(tf.math.equal(binary_prediction, tf.math.argmax(onehot_list[k], dimension=1)), tf.float32)) / tf.cast(
                    tf.size(binary_prediction, out_type=tf.int32), tf.float32)

                y_pred_accuracy += category_accuracy
                accuracy_list.append(category_accuracy)

                if not test:
                    tf.summary.scalar(f'loss_prediction_{k}', tf.reduce_mean(category_loss))
                    tf.summary.scalar(f'accuracy_prediction_{k}', category_accuracy)

            y_pred_loss_mean = tf.reduce_mean(y_pred_loss, name="LossClassReconstruction")
            if self.num_categories > 0:
                y_pred_accuracy /= self.num_categories

            loss_total = loss + y_pred_loss_mean

            # Fake use of truth data if no categories to make queue runners happy
            if self.num_categories == 0:
                fake_truth_loss = tf.to_float(0.0) * tf.to_float(tf.reduce_mean(input_truth))
                loss_total += fake_truth_loss

            if not test:
                tf.summary.scalar('loss_total', loss_total)
                tf.summary.scalar('loss_vae', loss)
                tf.summary.scalar('loss_rec', tf.reduce_mean(loss_reconstruction))
                tf.summary.scalar('loss_kl', tf.reduce_mean(loss_latent))
                tf.summary.scalar('loss_prediction_mean', y_pred_loss_mean)
                tf.summary.scalar('beta', beta)

            return loss_total, accuracy_list

    def embed_and_predict(self,
                          input_batch,
                          batch_size_real=None):

        _, _, _, _, _, z, _, y_prob_list, _ = self._create_network(
            input_batch, encode=True)

        # If a real batch size is given, cut off padding
        if batch_size_real is not None:
            z = tf.slice(z, [0, 0], [batch_size_real, -1])
            y_prob_list = [tf.slice(x, [0, 0], [batch_size_real, -1]) for x in y_prob_list]

        return z, y_prob_list

    def encode_and_reconstruct(self, input_batch):

        output, _, _, _, _, encoder_mu, _, _, _ = self._create_network(input_batch, encode=True)

        return encoder_mu, output

    def decode(self, input_batch):

        z = input_batch

        # Fully connected
        decoder_hidden = self.activation(tf.matmul(z, self.variables['decoder_fc']['W_z'])
                                                       + self.variables['decoder_fc']['b_z'])

        # Reshape
        decoder_hidden = tf.reshape(decoder_hidden, [-1, self.conv_out_shape[0], self.conv_out_shape[1],
                                                     self.param['conv_channels'][-1]])

        for l in range(self.layers_enc):

            pool_kernel = self.param['max_pooling'][-1 - l]
            decoder_hidden = two_d_deconv(decoder_hidden, self.variables['decoder_deconv'][l]['filter'],
                                          self.param['deconv_shape'][l], pool_kernel)
            if l < self.layers_enc - 1:
                decoder_hidden = self.activation_conv(decoder_hidden)

        decoder_output = tf.nn.sigmoid(decoder_hidden)

        return decoder_output