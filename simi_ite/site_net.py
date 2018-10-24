import tensorflow as tf
import numpy as np

from util import *

class site_net(object):


    def __init__(self, x, t, y_ , p_t, FLAGS, r_lambda, do_in, do_out, dims, three_pairs, three_pairs_simi, r_mid_point_mini, r_pddm):
        self.variables = {}
        self.wd_loss = 0

        if FLAGS.nonlin.lower() == 'elu':
            self.nonlin = tf.nn.elu
        else:
            self.nonlin = tf.nn.relu

        self._build_graph(x, t, y_ , p_t, FLAGS, r_lambda, do_in, do_out, dims, three_pairs,  three_pairs_simi, r_mid_point_mini, r_pddm)

    def _add_variable(self, var, name):
        ''' Adds variables to the internal track-keeper '''
        basename = name
        i = 0
        while name in self.variables:
            name = '%s_%d' % (basename, i) #@TODO: not consistent with TF internally if changed
            i += 1

        self.variables[name] = var

    def _create_variable(self, var, name):
        ''' Create and adds variables to the internal track-keeper '''

        var = tf.get_variable(name=name, initializer=var)

        return var

    def _create_variable_with_weight_decay(self, initializer, name, wd):
        ''' Create and adds variables to the internal track-keeper
            and adds it to the list of weight decayed variables '''
        var = self._create_variable(initializer, name)
        self.wd_loss += wd*tf.nn.l2_loss(var)
        return var

    def _build_graph(self, x, t, y_ , p_t, FLAGS, r_lambda, do_in, do_out, dims, three_pairs,  three_pairs_simi, r_mid_point_mini, r_pddm):

        self.x = x
        self.t = t
        self.y_ = y_
        self.p_t = p_t
        self.r_lambda = r_lambda
        self.do_in = do_in
        self.do_out = do_out
        self.three_pairs = three_pairs
        self.three_pairs_simi = three_pairs_simi
        self.r_mid_point_mini = r_mid_point_mini
        self.r_pddm = r_pddm


        dim_input = dims[0]
        dim_in = dims[1]
        dim_out = dims[2]
        dim_pddm = dims[3]
        dim_c = dims[4]
        dim_s = dims[5]

        weights_in = []; biases_in = []


        if FLAGS.n_in == 0 or (FLAGS.n_in == 1 and FLAGS.varsel):
            dim_in = dim_input
        if FLAGS.n_out == 0:
            if FLAGS.split_output == False:
                dim_out = dim_in+1
            else:
                dim_out = dim_in

        if FLAGS.batch_norm:
            bn_biases = []
            bn_scales = []
            bn_biases_batch = []
            bn_scales_batch = []

        ''' Construct input/representation layers '''
        h_in = [x]
        h_in_batch = [three_pairs]
        for i in range(0, FLAGS.n_in):
            if i==0:
                ''' If using variable selection, first layer is just rescaling'''
                if FLAGS.varsel:
                    weights_in.append(tf.Variable(1.0/dim_input*tf.ones([dim_input])))
                else:
                    weights_in.append(tf.Variable(tf.random_normal([dim_input, dim_in], stddev=FLAGS.weight_init/np.sqrt(dim_input))))
            else:
                weights_in.append(tf.Variable(tf.random_normal([dim_in,dim_in], stddev=FLAGS.weight_init/np.sqrt(dim_in))))

            ''' If using variable selection, first layer is just rescaling'''
            if FLAGS.varsel and i==0:
                biases_in.append([])
                h_in.append(tf.mul(h_in[i],weights_in[i]))
                h_in_batch.append(tf.mul(h_in_batch[i],weights_in[i]))
            else:
                biases_in.append(tf.Variable(tf.zeros([1,dim_in])))
                z = tf.matmul(h_in[i], weights_in[i]) + biases_in[i]
                z_batch = tf.matmul(h_in_batch[i], weights_in[i]) + biases_in[i]

                if FLAGS.batch_norm:
                    batch_mean, batch_var = tf.nn.moments(z, [0])
                    batch_mean_batch, batch_var_batch = tf.nn.moments(z_batch, [0])
                    if FLAGS.normalization == 'bn_fixed':
                        z = tf.nn.batch_normalization(z, batch_mean, batch_var, 0, 1, 1e-3)
                        z_batch = tf.nn.batch_normalization(z_batch, batch_mean_batch, batch_var_batch, 0, 1, 1e-3)
                    else:
                        bn_biases.append(tf.Variable(tf.zeros([dim_in])))
                        bn_scales.append(tf.Variable(tf.ones([dim_in])))
                        bn_biases_batch.append(tf.Variable(tf.zeros([dim_in])))
                        bn_scales_batch.append(tf.Variable(tf.ones([dim_in])))
                        z = tf.nn.batch_normalization(z, batch_mean, batch_var, bn_biases[-1], bn_scales[-1], 1e-3)
                        z_batch = tf.nn.batch_normalization(z_batch, batch_mean_batch, batch_var_batch, bn_biases_batch[-1], bn_scales_batch[-1], 1e-3)

                h_in.append(self.nonlin(z))
                h_in_batch.append(self.nonlin(z_batch))
                h_in[i+1] = tf.nn.dropout(h_in[i+1], do_in)
                h_in_batch[i + 1] = tf.nn.dropout(h_in_batch[i + 1], do_in)

        h_rep = h_in[len(h_in)-1]
        h_rep_batch = h_in_batch[len(h_in_batch) - 1]


        if FLAGS.normalization == 'divide':
            h_rep_norm = h_rep / safe_sqrt(tf.reduce_sum(tf.square(h_rep), axis=1, keep_dims=True))
            h_rep_norm_batch = h_rep_batch / safe_sqrt(tf.reduce_sum(tf.square(h_rep_batch), axis=1, keep_dims=True))
        else:
            h_rep_norm = 1.0*h_rep
            h_rep_norm_batch = 1.0 * h_rep_batch

        ''' Construct ouput layers '''
        y, weights_out, weights_pred = self._build_output_graph(h_rep_norm, t, dim_in, dim_out, do_out, FLAGS)


        ''' Compute sample reweighting '''
        if FLAGS.reweight_sample:
            w_t = t/(2*p_t)
            w_c = (1-t)/(2*1-p_t)
            sample_weight = w_t + w_c
        else:
            sample_weight = 1.0

        self.sample_weight = sample_weight

        ''' Construct factual loss function '''
        if FLAGS.loss == 'l1':
            risk = tf.reduce_mean(sample_weight*tf.abs(y_-y))
            pred_error = -tf.reduce_mean(res)

        elif FLAGS.loss == 'log':
            y = 0.995/(1.0+tf.exp(-y)) + 0.0025
            res = y_*tf.log(y) + (1.0-y_)*tf.log(1.0-y)
            risk = -tf.reduce_mean(sample_weight*res)
            pred_error = -tf.reduce_mean(res)
        else:
            risk = tf.reduce_mean(sample_weight*tf.square(y_ - y))
            pred_error = tf.sqrt(tf.reduce_mean(tf.square(y_ - y)))

        ''' Regularization '''
        if FLAGS.p_lambda>0 and FLAGS.rep_weight_decay:
            for i in range(0, FLAGS.n_in):
                if not (FLAGS.varsel and i==0): # No penalty on W in variable selection
                    self.wd_loss += tf.nn.l2_loss(weights_in[i])



        '''PDDM unit'''
        # get x_i, x_j,  x_k,  x_l,  x_m,  x_n
        x_i = tf.slice(h_rep_norm_batch, [0, 0], [1, dim_in])
        x_j = tf.slice(h_rep_norm_batch, [1, 0], [1, dim_in])
        x_k = tf.slice(h_rep_norm_batch, [2, 0], [1, dim_in])
        x_l = tf.slice(h_rep_norm_batch, [3, 0], [1, dim_in])
        x_m = tf.slice(h_rep_norm_batch, [4, 0], [1, dim_in])
        x_n = tf.slice(h_rep_norm_batch, [5, 0], [1, dim_in])
        with tf.variable_scope('pddm') as scope:
            s_kl, weights_pddm, biases_pddm = self.pddm(x_k, x_l, dim_in, dim_pddm, dim_c,dim_s, FLAGS)
            scope.reuse_variables()
            s_mn, weights_pddm, biases_pddm = self.pddm(x_m, x_n, dim_in, dim_pddm, dim_c, dim_s, FLAGS)
            s_km, weights_pddm, biases_pddm = self.pddm(x_k, x_m, dim_in, dim_pddm, dim_c, dim_s, FLAGS)
            s_ik, weights_pddm, biases_pddm = self.pddm(x_i, x_k, dim_in, dim_pddm, dim_c, dim_s, FLAGS)
            s_jm, weights_pddm, biases_pddm = self.pddm(x_j, x_m, dim_in, dim_pddm, dim_c, dim_s, FLAGS)

        '''pddm loss'''
        simi_kl = tf.slice(three_pairs_simi, [0, 0], [1, 1])
        simi_mn = tf.slice(three_pairs_simi, [1, 0], [1, 1])
        simi_km = tf.slice(three_pairs_simi, [2, 0], [1, 1])
        simi_ik = tf.slice(three_pairs_simi, [3, 0], [1, 1])
        simi_jm = tf.slice(three_pairs_simi, [4, 0], [1, 1])
        pddm_loss= tf.reduce_sum(tf.square(simi_kl - s_kl) + tf.square(simi_mn - s_mn)
                                 + tf.square(simi_km - s_km) + tf.square(simi_ik - s_ik)
                                 + tf.square(simi_jm - s_jm))


        '''mid_point distance minimization'''
        mid_jk = (x_j + x_k) / 2.0
        mid_im = (x_i + x_m) / 2.0
        mid_distance = tf.reduce_sum(tf.square(mid_jk - mid_im), [0, 1])

        ''' Total error '''
        tot_error = risk

        if FLAGS.p_lambda>0:
            tot_error = tot_error + r_lambda*self.wd_loss

        if FLAGS.p_mid_point_mini > 0:
            tot_error = tot_error + r_mid_point_mini * mid_distance

        if FLAGS.p_pddm > 0:
            tot_error = tot_error + r_pddm * pddm_loss

        if FLAGS.varsel:
            self.w_proj = tf.placeholder("float", shape=[dim_input], name='w_proj')
            self.projection = weights_in[0].assign(self.w_proj)

        self.output = y
        self.tot_loss = tot_error

        self.pred_loss = pred_error
        self.weights_in = weights_in
        self.weights_out = weights_out
        self.weights_pred = weights_pred
        self.h_rep = h_rep
        self.h_rep_norm = h_rep_norm
        self.weights_pddm = weights_pddm
        self.biases_pddm = biases_pddm
        self.mid_distance = mid_distance
        self.pddm_loss = pddm_loss

    def _build_output(self, h_input, dim_in, dim_out, do_out, FLAGS):
        h_out = [h_input]
        dims = [dim_in] + ([dim_out]*FLAGS.n_out)

        weights_out = []; biases_out = []

        for i in range(0, FLAGS.n_out):
            wo = self._create_variable_with_weight_decay(
                    tf.random_normal([dims[i], dims[i+1]],
                        stddev=FLAGS.weight_init/np.sqrt(dims[i])),
                    'w_out_%d' % i, 1.0)
            weights_out.append(wo)

            biases_out.append(tf.Variable(tf.zeros([1,dim_out])))
            z = tf.matmul(h_out[i], weights_out[i]) + biases_out[i]

            h_out.append(self.nonlin(z))
            h_out[i+1] = tf.nn.dropout(h_out[i+1], do_out)

        weights_pred = self._create_variable(tf.random_normal([dim_out,1],
            stddev=FLAGS.weight_init/np.sqrt(dim_out)), 'w_pred')
        bias_pred = self._create_variable(tf.zeros([1]), 'b_pred')

        if FLAGS.varsel or FLAGS.n_out == 0:
            self.wd_loss += tf.nn.l2_loss(tf.slice(weights_pred,[0,0],[dim_out-1,1])) #don't penalize treatment coefficient
        else:
            self.wd_loss += tf.nn.l2_loss(weights_pred)

        ''' Construct linear classifier '''
        h_pred = h_out[-1]
        y = tf.matmul(h_pred, weights_pred)+bias_pred

        return y, weights_out, weights_pred

    def pddm(self, x_i, x_j, dim_in, dim_pddm, dim_c,dim_s, FLAGS):

        weights_pddm = []
        bias_pddm = []

        u = tf.abs(x_i-x_j)
        v = (x_i+x_j)/2.0
        b_u = []
        b_v = []
        b_c = []
        b_s = []
        b_u.append(tf.Variable(tf.zeros([1, dim_pddm])))
        b_v.append(tf.Variable(tf.zeros([1, dim_pddm])))
        b_c.append(tf.Variable(tf.zeros([1, dim_c])))
        b_s.append(tf.Variable(tf.zeros([1, dim_s])))
        w_u = self._create_variable_with_weight_decay(
            tf.random_normal([dim_in, dim_pddm],
                             stddev=FLAGS.weight_init / np.sqrt(dim_in)),
            'w_u', 1.0)
        weights_pddm.append(w_u)
        w_v = self._create_variable_with_weight_decay(
            tf.random_normal([dim_in, dim_pddm],
                             stddev=FLAGS.weight_init / np.sqrt(dim_in)),
            'w_v', 1.0)
        weights_pddm.append(w_v)

        concate_dim = dim_pddm+dim_pddm
        w_c = self._create_variable_with_weight_decay(
            tf.random_normal([concate_dim, dim_c],
                             stddev=FLAGS.weight_init / np.sqrt(concate_dim)),
            'w_c', 1.0)
        weights_pddm.append(w_c)

        w_s = self._create_variable_with_weight_decay(
            tf.random_normal([dim_c, dim_s],
                             stddev=FLAGS.weight_init / np.sqrt(dim_c)),
            'w_s', 1.0)
        weights_pddm.append(w_s)

        u_1 = tf.matmul(u, w_u) + b_u
        v_1 = tf.matmul(v, w_v) + b_v
        u_1 = self.nonlin(u_1)
        v_1 = self.nonlin(v_1)
        u_1 = tf.nn.l2_normalize(u_1, dim = 0)
        v_1 = tf.nn.l2_normalize(v_1, dim = 0)
        c = tf.concat([u_1, v_1], 1)
        c = tf.reshape(c, [1,concate_dim])
        c = tf.matmul(c, w_c) + b_c
        c = self.nonlin(c)
        c = tf.reshape(c, [1, dim_c])
        s = tf.matmul(c, w_s) + b_s

        bias_pddm.append(b_u)
        bias_pddm.append(b_v)
        bias_pddm.append(b_c)
        bias_pddm.append(b_s)

        return s, weights_pddm, bias_pddm

    def _build_output_graph(self, rep, t, dim_in, dim_out, do_out, FLAGS):
        ''' Construct output/regression layers '''

        if FLAGS.split_output:

            i0 = tf.to_int32(tf.where(t < 1)[:,0])
            i1 = tf.to_int32(tf.where(t > 0)[:,0])

            rep0 = tf.gather(rep, i0)
            rep1 = tf.gather(rep, i1)
            with tf.variable_scope("control") as scope:

                y0, weights_out0, weights_pred0 = self._build_output(rep0, dim_in, dim_out, do_out, FLAGS)

            with tf.variable_scope("treated") as scope:
                y1, weights_out1, weights_pred1 = self._build_output(rep1, dim_in, dim_out, do_out, FLAGS)

            y = tf.dynamic_stitch([i0, i1], [y0, y1])
            weights_out = weights_out0 + weights_out1
            weights_pred = weights_pred0 + weights_pred1
        else:
            h_input = tf.concat([rep, t],1)
            y, weights_out, weights_pred = self._build_output(h_input, dim_in+1, dim_out, do_out, FLAGS)

        return y, weights_out, weights_pred
