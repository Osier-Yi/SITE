import tensorflow as tf
import numpy as np
import sys, os
import getopt
import random
import datetime
import traceback
from subprocess import call

import simi_ite.site_net as site
from simi_ite.util import *

''' Define parameter flags '''
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('loss', 'l2', """Which loss function to use (l1/l2/log)""")
tf.app.flags.DEFINE_integer('n_in', 2, """Number of representation layers. """)
tf.app.flags.DEFINE_integer('n_out', 2, """Number of regression layers. """)
tf.app.flags.DEFINE_float('p_lambda', 0.0, """Weight decay regularization parameter. """)
tf.app.flags.DEFINE_integer('rep_weight_decay', 1, """Whether to penalize representation layers with weight decay""")
tf.app.flags.DEFINE_float('dropout_in', 0.9, """Input layers dropout keep rate. """)
tf.app.flags.DEFINE_float('dropout_out', 0.9, """Output layers dropout keep rate. """)
tf.app.flags.DEFINE_string('nonlin', 'relu', """Kind of non-linearity. Default relu. """)
tf.app.flags.DEFINE_float('lrate', 0.05, """Learning rate. """)
tf.app.flags.DEFINE_float('decay', 0.5, """RMSProp decay. """)
tf.app.flags.DEFINE_integer('batch_size', 100, """Batch size. """)
tf.app.flags.DEFINE_integer('dim_in', 100, """Pre-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('dim_out', 100, """Post-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('batch_norm', 0, """Whether to use batch normalization. """)
tf.app.flags.DEFINE_string('normalization', 'none', """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
tf.app.flags.DEFINE_integer('experiments', 1, """Number of experiments. """)
tf.app.flags.DEFINE_integer('iterations', 2000, """Number of iterations. """)
tf.app.flags.DEFINE_float('weight_init', 0.01, """Weight initialization scale. """)
tf.app.flags.DEFINE_float('lrate_decay', 0.95, """Decay of learning rate every 100 iterations """)
tf.app.flags.DEFINE_integer('varsel', 0, """Whether the first layer performs variable selection. """)
tf.app.flags.DEFINE_string('outdir', '../results/ihdp/', """Output directory. """)
tf.app.flags.DEFINE_string('datadir', '../data/', """Data directory. """)
tf.app.flags.DEFINE_string('dataform', 'ihdp_npci_1-100.train.npz', """Training data filename form. """)
tf.app.flags.DEFINE_string('data_test', '', """Test data filename form. """)
tf.app.flags.DEFINE_integer('sparse', 0, """Whether data is stored in sparse format (.x, .y). """)
tf.app.flags.DEFINE_integer('seed', 1, """Seed. """)
tf.app.flags.DEFINE_integer('repetitions', 1, """Repetitions with different seed.""")
tf.app.flags.DEFINE_integer('use_p_correction', 1, """Whether to use population size p(t) in mmd/disc/wass.""")
tf.app.flags.DEFINE_string('optimizer', 'RMSProp', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
tf.app.flags.DEFINE_string('imb_fun', 'mmd_lin', """Which imbalance penalty to use (mmd_lin/mmd_rbf/mmd2_lin/mmd2_rbf/lindisc/wass). """)
tf.app.flags.DEFINE_integer('output_csv',0,"""Whether to save a CSV file with the results_try1""")
tf.app.flags.DEFINE_integer('output_delay', 100, """Number of iterations between log/loss outputs. """)
tf.app.flags.DEFINE_integer('pred_output_delay', -1, """Number of iterations between prediction outputs. (-1 gives no intermediate output). """)
tf.app.flags.DEFINE_integer('debug', 0, """Debug mode. """)
tf.app.flags.DEFINE_integer('save_rep', 0, """Save representations after training. """)
tf.app.flags.DEFINE_float('val_part', 0, """Validation part. """)
tf.app.flags.DEFINE_boolean('split_output', 0, """Whether to split output layers= between treated and control. """)
tf.app.flags.DEFINE_boolean('reweight_sample', 1, """Whether to reweight sample for prediction loss with average treatment probability. """)
tf.app.flags.DEFINE_float('p_pddm', 1.0, """PDDM unit parameter """)
tf.app.flags.DEFINE_float('p_mid_point_mini', 1.0, """Mid point distance minimization parameter """)
tf.app.flags.DEFINE_float('dim_pddm', 100.0, """Dimension in PDDM fist layer """)
tf.app.flags.DEFINE_float('dim_c', 100.0, """Dimension in PDDM unit for c """)
tf.app.flags.DEFINE_float('dim_s', 100.0, """Dimension in PDDM unit for s """)
tf.app.flags.DEFINE_string('propensity_dir','./propensity_score/ihdp_propensity_model.sav', """Dir where the propensity model is saved""" )
tf.app.flags.DEFINE_boolean('equal_sample', 0, """Whether to fectch equal number of samples with different labels. """)

if FLAGS.sparse:
    import scipy.sparse as sparse

NUM_ITERATIONS_PER_DECAY = 100

__DEBUG__ = False
if FLAGS.debug:
    __DEBUG__ = True


def three_pair_extration(x, t, yf, propensity_dir):
    '''
    :param x: pre-treatment covariates
    :param t: treatment
    :param yf: factual outcome
    :param propensity_dir: the directory that saves propensity model
    :return: the selected three pairs' pre-treatment covariates, index, treatment,
    factual_outcome, and the similarity score ((x_k, x_l), (x_m, x_n), (x_k, x_m), (x_i, x_k), (x_j, x_m))
    '''
    three_pairs, I_three_pairs = find_three_pairs(x, t, propensity_dir)
    t_three_pairs = t[I_three_pairs]
    y_three_pairs = yf[I_three_pairs]
    three_pairs_simi = get_three_pair_simi(three_pairs, propensity_dir)
    return three_pairs, I_three_pairs, t_three_pairs, y_three_pairs, three_pairs_simi


def train(SITE, sess, train_step, D, I_valid, D_test, logfile, i_exp):
    """ Trains the model on supplied data """

    ''' Train/validation split '''
    n = D['x'].shape[0]
    I = range(n); I_train = list(set(I)-set(I_valid))
    n_train = len(I_train)

    ''' Compute treatment probability'''
    p_treated = np.mean(D['t'][I_train,:])

    ''' Set up loss feed_dicts'''
    three_pairs_train, _, _, _, three_pairs_simi_train = three_pair_extration(
        D['x'][I_train, :], D['t'][I_train, :], D['yf'][I_train,:], FLAGS.propensity_dir)

    dict_factual = {SITE.x: D['x'][I_train,:], SITE.t: D['t'][I_train,:], SITE.y_: D['yf'][I_train,:],
                    SITE.do_in: 1.0, SITE.do_out: 1.0, SITE.r_lambda: FLAGS.p_lambda, SITE.p_t: p_treated,
                    SITE.three_pairs: three_pairs_train,  SITE.three_pairs_simi:three_pairs_simi_train,
                    SITE.r_mid_point_mini:FLAGS.p_mid_point_mini, SITE.r_pddm: FLAGS.p_pddm }

    if FLAGS.val_part > 0:
        three_pairs_valid, _, _, _, three_pairs_simi_valid = three_pair_extration(
            D['x'][I_valid, :],D['t'][I_valid, :], D['yf'][I_valid, :], FLAGS.propensity_dir)

        dict_valid = {SITE.x: D['x'][I_valid,:], SITE.t: D['t'][I_valid,:], SITE.y_: D['yf'][I_valid,:],
                      SITE.do_in: 1.0, SITE.do_out: 1.0, SITE.r_lambda: FLAGS.p_lambda, SITE.p_t: p_treated,
                      SITE.three_pairs: three_pairs_valid, SITE.three_pairs_simi:three_pairs_simi_valid,
                      SITE.r_mid_point_mini:FLAGS.p_mid_point_mini,SITE.r_pddm: FLAGS.p_pddm }

    if D['HAVE_TRUTH']:
        dict_cfactual = {SITE.x: D['x'][I_train,:], SITE.t: 1-D['t'][I_train,:], SITE.y_: D['ycf'][I_train,:],
                         SITE.do_in: 1.0, SITE.do_out: 1.0, SITE.three_pairs: three_pairs_train,
                         SITE.three_pairs_simi:three_pairs_simi_train,
                         SITE.r_mid_point_mini:FLAGS.p_mid_point_mini,SITE.r_pddm: FLAGS.p_pddm }

    ''' Initialize TensorFlow variables '''
    sess.run(tf.global_variables_initializer())

    ''' Set up for storing predictions '''
    preds_train = []
    preds_test = []

    ''' Compute losses '''
    losses = []
    obj_loss, f_error,  pddm_loss_batch, mid_dist_batch = sess.run([SITE.tot_loss, SITE.pred_loss, SITE.pddm_loss, SITE.mid_distance ],\
      feed_dict=dict_factual)

    cf_error = np.nan
    if D['HAVE_TRUTH']:
        cf_error = sess.run(SITE.pred_loss, feed_dict=dict_cfactual)

    valid_obj = np.nan; valid_imb = np.nan; valid_f_error = np.nan;
    if FLAGS.val_part > 0:
        valid_obj, valid_f_error = sess.run([SITE.tot_loss, SITE.pred_loss],\
          feed_dict=dict_valid)

    losses.append([obj_loss, f_error, cf_error,  valid_f_error,  valid_obj])

    objnan = False

    reps = []
    reps_test = []

    ''' Train for multiple iterations '''
    for i in range(FLAGS.iterations):

        ''' Fetch sample '''
        t_index = 0
        while t_index < 0.05 or t_index > 0.95:
            I = random.sample(range(0, n_train), FLAGS.batch_size)
            x_batch = D['x'][I_train, :][I, :]
            t_batch = D['t'][I_train, :][I]
            y_batch = D['yf'][I_train, :][I]
            t_index = np.mean(t_batch)


        if __DEBUG__:
            M = sess.run(site.pop_dist(SITE.x, SITE.t), feed_dict={SITE.x: x_batch, SITE.t: t_batch})
            log(logfile, 'Median: %.4g, Mean: %.4f, Max: %.4f' % (np.median(M.tolist()), np.mean(M.tolist()), np.amax(M.tolist())))

        ''' Do one step of gradient descent '''

        if not objnan:

            three_pairs_batch, _, _, _, three_pairs_simi = three_pair_extration(
                x_batch, t_batch, y_batch, FLAGS.propensity_dir)


            sess.run(train_step, feed_dict={SITE.x: x_batch, SITE.t: t_batch, SITE.y_: y_batch,
                                            SITE.do_in: FLAGS.dropout_in, SITE.do_out: FLAGS.dropout_out,
                                            SITE.r_lambda: FLAGS.p_lambda, SITE.p_t: p_treated, SITE.three_pairs: three_pairs_batch,
                                            SITE.three_pairs_simi:three_pairs_simi, SITE.r_mid_point_mini:FLAGS.p_mid_point_mini,
                                            SITE.r_pddm: FLAGS.p_pddm})

        ''' Project variable selection weights '''
        if FLAGS.varsel:
            wip = simplex_project(sess.run(SITE.weights_in[0]), 1)
            sess.run(SITE.projection, feed_dict={SITE.w_proj: wip})

        ''' Compute loss every N iterations '''
        if i % FLAGS.output_delay == 0 or i==FLAGS.iterations-1:
            obj_loss, f_error, pddm_loss_batch, mid_dist_batch = sess.run([SITE.tot_loss, SITE.pred_loss, SITE.pddm_loss, SITE.mid_distance ],
                feed_dict=dict_factual)

            rep = sess.run(SITE.h_rep_norm, feed_dict={SITE.x: D['x'], SITE.do_in: 1.0})
            rep_norm = np.mean(np.sqrt(np.sum(np.square(rep), 1)))
            # print rep

            cf_error = np.nan
            if D['HAVE_TRUTH']:
                cf_error = sess.run(SITE.pred_loss, feed_dict=dict_cfactual)

            valid_obj = np.nan; valid_imb = np.nan; valid_f_error = np.nan;
            if FLAGS.val_part > 0:
                valid_obj, valid_f_error = sess.run([SITE.tot_loss, SITE.pred_loss], feed_dict=dict_valid)

            losses.append([obj_loss, f_error, cf_error, valid_f_error, valid_obj])

            loss_str = str(i) + '\tObj: %.3f,\tF: %.3f,\tCf: %.3f,\tPDDM: %.2g,\tmid_p: %.2g,\tVal: %.3f,\tValObj: %.2f' \
                        % (obj_loss, f_error, cf_error, pddm_loss_batch, mid_dist_batch, valid_f_error, valid_obj)

            if FLAGS.loss == 'log':
                y_pred = sess.run(SITE.output, feed_dict={SITE.x: x_batch, \
                    SITE.t: t_batch, SITE.do_in: 1.0, SITE.do_out: 1.0})
                y_pred = 1.0*(y_pred > 0.5)
                acc = 100*(1 - np.mean(np.abs(y_batch - y_pred)))
                loss_str += ',\tAcc: %.2f%%' % acc

            log(logfile, loss_str)

            if np.isnan(obj_loss):
                log(logfile, 'Experiment %d: Objective is NaN. Skipping.' % i_exp)
                objnan = True

        ''' Compute predictions every M iterations '''
        if (FLAGS.pred_output_delay > 0 and i % FLAGS.pred_output_delay == 0) or i==FLAGS.iterations-1:

            y_pred_f = sess.run(SITE.output, feed_dict={SITE.x: D['x'], \
                SITE.t: D['t'], SITE.do_in: 1.0, SITE.do_out: 1.0})
            y_pred_cf = sess.run(SITE.output, feed_dict={SITE.x: D['x'], \
                SITE.t: 1-D['t'], SITE.do_in: 1.0, SITE.do_out: 1.0})
            preds_train.append(np.concatenate((y_pred_f, y_pred_cf),axis=1))

            if D_test is not None:
                y_pred_f_test = sess.run(SITE.output, feed_dict={SITE.x: D_test['x'], \
                    SITE.t: D_test['t'], SITE.do_in: 1.0, SITE.do_out: 1.0})
                y_pred_cf_test = sess.run(SITE.output, feed_dict={SITE.x: D_test['x'], \
                    SITE.t: 1-D_test['t'], SITE.do_in: 1.0, SITE.do_out: 1.0})
                preds_test.append(np.concatenate((y_pred_f_test, y_pred_cf_test),axis=1))

            if FLAGS.save_rep and i_exp == 1:
                reps_i = sess.run([SITE.h_rep], feed_dict={SITE.x: D['x'], \
                    SITE.do_in: 1.0, SITE.do_out: 0.0})
                reps.append(reps_i)

                if D_test is not None:
                    reps_test_i = sess.run([SITE.h_rep], feed_dict={SITE.x: D_test['x'], \
                        SITE.do_in: 1.0, SITE.do_out: 0.0})
                    reps_test.append(reps_test_i)

    return losses, preds_train, preds_test, reps, reps_test

def run(outdir):
    """ Runs an experiment and stores result in outdir """

    ''' Set up paths and start log '''
    npzfile = outdir+'result'
    npzfile_test = outdir+'result.test'
    repfile = outdir+'reps'
    repfile_test = outdir+'reps.test'
    outform = outdir+'y_pred'
    outform_test = outdir+'y_pred.test'
    lossform = outdir+'loss'
    logfile = outdir+'log.txt'
    f = open(logfile,'w')
    f.close()
    dataform = FLAGS.datadir + FLAGS.dataform

    has_test = False
    if not FLAGS.data_test == '': # if test set supplied
        has_test = True
        dataform_test = FLAGS.datadir + FLAGS.data_test

    ''' Set random seeds '''
    random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    ''' Save parameters '''
    save_config(outdir+'config.txt')

    log(logfile, 'Training with hyperparameters: p_pddm=%.2g, r_mid_point_mini=%.2g' % (FLAGS.p_pddm,FLAGS.p_mid_point_mini))

    ''' Load Data '''
    npz_input = False
    if dataform[-3:] == 'npz':
        npz_input = True
    if npz_input:
        datapath = dataform
        if has_test:
            datapath_test = dataform_test
    else:
        datapath = dataform % 1
        if has_test:
            datapath_test = dataform_test % 1

    log(logfile,     'Training data: ' + datapath)
    if has_test:
        log(logfile, 'Test data:     ' + datapath_test)
    D = load_data(datapath)
    D_test = None
    if has_test:
        D_test = load_data(datapath_test)

    log(logfile, 'Loaded data with shape [%d,%d]' % (D['n'], D['dim']))

    ''' Start Session '''
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True

    sess = tf.Session(config=config)

    ''' Initialize input placeholders '''
    x  = tf.placeholder("float", shape=[None, D['dim']], name='x') # Features
    t  = tf.placeholder("float", shape=[None, 1], name='t')   # Treatent
    y_ = tf.placeholder("float", shape=[None, 1], name='y_')  # Outcome

    # the pre-treatment covariates of selected three pairs
    three_pairs = tf.placeholder("float", shape=[6, D['dim']], name='three_pairs')

    # the similarity score of selected three pairs (ground truth similarity)
    three_pairs_simi = tf.placeholder("float", shape = [5,1], name = 'three_pairs_simi')




    ''' Parameter placeholders '''
    r_lambda = tf.placeholder("float", name='r_lambda')
    do_in = tf.placeholder("float", name='dropout_in')
    do_out = tf.placeholder("float", name='dropout_out')
    p = tf.placeholder("float", name='p_treated')
    n_train_graph = tf.placeholder("float", name='n_train_graph')

    # the hyperparameter for MPDM loss
    r_mid_point_mini = tf.placeholder("float", name='r_mid_point_mini')

    # the hyperparameter for PDDM loss
    r_pddm = tf.placeholder("float", name='r_pddm')

    ''' Define model graph '''
    log(logfile, 'Defining graph...\n')
    dims = [D['dim'], FLAGS.dim_in, FLAGS.dim_out, int(FLAGS.dim_pddm), int(FLAGS.dim_c), int(FLAGS.dim_s)]
    SITE = site.site_net(x, t, y_, p, FLAGS, r_lambda, do_in, do_out, dims, three_pairs, three_pairs_simi, r_mid_point_mini, r_pddm)

    ''' Set up optimizer '''
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(FLAGS.lrate, global_step, \
        NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)

    opt = None
    if FLAGS.optimizer == 'Adagrad':
        opt = tf.train.AdagradOptimizer(lr)
    elif FLAGS.optimizer == 'GradientDescent':
        opt = tf.train.GradientDescentOptimizer(lr)
    elif FLAGS.optimizer == 'Adam':
        opt = tf.train.AdamOptimizer(lr)
    else:
        opt = tf.train.RMSPropOptimizer(lr, FLAGS.decay)

    ''' Unused gradient clipping '''
    #gvs = opt.compute_gradients(SITE.tot_loss)
    #capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
    #train_step = opt.apply_gradients(capped_gvs, global_step=global_step)

    train_step = opt.minimize(SITE.tot_loss,global_step=global_step)

    ''' Set up for saving variables '''
    all_losses = []
    all_preds_train = []
    all_preds_test = []
    all_valid = []
    all_train_rep = []
    all_test_rep = []
    if FLAGS.varsel:
        all_weights = None
        all_beta = None

    all_preds_test = []

    ''' Handle repetitions '''
    n_experiments = FLAGS.experiments
    if FLAGS.repetitions>1:
        if FLAGS.experiments>1:
            log(logfile, 'ERROR: Use of both repetitions and multiple experiments is currently not supported.')
            sys.exit(1)
        n_experiments = FLAGS.repetitions

    ''' Run for all repeated experiments '''
    for i_exp in range(1,n_experiments+1):

        if FLAGS.repetitions>1:
            log(logfile, 'Training on repeated initialization %d/%d...' % (i_exp, FLAGS.repetitions))
        else:
            log(logfile, 'Training on experiment %d/%d...' % (i_exp, n_experiments))

        ''' Load Data (if multiple repetitions, reuse first set)'''

        if i_exp==1 or FLAGS.experiments>1:
            D_exp_test = None
            if npz_input:
                D_exp = {}
                D_exp['x']  = D['x'][:,:,i_exp-1]
                D_exp['t']  = D['t'][:,i_exp-1:i_exp]
                D_exp['yf'] = D['yf'][:,i_exp-1:i_exp]
                if D['HAVE_TRUTH']:
                    D_exp['ycf'] = D['ycf'][:,i_exp-1:i_exp]
                else:
                    D_exp['ycf'] = None

                if has_test:
                    D_exp_test = {}
                    D_exp_test['x']  = D_test['x'][:,:,i_exp-1]
                    D_exp_test['t']  = D_test['t'][:,i_exp-1:i_exp]
                    D_exp_test['yf'] = D_test['yf'][:,i_exp-1:i_exp]
                    if D_test['HAVE_TRUTH']:
                        D_exp_test['ycf'] = D_test['ycf'][:,i_exp-1:i_exp]
                    else:
                        D_exp_test['ycf'] = None
            else:
                datapath = dataform % i_exp
                D_exp = load_data(datapath)
                if has_test:
                    datapath_test = dataform_test % i_exp
                    D_exp_test = load_data(datapath_test)

            D_exp['HAVE_TRUTH'] = D['HAVE_TRUTH']
            if has_test:
                D_exp_test['HAVE_TRUTH'] = D_test['HAVE_TRUTH']

        ''' Split into training and validation sets '''
        if FLAGS.equal_sample > 0:
            index_y_c_0 = np.intersect1d(np.where(D_exp['t'] < 1), np.where(D_exp['yf'] < 1))
            index_y_c_1 = np.intersect1d(np.where(D_exp['t'] < 1), np.where(D_exp['yf'] > 0))
            index_y_t_0 = np.intersect1d(np.where(D_exp['t'] > 0), np.where(D_exp['yf'] < 1))
            index_y_t_1 = np.intersect1d(np.where(D_exp['t'] > 0), np.where(D_exp['yf'] > 0))

            I_train_c_0, I_valid_c_0 = validation_split_equal(index_y_c_0, FLAGS.val_part)
            I_train_c_1, I_valid_c_1 = validation_split_equal(index_y_c_1, FLAGS.val_part)
            I_train_t_0, I_valid_t_0 = validation_split_equal(index_y_t_0, FLAGS.val_part)
            I_train_t_1, I_valid_t_1 = validation_split_equal(index_y_t_1, FLAGS.val_part)
            I_valid = index_y_c_0[I_valid_c_0].tolist() + index_y_c_1[I_valid_c_1].tolist() + \
                      index_y_t_0[I_valid_t_0].tolist() + index_y_t_1[I_valid_t_1].tolist()
        else:

            I_train, I_valid = validation_split(D_exp, FLAGS.val_part)

        ''' Run training loop '''
        losses, preds_train, preds_test, reps, reps_test = \
            train(SITE, sess, train_step, D_exp, I_valid, \
                D_exp_test, logfile, i_exp)

        ''' Collect all reps '''
        all_preds_train.append(preds_train)
        all_preds_test.append(preds_test)
        all_losses.append(losses)

        ''' Fix shape for output (n_units, dim, n_reps, n_outputs) '''
        out_preds_train = np.swapaxes(np.swapaxes(all_preds_train,1,3),0,2)
        if  has_test:
            out_preds_test = np.swapaxes(np.swapaxes(all_preds_test,1,3),0,2)
        out_losses = np.swapaxes(np.swapaxes(all_losses,0,2),0,1)

        ''' Store predictions '''
        log(logfile, 'Saving result to %s...\n' % outdir)
        if FLAGS.output_csv:
            np.savetxt('%s_%d.csv' % (outform,i_exp), preds_train[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (outform_test,i_exp), preds_test[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (lossform,i_exp), losses, delimiter=',')

        ''' Compute weights if doing variable selection '''
        if FLAGS.varsel:
            if i_exp == 1:
                all_weights = sess.run(SITE.weights_in[0])
                all_beta = sess.run(SITE.weights_pred)
            else:
                all_weights = np.dstack((all_weights, sess.run(SITE.weights_in[0])))
                all_beta = np.dstack((all_beta, sess.run(SITE.weights_pred)))

        ''' Save results_try1 and predictions '''
        all_valid.append(I_valid)
        if FLAGS.varsel:
            np.savez(npzfile, pred=out_preds_train, loss=out_losses, w=all_weights, beta=all_beta, val=np.array(all_valid))
        else:
            np.savez(npzfile, pred=out_preds_train, loss=out_losses, val=np.array(all_valid))

        if has_test:
            np.savez(npzfile_test, pred=out_preds_test)

        ''' Save representations '''
        if FLAGS.save_rep and i_exp == 1:
            np.savez(repfile, rep=reps)

            if has_test:
                np.savez(repfile_test, rep=reps_test)

def main(argv=None):  # pylint: disable=unused-argument
    """ Main entry point """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
    outdir = FLAGS.outdir+'/results_'+timestamp+'/'
    os.mkdir(outdir)



    try:
        run(outdir)
    except Exception as e:
        with open(outdir+'error.txt','w') as errfile:
            errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise

if __name__ == '__main__':
    tf.app.run()
