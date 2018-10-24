import tensorflow as tf
import numpy as np
from simi_ite.propensity import *
import math
from scipy.spatial.distance import cdist
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random



SQRT_CONST = 1e-10

FLAGS = tf.app.flags.FLAGS

def validation_split(D_exp, val_fraction):
    """ Construct a train/validation split """
    n = D_exp['x'].shape[0]

    if val_fraction > 0:
        n_valid = int(val_fraction*n)
        n_train = n-n_valid
        I = np.random.permutation(range(0,n))
        I_train = I[:n_train]
        I_valid = I[n_train:]
    else:
        I_train = range(n)
        I_valid = []

    return I_train, I_valid

def validation_split_equal(D_exp, val_fraction):
    """ Construct a train/validation split """
    n = D_exp.shape[0]

    if val_fraction > 0:
        n_valid = int(val_fraction*n)
        n_train = n-n_valid
        I = np.random.permutation(range(0,n))
        I_train = I[:n_train]
        I_valid = I[n_train:]
    else:
        I_train = range(n)
        I_valid = []

    return I_train, I_valid



def log(logfile,str):
    """ Log a string in a file """
    with open(logfile,'a') as f:
        f.write(str+'\n')
    print str

def save_config(fname):
    """ Save configuration """
    # flagdict =  FLAGS.__dict__['__flags']
    flagdict = FLAGS.flag_values_dict()
    s = '\n'.join(['%s: %s' % (k,str(flagdict[k])) for k in sorted(flagdict.keys())])
    f = open(fname,'w')
    f.write(s)
    f.close()

def load_data(fname):
    """ Load data set """
    if fname[-3:] == 'npz':
        data_in = np.load(fname)
        data = {'x': data_in['x'], 't': data_in['t'], 'yf': data_in['yf']}
        try:
            data['ycf'] = data_in['ycf']
        except:
            data['ycf'] = None
    else:
        if FLAGS.sparse>0:
            data_in = np.loadtxt(open(fname+'.y',"rb"),delimiter=",")
            x = load_sparse(fname+'.x')
        else:
            data_in = np.loadtxt(open(fname,"rb"),delimiter=",")
            x = data_in[:,5:]

        data['x'] = x
        data['t'] = data_in[:,0:1]
        data['yf'] = data_in[:,1:2]
        data['ycf'] = data_in[:,2:3]

    data['HAVE_TRUTH'] = not data['ycf'] is None

    data['dim'] = data['x'].shape[1]
    data['n'] = data['x'].shape[0]

    return data

def load_sparse(fname):
    """ Load sparse data set """
    E = np.loadtxt(open(fname,"rb"),delimiter=",")
    H = E[0,:]
    n = int(H[0])
    d = int(H[1])
    E = E[1:,:]
    S = sparse.coo_matrix((E[:,2],(E[:,0]-1,E[:,1]-1)),shape=(n,d))
    S = S.todense()

    return S

def safe_sqrt(x, lbound=SQRT_CONST):
    ''' Numerically safe version of TensorFlow sqrt '''
    return tf.sqrt(tf.clip_by_value(x, lbound, np.inf))


def pdist2sq(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*tf.matmul(X,tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X),1,keep_dims=True)
    ny = tf.reduce_sum(tf.square(Y),1,keep_dims=True)
    D = (C + tf.transpose(ny)) + nx
    return D

def pdist2(X,Y):
    """ Returns the tensorflow pairwise distance matrix """
    return safe_sqrt(pdist2sq(X,Y))


def simplex_project(x,k):
    """ Projects a vector x onto the k-simplex """
    d = x.shape[0]
    mu = np.sort(x,axis=0)[::-1]
    nu = (np.cumsum(mu)-k)/range(1,d+1)
    I = [i for i in range(0,d) if mu[i]>nu[i]]
    theta = nu[I[-1]]
    w = np.maximum(x-theta,0)
    return w


def sigmoid(x):
	return 1 / float(1 + math.exp(-x))

def similarity_score(s_i, s_j, mode = 'linear'):
    if mode == 'sigmoid':
	    _mid = (s_i + s_j)/float(2)
	    _dis = abs(s_j - s_i)/float(2)
	    score = 2*sigmoid(abs(_mid-0.5)) - 3*sigmoid(_dis)+1
    if mode == 'linear':
        _mid = (s_i + s_j) / float(2)
        _dis = abs(s_j - s_i) / float(2)
        score = (1.5 * abs(_mid - 0.5) - 2 * _dis + 1)/float(2)
    return score

def propensity_dist(x,y):
    s_x = load_propensity_score('./tmp/propensity_model.sav', x.reshape(1,x.shape[0]))
    s_y = load_propensity_score('./tmp/propensity_model.sav', y.reshape(1,y.shape[0]))

    edu_dist = np.power(np.linalg.norm(x-y),2)
    score = np.exp(-1*(1-similarity_score(s_x, s_y)) * edu_dist)
    '''Here the 1-similarity_score: when change the expression of similarity_score, '''
    # print score
    return score

def square_dist(x, y):
    dist = np.power(np.linalg.norm(x-y),2)
    return dist


def similarity_error_cal(x, h_rep_norm):
    distance_matrix_x = cdist(x, x, propensity_dist)
    distance_matrix_h = cdist(h_rep_norm, h_rep_norm, "sqeuclidean")
    dim = distance_matrix_h.shape[0]
    il2 = np.tril_indices(dim, -1)
    p_x = distance_matrix_x[il2]
    p_x = p_x/sum(p_x)
    p_h = distance_matrix_h[il2]
    p_h = p_h / sum(p_h)
    print p_x
    print p_h
    k_l = entropy(p_x, p_h)

    return k_l

def row_wise_dist(x):
    r = tf.reduce_sum(x * x, 1)

    # turn r into column vector
    r = tf.reshape(r, [-1, 1])
    D = r - 2 * tf.matmul(x, tf.transpose(x)) + tf.transpose(r)
    return D

def get_simi_ground(x, file_dir ='./simi_ite/tmp/propensity_model.sav' ):
    x_propensity_score = load_propensity_score(file_dir, x)
    n_train = x.shape[0]
    s_x_matrix = np.ones([n_train, n_train])
    for i in range(n_train):
      for j in range(n_train):
         s_x_matrix[i, j] = similarity_score(x_propensity_score[i], x_propensity_score[j])
    return s_x_matrix

def find_nearest_point(x, p):
    diff = np.abs(x-p)
    diff_1 = diff[diff>0]
    # print p
    min_val = np.min(diff_1)
    # print min_val
    I_diff = np.where(diff == min_val)[0]
    I_diff = I_diff[0]
    if I_diff.size>1:
        I_diff = I_diff[0]
    # print x[I_diff]
    return I_diff


def find_three_pairs(x, t, propensity_dir = './simi_ite/tmp/propensity_model.sav'):
    try:
        x_return = np.ones([6, x.shape[1]])
        I_x_return = np.zeros(6,dtype=np.int)
        x_propensity_score = load_propensity_score(propensity_dir, x)
        I_t = np.where(t>0)[0]
        I_c = np.where(t<1)[0]

        # print x_propensity_score.shape
        prop_t = x_propensity_score[I_t]
        prop_c = x_propensity_score[I_c]
        # print prop_c
        # print prop_t
        x_t = x[I_t]
        # print type(I_t)
        # print I_t[2]
        x_c = x[I_c]
        # find x_i, x_j
        index_t, index_c = find_middle_pair(prop_t, prop_c)
        # find x_k, x_l
        index_k = np.argmax(np.abs(prop_c - prop_t[index_t]))
        index_l = find_nearest_point(prop_c, prop_c[index_k])

        # find x_n, x_m
        index_m = np.argmax(np.abs(prop_t - prop_c[index_c]))
        index_n = find_nearest_point(prop_t, prop_t[index_m,])
        # print index_t
        # print index_c
        x_return[0,:] = x_t[index_t,:]
        x_return[1, :] = x_c[index_c, :]
        x_return[2, :] = x_c[index_k, :]
        x_return[3, :] = x_c[index_l, :]
        x_return[4, :] = x_t[index_m, :]
        x_return[5, :] = x_t[index_n, :]
        I_x_return[0] = int(I_t[index_t])
        I_x_return[1] = int(I_c[index_c])
        I_x_return[2] = int(I_c[index_k])
        I_x_return[3] = int(I_c[index_l])
        I_x_return[4] = int(I_t[index_m])
        I_x_return[5] = int(I_t[index_n])
    except:
        x_return = x[0:6,:]
        I_x_return = np.array([0,1,2,3,4,5])
        print('some error happens here!')

    return x_return, I_x_return

def find_middle_pair(x, y):
    min = np.abs(x[0]-0.5) + np.abs(y[0]-0.5)
    index_1 = 0
    index_2 = 0
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            value = np.abs(x[i]-0.5) + np.abs(y[j]-0.5)
            if  value < min:
                min = value
                index_1 = i
                index_2 = j
    return index_1, index_2

def get_three_pair_simi(three_pairs, file_dir ='./simi_ite/tmp/propensity_model.sav' ):
    three_pairs_simi = get_simi_ground(three_pairs, file_dir)
    simi = np.ones([5,1])
    '''
    S(k, l), S(m, n), S(k, l), S(i, k), S(j, m)
    '''
    simi[0, 0] = three_pairs_simi[2, 3]
    simi[1, 0] = three_pairs_simi[4, 5]
    simi[2, 0] = three_pairs_simi[2, 4]
    simi[3, 0] = three_pairs_simi[0, 2]
    simi[4, 0] = three_pairs_simi[1, 4]
    return simi


if __name__ == "__main__":
    main()