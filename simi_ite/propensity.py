# import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn import tree
import cPickle


def propensity_score_training(data, label, mode):

    '''
    :param data: pre-treatment covariates
    :param label: treatment that the units accually took
    :param mode: the method to to get the propsensity score
    :return: the propensity socre (the probability that a unit is in the treated group); the trainied propensity calcualtion model
    '''

    train_x, eva_x, train_t, eva_t = train_test_split(data, label, test_size=0.3, random_state=42)

    if mode == 'Logistic-regression':
        n_treat = np.sum(data)
        n_control = data.shape[0] - n_treat
        control_weight = n_treat/float(n_control)
        train_t = train_t.flatten()
        weight_array = np.ones([train_t.shape[0]])

        clf = LogisticRegression('l2', class_weight='balanced',C=3.0 )
        clf.fit(train_x, train_t.flatten())

        pred_eva = clf.predict(eva_x)
        pred_train = clf.predict(train_x)
        acc_train = accuracy_score(train_t, pred_train)
        f1_train = f1_score(train_t, pred_train)
        f1_eva = f1_score(eva_t, pred_eva)


        prob_eva = clf.predict_proba(eva_x)
        prob_train = clf.predict_proba(train_x)

        acc_eva = accuracy_score(eva_t, pred_eva)

        prob_all = clf.predict_proba(data)
        result_all = clf.predict(data)

        return prob_all, clf
    if mode == 'SVM':
        n_treat = np.sum(data)
        n_control = data.shape[0] - n_treat
        control_weight = n_treat / float(n_control)
        train_t = train_t.flatten()

        clf = svm.SVC(probability=True, class_weight = 'balanced')
        clf.fit(train_x, train_t.flatten())
        pred_eva = clf.predict(eva_x)

        pred_train = clf.predict(train_x)
        acc_train = accuracy_score(train_t, pred_train)
        f1_train = f1_score(train_t, pred_train)
        f1_eva = f1_score(eva_t, pred_eva)

        prob_eva = clf.predict_proba(eva_x)
        prob_train = clf.predict_proba(train_x)

        acc_eva = accuracy_score(eva_t, pred_eva)
        print acc_train
        print acc_eva
        print f1_train
        print f1_eva
        prob_all = clf.predict_proba(data)
        result_all = clf.predict(data)
        print result_all[1:10]
        print prob_all[1:10,:]
        return prob_all, clf
    if mode == 'CART':
        clf = tree.DecisionTreeClassifier(max_depth=6 ,class_weight = 'balanced')
        clf = clf.fit(train_x, train_t.flatten())
        pred_eva = clf.predict(eva_x)
        pred_eva_prob = clf.predict_proba(eva_x)

        f1_eva = f1_score(eva_t, pred_eva)
        acc_eva = accuracy_score(eva_t, pred_eva)
        print pred_eva_prob
        print acc_eva
        print f1_eva

def onehot_trans(t, catog):
    # control treat: [0,1] ======> treated
    trans = np.zeros([t.shape[0], catog.size])
    for i in range(t.shape[0]):
        if t[i,0] == 0:
            trans[i,0] = 1
        else:
            trans[i,1] = 1
    return trans

def load_propensity_score(model_file_name,x):
    loaded_model = cPickle.load(open(model_file_name, 'rb'))
    result = loaded_model.predict_proba(x)
    propensity_score = result[:,1]
    propensity_score = propensity_score.flatten()
    return propensity_score







