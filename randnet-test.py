import os
import pathlib
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *
from collections import Counter

def Model(_abnormal_data, _abnormal_label, _hidden_num, _file_name):
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        # placeholder list
        p_input = tf.placeholder(tf.float32, shape=(_abnormal_data.shape[0], _abnormal_data.shape[1]))
        p_input_reshape = tf.reshape(p_input, [batch_num, _abnormal_data.shape[0] * _abnormal_data.shape[1]])

        # initialize weights randomly from a Gaussian distribution
        # step 1: create the initializer for weights
        weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        # step 2: create the weight variable with proper initialization
        w_enc = tf.get_variable(name="weight_enc", dtype=tf.float32, shape=[_abnormal_data.shape[0] * _abnormal_data.shape[1], _hidden_num], initializer=weight_initer)
        w_dec = tf.get_variable(name="weight_dec", dtype=tf.float32, shape=[_hidden_num, _abnormal_data.shape[0] * _abnormal_data.shape[1]], initializer=weight_initer)

        w_enc_sparse = tf.convert_to_tensor(np.random.randint(2, size=(_abnormal_data.shape[0] * _abnormal_data.shape[1], _hidden_num)), dtype=tf.float32)
        w_dec_sparse = tf.convert_to_tensor(np.random.randint(2, size=(_hidden_num, _abnormal_data.shape[0] * _abnormal_data.shape[1])), dtype=tf.float32)

        b_enc = tf.Variable(tf.zeros(_hidden_num), dtype=tf.float32)
        b_dec = tf.Variable(tf.zeros(_abnormal_data.shape[0] * _abnormal_data.shape[1]), dtype=tf.float32)

        bottle_neck = tf.nn.sigmoid(tf.matmul(p_input_reshape, w_enc * w_enc_sparse) + b_enc)
        dec_output = tf.nn.sigmoid(tf.matmul(bottle_neck, w_dec * w_dec_sparse) + b_dec)
        dec_output_reshape = tf.reshape(dec_output, [batch_num, _abnormal_data.shape[0], _abnormal_data.shape[1]])

        loss = tf.reduce_mean(tf.square(p_input - dec_output_reshape))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    return g, p_input, dec_output_reshape, loss, optimizer, saver

def RunModel(_abnormal_data, _abnormal_label, _hidden_num, _file_name):
    error = []
    for j in range(ensemble_space):
        graph, p_input, dec_outputs, loss, optimizer, saver = Model(_abnormal_data, _abnormal_label, _hidden_num, _file_name)
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
                # print('iter %d:' % (i + 1), loss_val)

            if save_model:
                save_path = saver.save(sess, './saved_model/' + pathlib.Path(_file_name).parts[
                    0] + '/randnet_' + os.path.basename(_file_name) + '.ckpt')
                print("Model saved in path: %s" % save_path)

            (input_, output_) = sess.run([p_input, dec_outputs], {p_input: _abnormal_data})
            error.append(SquareErrorDataPoints(np.expand_dims(input_, 0), output_))

    ensemble_errors = np.asarray(error)
    anomaly_score = CalculateFinalAnomalyScore(ensemble_errors)

    zscore = Z_Score(anomaly_score)
    y_pred = CreateLabelBasedOnZscore(zscore, 0.5)
    print('anomaly_score:{0}'.format(anomaly_score))
    print('y_pred:{0}'.format(y_pred))
    print('_abnormal_label:{0}'.format(_abnormal_label))
    print('anomaly_score:{0}'.format(Counter(_abnormal_label)))
    print('y_pred:{0}'.format(Counter(y_pred)))

    result_temp=[]
    temp_list=[5,10,30,60,90,120,130,140,150,200,300,340]
    max_pred=Counter(_abnormal_label)[-1]
    print('max_pred:{0}'.format(max_pred))
    for m in temp_list:
        m_count=0
        real_count=0
        if m>max_pred:
            m=max_pred
        for index, j in enumerate(y_pred):
            if m_count<m:
                if j==-1:
                    m_count+=1
                    if _abnormal_label[index]==-1:
                        real_count+=1
            else:
                result_temp.append(real_count)
                break
    print(result_temp)

    precision, recall, f1 = CalculatePrecisionRecallF1Metrics(_abnormal_label, y_pred)
    fpr, tpr, roc_auc = CalculateROCAUCMetrics(_abnormal_label, anomaly_score)
    precision_curve, recall_curve, average_precision = CalculatePrecisionRecallCurve(_abnormal_label, anomaly_score)
    cks = CalculateCohenKappaMetrics(_abnormal_label, y_pred)

    return anomaly_score, precision, recall, f1, roc_auc, average_precision, cks

if __name__ == '__main__':
    batch_num = 1
    hidden_num = 12
    k_partition = 10
    iteration = 50
    ensemble_space = 20
    learning_rate = 1e-3
    save_model = False
    _normalize = True

    dataset = 1
    if dataset == 1:
        _file_name = r"data/ionosphere.txt"

        abnormal_data = np.loadtxt(_file_name, delimiter=",", usecols=np.arange(0, 34))
        abnormal_label = np.loadtxt(_file_name, delimiter=",", usecols=(-1,))
        # abnormal_label = np.expand_dims(abnormal_label, axis=1)
    else:
        _file_name = r"data/clean2.data"
        X = pd.read_csv(_file_name, header=None, index_col=None, skiprows=0, sep=',')
        abnormal_data = X.iloc[:, 2:168].as_matrix()
        abnormal_label = X.iloc[:, 168].as_matrix()
        # abnormal_label = np.expand_dims(abnormal_label, axis=1)

    if _normalize == True:
        scaler = MinMaxScaler(feature_range=(0, 1))
        abnormal_data = scaler.fit_transform(abnormal_data)
    abnormal_label[abnormal_label == 1] = -1
    abnormal_label[abnormal_label == 0] = 1


    # _file_name = r"data/ISOLET-23/data_23.dat"
    # X = pd.read_csv(_file_name, header=None, index_col=None, skiprows=0, sep=',')
    # abnormal_data = X.as_matrix()
    # y_loc = r"data/ISOLET-23/classid_23.dat"
    # y = pd.read_csv(y_loc, header=None, index_col=None, skiprows=0, sep=',')
    # abnormal_label = y.iloc[:, 0].as_matrix()
    # # abnormal_label = np.expand_dims(abnormal_label, axis=1)
    # if _normalize == True:
    #     scaler = MinMaxScaler(feature_range=(0, 1))
    #     abnormal_data = scaler.fit_transform(abnormal_data)
    # abnormal_label[abnormal_label != 23] = 1
    # abnormal_label[abnormal_label == 23] = -1

    # _file_name = r"data/MF-3/data_3.dat"
    # X = pd.read_csv(_file_name, header=None, index_col=None, skiprows=0, sep=',')
    # abnormal_data = X.as_matrix()
    # y_loc = r"data/MF-3/classid_3.dat"
    # y = pd.read_csv(y_loc, header=None, index_col=None, skiprows=0, sep=',')
    # abnormal_label = y.iloc[:, 0].as_matrix()
    # # abnormal_label = np.expand_dims(abnormal_label, axis=1)
    # if _normalize == True:
    #     scaler = MinMaxScaler(feature_range=(0, 1))
    #     abnormal_data = scaler.fit_transform(abnormal_data)
    # abnormal_label[abnormal_label != 3] = 1
    # abnormal_label[abnormal_label == 3] = -1

    # _file_name = r"data/Arrhythmia_withoutdupl_05_v03.dat"   #以2为分割点
    # X = pd.read_csv(_file_name, header=None, index_col=None, skiprows=0, sep=' ')
    # abnormal_data = X.iloc[:, :260].as_matrix()
    # abnormal_label = X.iloc[:, 260].as_matrix()
    # # abnormal_label = np.expand_dims(abnormal_label, axis=1)
    # if _normalize == True:
    #     scaler = MinMaxScaler(feature_range=(0, 1))
    #     abnormal_data = scaler.fit_transform(abnormal_data)
    # abnormal_label[abnormal_label == 1] = -1
    # abnormal_label[abnormal_label == 0] = 1


    s_precision = []
    s_recall = []
    s_f1 = []
    s_roc_auc = []
    s_pr_auc = []
    s_cks = []
    error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label, hidden_num,
                                                                  _file_name=_file_name)
    s_precision.append(precision)
    s_recall.append(recall)
    s_f1.append(f1)
    s_roc_auc.append(roc_auc)
    s_pr_auc.append(pr_auc)
    s_cks.append(cks)

    print('########################################')
    avg_precision = CalculateAverageMetric(s_precision)
    print('avg_precision=' + str(avg_precision))
    avg_recall = CalculateAverageMetric(s_recall)
    print('avg_recall=' + str(avg_recall))
    avg_f1 = CalculateAverageMetric(s_f1)
    print('avg_f1=' + str(avg_f1))
    avg_roc_auc = CalculateAverageMetric(s_roc_auc)
    print('avg_roc_auc=' + str(avg_roc_auc))
    avg_pr_auc = CalculateAverageMetric(s_pr_auc)
    print('avg_pr_auc=' + str(avg_pr_auc))
    avg_cks = CalculateAverageMetric(s_cks)
    print('avg_cks=' + str(avg_cks))
    print('########################################')