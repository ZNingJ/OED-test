import os
import pathlib
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *
from collections import Counter
import datetime

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

def RunModel(_abnormal_data, _abnormal_label, _hidden_num, _file_name,temp_list):
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
    print('_abnormal_label:{0}'.format(Counter(_abnormal_label)))
    zscore_abs = np.fabs(zscore)
    result_temp = []
    print('m的取值有：{0}'.format(temp_list))
    for m in temp_list:
        count=0
        index = np.argpartition(zscore_abs, -m)[-m:]
        for each_index in index:
            if _abnormal_label[each_index] == -1:
                count += 1
        result_temp.append(count)
    print(result_temp)

    t2=datetime.datetime.now()
    print('从当前时间结束:{0}'.format(t2))
    print('一共用时：{0}'.format(t2-t1))
    return anomaly_score, precision, recall, f1, roc_auc,precision , cks

if __name__ == '__main__':
    batch_num = 1
    hidden_num = 12
    k_partition = 10
    iteration = 50
    ensemble_space = 20
    learning_rate = 1e-3
    save_model = False
    _normalize = True

    for n in range(1,8):
        dataset = 1
        # 1-ionosphere,2-Musk2,3-ISOLET,4-MF-3,5-Arrhythmia,6-MF-5,7-MF-7
        if dataset == 1:
            _file_name = r"data/ionosphere.txt"
            print('当前数据集是：{0}'.format(_file_name))
            t1=datetime.datetime.now()
            print('从当前时间开始:{0}'.format(t1))
            abnormal_data = np.loadtxt(_file_name, delimiter=",", usecols=np.arange(0, 34))
            abnormal_label = np.loadtxt(_file_name, delimiter=",", usecols=(-1,))
            abnormal_label[abnormal_label == 0] = -1
            abnormal_label[abnormal_label == 1] = 1
            # abnormal_label = np.expand_dims(abnormal_label, axis=1)

            if _normalize == True:
                scaler = MinMaxScaler(feature_range=(0, 1))
                abnormal_data = scaler.fit_transform(abnormal_data)
            temp_list=[5,10,30,60,90,120,130,140,150,200,300,340]
            s_precision = []
            s_recall = []
            s_f1 = []
            s_roc_auc = []
            s_pr_auc = []
            s_cks = []
            error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label, hidden_num,_file_name=_file_name,temp_list=temp_list)
            s_precision.append(precision)
            s_recall.append(recall)
            s_f1.append(f1)
            s_roc_auc.append(roc_auc)
            s_pr_auc.append(pr_auc)
            s_cks.append(cks)

        if dataset==2:
            _file_name = r"data/clean2.data"
            print('当前数据集是：{0}'.format(_file_name))
            t1 = datetime.datetime.now()
            print('从当前时间开始:{0}'.format(t1))
            X = pd.read_csv(_file_name, header=None, index_col=None, skiprows=0, sep=',')
            abnormal_data = X.iloc[:, 2:168].as_matrix()
            abnormal_label = X.iloc[:, 168].as_matrix()
            abnormal_label[abnormal_label == 1] = -1
            abnormal_label[abnormal_label == 0] = 1
            # abnormal_label = np.expand_dims(abnormal_label, axis=1)

            if _normalize == True:
                scaler = MinMaxScaler(feature_range=(0, 1))
                abnormal_data = scaler.fit_transform(abnormal_data)
            temp_list=[1000,2000,3000,4000,5000,6000,6598]
            s_precision = []
            s_recall = []
            s_f1 = []
            s_roc_auc = []
            s_pr_auc = []
            s_cks = []
            error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label, hidden_num,
                                                                          _file_name=_file_name,temp_list=temp_list)
            s_precision.append(precision)
            s_recall.append(recall)
            s_f1.append(f1)
            s_roc_auc.append(roc_auc)
            s_pr_auc.append(pr_auc)
            s_cks.append(cks)

        if dataset==3:
            _file_name = r"data/ISOLET-23/data_23.dat"
            print('当前数据集是：{0}'.format(_file_name))
            t1 = datetime.datetime.now()
            print('从当前时间开始:{0}'.format(t1))
            X = pd.read_csv(_file_name, header=None, index_col=None, skiprows=0, sep=',')
            abnormal_data = X.as_matrix()
            y_loc = r"data/ISOLET-23/classid_23.dat"
            y = pd.read_csv(y_loc, header=None, index_col=None, skiprows=0, sep=',')
            abnormal_label = y.iloc[:, 0].as_matrix()
            # abnormal_label = np.expand_dims(abnormal_label, axis=1)
            if _normalize == True:
                scaler = MinMaxScaler(feature_range=(0, 1))
                abnormal_data = scaler.fit_transform(abnormal_data)
            abnormal_label[abnormal_label != 23] = 1
            abnormal_label[abnormal_label == 23] = -1
            temp_list=[5,10,15,20,30,50,60,80,100,150]
            s_precision = []
            s_recall = []
            s_f1 = []
            s_roc_auc = []
            s_pr_auc = []
            s_cks = []
            error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label, hidden_num,
                                                                          _file_name=_file_name,temp_list=temp_list)
            s_precision.append(precision)
            s_recall.append(recall)
            s_f1.append(f1)
            s_roc_auc.append(roc_auc)
            s_pr_auc.append(pr_auc)
            s_cks.append(cks)

        if dataset==4:
            _file_name = r"data/MF-3/data_3.dat"
            print('当前数据集是：{0}'.format(_file_name))
            t1 = datetime.datetime.now()
            print('从当前时间开始:{0}'.format(t1))
            X = pd.read_csv(_file_name, header=None, index_col=None, skiprows=0, sep=',')
            abnormal_data = X.iloc[:, :649].as_matrix()
            y_loc = r"data/MF-3/classid_3.dat"
            y = pd.read_csv(y_loc, header=None, index_col=None, skiprows=0, sep=',')
            abnormal_label = y.iloc[:, 0].as_matrix()
            # abnormal_label = np.expand_dims(abnormal_label, axis=1)
            if _normalize == True:
                scaler = MinMaxScaler(feature_range=(0, 1))
                abnormal_data = scaler.fit_transform(abnormal_data)
            abnormal_label[abnormal_label != 3] = 1
            abnormal_label[abnormal_label == 3] = -1
            temp_list=[20,30,50,60,90,100,150]
            s_precision = []
            s_recall = []
            s_f1 = []
            s_roc_auc = []
            s_pr_auc = []
            s_cks = []
            error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label, hidden_num,
                                                                          _file_name=_file_name,temp_list=temp_list)
            s_precision.append(precision)
            s_recall.append(recall)
            s_f1.append(f1)
            s_roc_auc.append(roc_auc)
            s_pr_auc.append(pr_auc)
            s_cks.append(cks)

        if dataset==5:
            _file_name = r"data/Arrhythmia_withoutdupl_05_v03.dat"   #以2为分割点
            print('当前数据集是：{0}'.format(_file_name))
            t1 = datetime.datetime.now()
            print('从当前时间开始:{0}'.format(t1))
            X = pd.read_csv(_file_name, header=None, index_col=None, skiprows=0, sep=' ')
            abnormal_data = X.iloc[:, :260].as_matrix()
            abnormal_label = X.iloc[:, 260].as_matrix()
            # abnormal_label = np.expand_dims(abnormal_label, axis=1)
            if _normalize == True:
                scaler = MinMaxScaler(feature_range=(0, 1))
                abnormal_data = scaler.fit_transform(abnormal_data)
            abnormal_label[abnormal_label == 1] = -1
            abnormal_label[abnormal_label == 0] = 1
            temp_list=[5,10,15,25,30,35,45,50,55,60,80,90,100,110,120,140,150,160,170,180,190,200]
            s_precision = []
            s_recall = []
            s_f1 = []
            s_roc_auc = []
            s_pr_auc = []
            s_cks = []
            error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label, hidden_num,
                                                                          _file_name=_file_name,temp_list=temp_list)
            s_precision.append(precision)
            s_recall.append(recall)
            s_f1.append(f1)
            s_roc_auc.append(roc_auc)
            s_pr_auc.append(pr_auc)
            s_cks.append(cks)

        if dataset==6:
            _file_name = r"data/MF-5/data_5.dat"
            print('当前数据集是：{0}'.format(_file_name))
            t1 = datetime.datetime.now()
            print('从当前时间开始:{0}'.format(t1))
            X = pd.read_csv(_file_name, header=None, index_col=None, skiprows=0, sep=',')
            abnormal_data = X.iloc[:, :649].as_matrix()
            y_loc = r"data/MF-5/classid_5.dat"
            y = pd.read_csv(y_loc, header=None, index_col=None, skiprows=0, sep=',')
            abnormal_label = y.iloc[:, 0].as_matrix()

            # abnormal_label = np.expand_dims(abnormal_label, axis=1)
            if _normalize == True:
                scaler = MinMaxScaler(feature_range=(0, 1))
                abnormal_data = scaler.fit_transform(abnormal_data)
            abnormal_label[abnormal_label != 5] = 1
            abnormal_label[abnormal_label == 5] = -1
            temp_list=[20,30,50,60,70,100,150]
            s_precision = []
            s_recall = []
            s_f1 = []
            s_roc_auc = []
            s_pr_auc = []
            s_cks = []
            error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label, hidden_num,
                                                                          _file_name=_file_name,temp_list=temp_list)
            s_precision.append(precision)
            s_recall.append(recall)
            s_f1.append(f1)
            s_roc_auc.append(roc_auc)
            s_pr_auc.append(pr_auc)
            s_cks.append(cks)

        if dataset==7:
            _file_name = r"data/MF-7/data_7.dat"
            print('当前数据集是：{0}'.format(_file_name))
            t1 = datetime.datetime.now()
            print('从当前时间开始:{0}'.format(t1))
            X = pd.read_csv(_file_name, header=None, index_col=None, skiprows=0, sep=',')
            abnormal_data = X.iloc[:, :649].as_matrix()
            y_loc = r"data/MF-7/classid_7.dat"
            y = pd.read_csv(y_loc, header=None, index_col=None, skiprows=0, sep=',')
            abnormal_label = y.iloc[:, 0].as_matrix()

            # abnormal_label = np.expand_dims(abnormal_label, axis=1)
            if _normalize == True:
                scaler = MinMaxScaler(feature_range=(0, 1))
                abnormal_data = scaler.fit_transform(abnormal_data)
            abnormal_label[abnormal_label != 7] = 1
            abnormal_label[abnormal_label == 7] = -1
            temp_list=[20,30,50,60,90,100,150]
            s_precision = []
            s_recall = []
            s_f1 = []
            s_roc_auc = []
            s_pr_auc = []
            s_cks = []
            error, precision, recall, f1, roc_auc, pr_auc, cks = RunModel(abnormal_data, abnormal_label, hidden_num,
                                                                          _file_name=_file_name,temp_list=temp_list)
            s_precision.append(precision)
            s_recall.append(recall)
            s_f1.append(f1)
            s_roc_auc.append(roc_auc)
            s_pr_auc.append(pr_auc)
            s_cks.append(cks)
