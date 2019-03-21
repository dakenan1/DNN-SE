#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 15:20:02 2018

@author: wuxc
"""
import datetime
import numpy as np
import tensorflow as tf
import os
from mylib import time_freq_represent, time_freq_synthesize, calc_stoi, calc_SDR
from scipy.io import wavfile
import time

def load_data(txtfile, n_row, n_col):
    x_ = np.zeros((n_row, n_col))
    with open(txtfile) as f:
        line_sum = 0
        while line_sum < n_row:
            temp_line = f.readline().strip()
            if temp_line == "":
                break
            try:
                x_cur_line = np.asarray(list(map(float, temp_line.strip().split(' '))))
            except:
                raise Exception(temp_line)
            x_[line_sum, :] = x_cur_line
            line_sum += 1
            if line_sum % 10000 == 0:
                print('load %s %d/%d'%(txtfile, line_sum, n_row))
    return x_

def scale_params(txtfile):
    """
    randomly select 1/500 samples
    """
    with open(txtfile) as f:
        x = []
        while True:
            line = f.readline()
            if not line:
                break
            flag = np.random.randint(100)
            if flag == 0:
                line = line.strip().split(' ')
                x.append(list(map(float, line)))
        x = np.asarray(x)
        mean = np.mean(x, axis = 0)
        std = np.std(x, axis = 0)
    return (mean, std)
            
def standard_scale(x, mean, std): 
    batch_size = x.shape[0]
    for j in range(batch_size):
        x[j,:] = (x[j,:] - mean) / std
    return x
    
#def train_valid_split():

    
def train_model(model_id, x_mean, x_std, train_size, dev_size, total_batch, training_epochs=200, batch_size=1024, lr=0.001, EarlyStop=True, min_cost=0.15, dropout_keep=0.7):
    tf.set_random_seed(777)

    input_dim = 246*5
    layer1_unit = 1024
    layer2_unit = 1024
    layer3_unit = 1024
    layer4_unit = 1024
    output_dim = 257

    w1 = tf.get_variable('w1', shape=[input_dim, layer1_unit], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.truncated_normal([layer1_unit],stddev=0.1))

    w2 = tf.get_variable('w2', shape=[layer1_unit, layer2_unit], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.truncated_normal([layer2_unit],stddev=0.1))

    w3 = tf.get_variable('w3', shape=[layer2_unit, layer3_unit], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.truncated_normal([layer3_unit],stddev=0.1))

    w4 = tf.get_variable('w4', shape=[layer3_unit, layer4_unit], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.truncated_normal([layer4_unit],stddev=0.1))

    w5 = tf.get_variable('w5', shape=[layer4_unit, output_dim], initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.truncated_normal([output_dim],stddev=0.1))
    x = tf.placeholder(tf.float32, [None, input_dim], name='x')
    y = tf.placeholder(tf.float32, [None, output_dim])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    l1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    l1 = tf.nn.dropout(l1, keep_prob=keep_prob)
    l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
    l2 = tf.nn.dropout(l2, keep_prob=keep_prob)
    l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)
    l3 = tf.nn.dropout(l3, keep_prob=keep_prob)
    l4 = tf.nn.relu(tf.matmul(l3, w4) + b4)
    l4 = tf.nn.dropout(l4, keep_prob=keep_prob)
    y_pred = tf.nn.sigmoid(tf.matmul(l4, w5) + b5)
    tf.add_to_collection('y_pred', y_pred)
    #lps_noisy = tf.placeholder(tf.float32, [None, output_dim], name='lps_noisy')
    #lps_clean = tf.placeholder(tf.float32, [None, output_dim], name='lps_clean')
    #spec_pred = y_pred * tf.exp(lps_noisy)
    #cost = tf.reduce_mean(tf.square(tf.log(tf.clip_by_value(spec_pred, 1e-10, tf.reduce_max(spec_pred))) - lps_clean))
    #cost = tf.reduce_mean(tf.square(spec_pred - tf.exp(lps_clean)))
    cost = tf.reduce_mean(tf.square(y - y_pred))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    
    config = tf.ConfigProto(
        device_count = {'GPU': 1}
            )
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    save_path = './model/%s/'%model_id 
    print(save_path);
    start = time.time()
    x_valid = load_data('./dataset/dev/com_feat_5frames.txt', dev_size, input_dim)
    y_valid = load_data('./dataset/dev/irm_2_1.txt', dev_size, output_dim)
    x_valid = standard_scale(x_valid, x_mean, x_std)
    x_train = load_data('./dataset/train/com_feat_5frames.txt', train_size, input_dim)
    y_train = load_data('./dataset/train/irm_2_1.txt', train_size, output_dim)
    x_train = standard_scale(x_train, x_mean, x_std)
    #lps_noisy_train = load_data('./data/train/lps_noisy.txt', train_size, output_dim)
    #lps_clean_train = load_data('./data/train/lps_clean.txt', train_size, output_dim)
    #lps_noisy_dev = load_data('./data/dev/lps_noisy.txt', dev_size, output_dim)
    #lps_clean_dev = load_data('./data/dev/lps_clean.txt', dev_size, output_dim)
    print('load data time : %.2f'%(time.time() - start))
    if not os.path.exists('./model/%s/'%model_id):
        os.makedirs('./model/%s/'%model_id)
    if EarlyStop == True:
        print('Start Training...')
        patience = 0
        for epoch in range(training_epochs):
            train_cost = 0
            seed_i = np.random.randint(200)
            np.random.seed(seed_i)
            np.random.shuffle(x_train)
            np.random.seed(seed_i)
            np.random.shuffle(y_train)
            for batch in range(total_batch):
                if batch % 500 == 0:
                    print('batch %d/%d'%(batch, total_batch))
                x_train_batch = x_train[batch_size*batch:batch_size*(batch+1),:]
                y_train_batch = y_train[batch_size*batch:batch_size*(batch+1),:]
                #lps_noisy_batch = lps_noisy_train[batch_size*batch:batch_size*(batch+1),:]
                #lps_clean_batch = lps_clean_train[batch_size*batch:batch_size*(batch+1),:]
                
                #feed_dict = {x:x_train_batch, y:y_train_batch, keep_prob:dropout_keep, lps_noisy:lps_noisy_batch, lps_clean:lps_clean_batch}
                feed_dict = {x:x_train_batch, y:y_train_batch, keep_prob:dropout_keep}
                [c] = sess.run([cost], feed_dict=feed_dict)
                [c,_] = sess.run([cost, optimizer], feed_dict=feed_dict)
                train_cost += c/total_batch
#            [valid_cost] = sess.run([cost],feed_dict={x:x_valid, y:y_valid, keep_prob:1, lps_noisy:lps_noisy_dev, lps_clean:lps_clean_dev})
            [valid_cost] = sess.run([cost],feed_dict={x:x_valid, y:y_valid, keep_prob:1})
            print('Epoch:','{}'.format(epoch + 1), 'train_cost=', '{:.4f}'.format(train_cost), 'valid_cost=', '{:.4f}'.format(valid_cost))
            if epoch >= 0:
                if valid_cost < min_cost:
                    min_cost = valid_cost
                    saver.save(sess, './model/%s/%.4f_%.4f.ckpt'%(model_id, train_cost, valid_cost))
                    print('model saved')
                    patience = 0
                else:
                    patience += 1
                if patience > 10:
                    print('overfitting!')
                    break
            print('Learning Finished!')
    saver.restore(sess, tf.train.latest_checkpoint(save_path))
    sess.close() 
    

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    timestamp = t = datetime.datetime.now().strftime('%m%d-%H:%M')
    x_file = './dataset/train/com_feat_5frames.txt'
    y_file = './dataset/train/irm_2_1.txt'
    train_size = 1279862
    dev_size = 146274
    batch_size = 256*8
    x_dim = 246*5
    y_dim = 257
    total_batch = train_size // batch_size
    print("MVN inputs")
    x_mean, x_std = scale_params(x_file)
    #model_id = timestamp
    model_id = '1125-23:46'
    train_model(model_id=model_id, x_mean=x_mean, x_std=x_std, train_size=train_size, dev_size=dev_size, total_batch=total_batch, training_epochs=200, batch_size=batch_size, lr=0.00001, EarlyStop=True, min_cost=np.inf, dropout_keep=0.7)
