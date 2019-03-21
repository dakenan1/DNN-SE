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
from mylib import time_freq_represent, time_freq_synthesize, calc_stoi, calc_SDR, listFile
from scipy.io import wavfile
import time

def get_batch(train_file, train_size, batch_size, batch_dim):
    x_ = np.zeros((batch_size, batch_dim))
    line_i = 0  #if train_size % batch_size != 0  the last batch contains the tail and the head
    while True:
        with open(train_file) as f:
#            line_i = 0   #if train_size % batch_size != 0  the tail will not be trained
            line_sum = 0
            while line_sum < train_size:
                temp_line = f.readline().strip()
                #if temp_line is None or temp_line == '':
                #    f.seek(0)
                #    temp_line = f.readline().strip()
                try:
                    x_cur_line = np.asarray(list(map(float, temp_line.strip().split(' '))))
                except:
                    raise Exception(temp_line)
                x_[line_i, :] = x_cur_line
                line_i += 1
                line_sum += 1
                if line_i == batch_size:
                    yield x_
                    x_ = np.zeros((batch_size, batch_dim))
                    line_i = 0

def generator_train(x_file, y_file, train_size, batch_size, x_dim, y_dim):
    g_x = get_batch(x_file, train_size, batch_size, x_dim)
    g_y = get_batch(y_file, train_size, batch_size, y_dim)
    while True:
        yield (next(g_x), next(g_y))

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
            flag = np.random.randint(500)
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
    
def test_model(model_id, x_mean, x_std):
    config = tf.ConfigProto(
        device_count = {'GPU': 0}
            )
    sess = tf.Session(config=config)
    if not os.path.exists('./result_temp/%s/wav/'%model_id):
        os.makedirs('./result_temp/%s/wav/'%model_id)
    model_list = listFile('./model/%s/'%model_id, '.meta')
    model_file = max(model_list, key=os.path.getctime)
    print('load model : %s'%model_file)
    saver = tf.train.import_meta_graph(model_file)
    saver.restore(sess, tf.train.latest_checkpoint('./model/%s/'%model_id))
    graph = tf.get_default_graph()  
    #for op in graph.get_operations():
    #    print(op.name)

    x = graph.get_operation_by_name('x').outputs[0] 
    keep_prob = graph.get_operation_by_name('keep_prob').outputs[0] 
    y_pred = tf.get_collection('y_pred')[0]
    x_test = np.loadtxt('./dataset/test/com_feat_5frames.txt')
    x_test = standard_scale(x_test, x_mean, x_std)
    print('Testing')
    [irm_pred] = sess.run([y_pred], feed_dict={x:x_test, keep_prob:1})
    print('irm_pred_mean:',np.mean(irm_pred), 'irm_pred_std:',np.std(irm_pred))
    np.savetxt('./result_temp/%s/irm_pred.txt'%model_id, irm_pred)
    sess.close()

def evaluate_model(model_id):
    start = time.time()
    print('Evaluating result')
    print('loading data')
    IRM_pred = np.loadtxt('./result_temp/%s/irm_pred.txt'%model_id)
    IRM_label = np.loadtxt('/home/wuxc/dataset/test/irm_2_1.txt')
    load_time = time.time() - start
    print('load_time: %.3f'%load_time)
    result_file = '/home/wuxc/result_temp/%s/result.txt'%model_id
    with open('/home/wuxc/dataset/test/utt_info.txt') as f, open(result_file, 'w') as r:
        sum_frames = 0
        utt_i = 0
        irm_cost = 0
        while True:
            line = f.readline()
            if line == "":
                break
            else:
                utt, noise_type, snr, num_frames, sig_len = line.strip().split(' ')
                snr = int(snr)
                num_frames = int(num_frames)
                sig_len = int(sig_len)
                irm_pred = IRM_pred[sum_frames: sum_frames + num_frames+1, :]
                irm_label = IRM_label[sum_frames: sum_frames + num_frames+1, :]
                cur_cost = np.mean(np.square(irm_pred - irm_label))
                irm_cost += cur_cost / 400
                sum_frames += num_frames
                [fs, noisy_sig] = wavfile.read('./dataset/test/wav_noisy/'+ noise_type +'_' + str(snr) + '_' +utt.split('/')[-1])
                [tf_noisy, params] = time_freq_represent(noisy_sig, 320, 160, 512)
                print(sum_frames)
                print(num_frames)
                print(tf_noisy.shape)
                print(irm_pred.shape) 
                pow_pred = np.square(np.abs(tf_noisy)) * irm_pred.T
                tf_mat = np.sqrt(pow_pred) * (tf_noisy / np.abs(tf_noisy))
                enhanced_sig = time_freq_synthesize(tf_mat, params)        
                #if utt_i % 10 == 0:
                wavfile.write('./result_temp/%s/wav/'%model_id +'%s_%d_'%(noise_type,snr) +  utt.split('/')[-1][:-4] + '_enhanced.wav', 16000, enhanced_sig) 
                [fs, clean_sig] = wavfile.read(utt)
                if len(clean_sig) != len(enhanced_sig):
                    print('ERROR:len(clean_sig) != len(enhanced_sig)')
                    break
                stoi_proc = calc_stoi(clean_sig, enhanced_sig, fs)
                stoi_ori = calc_stoi(clean_sig, noisy_sig, fs)
                sdr = calc_SDR(clean_sig, enhanced_sig)
                r.write(line.strip())
                r.write(' %.5f'%stoi_ori + ' %.5f'%stoi_proc + ' %.5f'%sdr + ' %.5f'%cur_cost +'\n')
                #r.write( ' %.5f'%sdr + ' %.5f'%cur_cost +'\n')                                   
                utt_i += 1
                if utt_i % 100 == 0:
                    print('%d/400'%(utt_i))
        print('irm_cost %.5f' %irm_cost)
    with open(result_file) as f:
        stoi_ori_mean = 0
        stoi_enhanced_mean = 0
        sdr_enhanced_mean = 0
        irm_cost_mean = 0
        line_i = 0
        while True:
            line = f.readline()
            if not line:
                break
            else:
                line_i += 1
                utt, noise_type, snr, num_frames, sig_len, stoi_ori, stoi_enhanced, sdr, irm_cost = line.strip().split(' ')
                snr = int(snr)
                stoi_ori = float(stoi_ori)
                stoi_enhanced = float(stoi_enhanced)
                sdr_enhanced = float(sdr)
                irm_cost = float(irm_cost)
                stoi_ori_mean += stoi_ori/10
                stoi_enhanced_mean += stoi_enhanced/10
                sdr_enhanced_mean += sdr_enhanced/10
                irm_cost_mean += irm_cost/10
                if line_i % 10 == 0:
                    print('%s\t%ddB:\tstoi:%.4f -> %.4f sdr:%.3f irm_cost:%.5f'%(noise_type, snr, stoi_ori_mean, stoi_enhanced_mean, sdr_enhanced_mean, irm_cost_mean))
                    stoi_ori_mean = 0
                    stoi_enhanced_mean = 0
                    sdr_enhanced_mean = 0
                    irm_cost_mean = 0
if __name__ == "__main__":
    x_file = './dataset/train/com_feat_5frames.txt'
    x_mean, x_std = scale_params(x_file)
    model_id = '1125-23:46'
    test_model(model_id=model_id, x_mean=x_mean, x_std=x_std)
    evaluate_model(model_id)
