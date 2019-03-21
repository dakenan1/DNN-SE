#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 16:10:41 2018

train_set:
    500 speech * 12 noise type * 3 SNR = 18000
test_set:
    60 speech * 18 noise type * 5SNR = 5400

@author: wuxc
"""

import numpy as np
from scipy.io import wavfile
from mylib import time_freq_represent,listFile,get_ideal_mask,add_noise,concat_frames,write_mat
import matlab
import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath(eng.genpath('/home/wuxc/web_dnn_toolbox/get_feat/features/'))
trainwave_path = '/home/wuxc/dataset/TIMIT/train/'
testwave_path = '/home/wuxc/dataset/TIMIT/test/'
train_filelist = listFile(trainwave_path, '.wav')
test_filelist = listFile(testwave_path, '.wav')
noise_path = '/home/wuxc/dataset/noiseX_16k/'

SNR_train = [-3, 0, 3]
SNR_test = [-6, -3, 0, 3, 6]
noise_train = ['babble']
noise_test = ['babble']

#mode = 'train'
mode = 'test'
if mode == 'train':
    filelist = train_filelist
    SNR = SNR_train
    noise = noise_train
else:
    filelist = test_filelist
    SNR = SNR_test
    noise = noise_test

with open('./'+mode+'/lps_noisy.txt',   'w') as f_lps_noisy, \
     open('./'+mode+'/lps_clean.txt',   'w') as f_lps_clean, \
     open('./'+mode+'/com_feat.txt',   'w') as f_com_feat, \
     open('./'+mode+'/com_feat_5frames.txt',   'w') as f_com_feat_5frames, \
     open('./'+mode+'/irm.txt',   'w') as f_irm, \
     open('./'+mode+'/irm_2.txt',   'w') as f_irm_2, \
     open('./'+mode+'/irm_2_1.txt',   'w') as f_irm_2_1, \
     open('./'+mode+'/utt_info.txt',   'w') as f_utt: #utt_index noise_type snr num_frames
    batch = 10    
    batch_i = 0
    for snr in SNR:
        for noise_type in noise:
            [fs, noise_wav] = wavfile.read(noise_path + noise_type + '.wav')
            noise_wav = np.asarray(noise_wav, dtype=np.int32)
            i = 0
#            np.random.shuffle(filelist)
            for wav_file in filelist[batch_i*batch : (batch_i+1)*batch]:
                [fs, clean_sig] = wavfile.read(wav_file)
                clean_sig = np.asarray(clean_sig, dtype=np.int32)
                noise_start = np.random.choice(len(noise_wav)-len(clean_sig))
                noise_end = noise_start + len(clean_sig)
                noise_sig = noise_wav[noise_start:noise_end]
                noisy_sig = add_noise(clean_sig, noise_sig, snr)
                write_file = '/home/wuxc/dataset/test/wav_noisy/'+ noise_type + '_' + str(snr) + '_' + wav_file.split('/')[-1]
                wavfile.write(write_file, fs, noisy_sig)
                
                [tf_noisy, params] = time_freq_represent(noisy_sig, frame_len=320, overlap_len=160, fft_len=512)
                [tf_clean, params] = time_freq_represent(clean_sig, frame_len=320, overlap_len=160, fft_len=512)
                [tf_noise, params] = time_freq_represent(noise_sig, frame_len=320, overlap_len=160, fft_len=512)
                pow_noisy =  np.abs(tf_noisy)**2
                pow_clean =  np.abs(tf_clean)**2
                pow_noise =  np.abs(tf_noise)**2
                lps_noisy =  np.log(pow_noisy)
                lps_clean =  np.log(pow_clean)
                com_feat = np.asarray(eng.my_features_AmsRastaplpMfccGf(eng.cell2mat(noisy_sig.tolist())))
                com_feat_5frames = concat_frames(com_feat, 2)
                irm = pow_clean / (pow_clean + pow_noise)
                irm_2 = pow_clean / pow_noisy
                irm_2_1 = np.where(irm_2<=1, irm_2, 1)
                phase_noisy = tf_noisy.real /  np.abs(tf_noisy)
                phase_clean = tf_clean.real /  np.abs(tf_clean)
                num_frames = tf_noisy.shape[1]
                print(num_frames) 
                write_mat(f_lps_noisy, np.float32(lps_noisy).T)
                write_mat(f_lps_clean, np.float32(lps_clean).T)
                write_mat(f_com_feat, np.float32(com_feat).T)
                write_mat(f_com_feat_5frames, np.float32(com_feat_5frames).T)
                write_mat(f_irm, np.float32(irm).T)
                write_mat(f_irm_2, np.float32(irm_2).T)
                write_mat(f_irm_2_1, np.float32(irm_2_1).T)
                print(wav_file + ' ' + noise_type + ' ' + str(snr) + ' ' + str(num_frames) + str(len(noisy_sig)))
                f_utt.write(wav_file + ' ' + noise_type + ' ' + str(snr) + ' ' + str(num_frames) + ' ' +  str(len(noisy_sig)) + '\n')
                i += 1
                print(batch_i*batch + i)
            batch_i += 1
eng.exit()              
