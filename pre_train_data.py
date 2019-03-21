#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 16:10:41 2018

train & dev : 4608 utts 

@author: wuxc
"""

import numpy as np
from scipy.io import wavfile
from mylib import time_freq_represent,listFile,get_ideal_mask,add_noise,concat_frames,write_mat
import matlab
import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath(eng.genpath('/home/wuxc/features/'))
trainwave_path = '/home/wuxc/dataset/TIMIT/train/'
train_filelist = listFile(trainwave_path, '.wav')
noise_path = '/home/wuxc/dataset/noiseX_16k/'

SNR_train = [-3, 0, 3]
noise_train = ['babble', 'factory1', 'white', 'pink']

with open('./train/lps_noisy.txt',   'w') as f_lps_noisy, \
     open('./train/lps_clean.txt',   'w') as f_lps_clean, \
     open('./train/com_feat.txt',   'w') as f_com_feat, \
     open('./train/com_feat_5frames.txt',   'w') as f_com_feat_5frames, \
     open('./train/irm.txt',   'w') as f_irm, \
     open('./train/irm_2.txt',   'w') as f_irm_2, \
     open('./train/irm_2_1.txt',   'w') as f_irm_2_1, \
     open('./train/utt_info.txt',   'w') as f_utt, \
     open('./dev/lps_noisy.txt',   'w') as f_lps_noisy_dev, \
     open('./dev/ps_clean.txt',   'w') as f_lps_clean_dev, \
     open('./dev/com_feat.txt',   'w') as f_com_feat_dev, \
     open('./dev/com_feat_5frames.txt',   'w') as f_com_feat_5frames_dev, \
     open('./dev/irm.txt',   'w') as f_irm_dev, \
     open('./dev/irm_2.txt',   'w') as f_irm_2_dev, \
     open('./dev/irm_2_1.txt',   'w') as f_irm_2_1_dev, \
     open('./dev/utt_info.txt',   'w') as f_utt_dev: #utt_index noise_type snr num_frames
    
    batch = len(train_filelist) // (len(SNR_train) * len(noise_train))
    batch_i = 0
    for snr in SNR_train:
        for noise_type in noise_train:
            [fs, noise_wav] = wavfile.read(noise_path + noise_type + '.wav') 
            noise_wav = np.asarray(noise_wav, dtype=np.int32)
            
            i = 0
            for wav_file in train_filelist[batch_i * batch:(batch_i+1)*batch]:
                [fs, clean_sig] = wavfile.read(wav_file)
                clean_sig = np.asarray(clean_sig, dtype=np.int32)
                noise_start = np.random.choice(len(noise_wav)-len(clean_sig))
                noise_end = noise_start + len(clean_sig)
                noise_sig = noise_wav[noise_start:noise_end]
                noisy_sig = add_noise(clean_sig, noise_sig, snr)
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
                num_frames = tf_noisy.shape[1]
               
                flag = np.random.randint(10)
                if flag <= 8:
                    print('train')
                    write_mat(f_lps_noisy, np.float32(lps_noisy).T)
                    write_mat(f_lps_clean, np.float32(lps_clean).T)
                    write_mat(f_com_feat, np.float32(com_feat).T)
                    write_mat(f_com_feat_5frames, np.float32(com_feat_5frames).T)
                    write_mat(f_irm, np.float32(irm).T)
                    write_mat(f_irm_2, np.float32(irm_2).T)
                    write_mat(f_irm_2_1, np.float32(irm_2_1).T)
                    f_utt.write(wav_file + ' ' + noise_type + ' ' + str(snr) + ' ' + str(num_frames) + '\n')
                    write_file = '/home/wuxc/dataset/train/wav_noisy/'+ noise_type + '_' + str(snr) + '_' + wav_file.split('/')[-1]
                    wavfile.write(write_file, fs, noisy_sig)
                else:
                    print('dev')
                    write_mat(f_lps_noisy_dev, np.float32(lps_noisy).T)
                    write_mat(f_lps_clean_dev, np.float32(lps_clean).T)
                    write_mat(f_com_feat_dev, np.float32(com_feat).T)
                    write_mat(f_com_feat_5frames_dev, np.float32(com_feat_5frames).T)
                    write_mat(f_irm_dev, np.float32(irm).T)
                    write_mat(f_irm_2_dev, np.float32(irm_2).T)
                    write_mat(f_irm_2_1_dev, np.float32(irm_2_1).T)
                    f_utt_dev.write(wav_file + ' ' + noise_type + ' ' + str(snr) + ' ' + str(num_frames) + '\n')
                    write_file = '/home/wuxc/dataset/dev/wav_noisy/'+ noise_type + '_' + str(snr) + '_' + wav_file.split('/')[-1]
                    wavfile.write(write_file, fs, noisy_sig)
                i += 1
                print(batch_i*batch + i)
            batch_i += 1
eng.exit()              
