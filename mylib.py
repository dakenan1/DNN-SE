#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 16:31:42 2018

@author: jiaxp
"""

'''
my python lib for speech signal processing
listFile:
  list all files in a dir and sub dir with specified appendix
makedirs:
  make dir according to the dir name and base path. 
  if dir already exists, use num appendix to avoid repeat.
time_freq_represent/time_freq_synthesize:
  time freq analysis
calc_SDR:
  SDR calculation
'''

import os
import pickle
import numpy as np

#import pickle
#testpk1 = pickle.loads(open("test.pickle","rb").read())
#pickl.dump(w,open("testpk1_py2.pk1","wb"),protocol=2)

def write_mat(f, mat):
	[row, col] = mat.shape
	for i in range(row):
		for j in range(col):
			f.write(str(mat[i,j])+ ' ')
		f.write('\n')

def listFile(root_path, appendix='', search_log=False):
  ''' find all the files in the path (including sub-directories)
      with a specified appendix.
      if appendix=='' then return all the files
      return in a string list(filename with abs path)
  '''
  ap_len = len(appendix)
  if ap_len > 0:
    if appendix[0]!='.':
      appendix = '.'+appendix
      ap_len += 1
      
  log_path = os.path.join(root_path, '_log_{0}.txt'.format(appendix[1:]))
  if search_log:
    if os.path.exists(log_path) and os.path.getsize(log_path):
      with open(log_path,'rb') as f:
        file_list = pickle.load(f)
      return file_list
  # get absolute path
  root_path = os.path.abspath(root_path)
  # get the files and directories in current path
  temp_list = os.listdir(root_path)
  # list of wav file
  file_list = []
  
  for name in temp_list:
    name = os.path.join(root_path,name)
    # if is dir, check the files in it 
    if os.path.isdir(name):
      file_list.extend(listFile(name, appendix, False))
    # if is file, check whether it is wav file
    elif os.path.isfile(name):
      if ap_len > 0:
        if name[-1*ap_len:].lower()==appendix.lower():
          file_list.append(name)
      else:
        file_list.append(name)
        
  with open(log_path, 'wb') as f:
    pickle.dumps(file_list)
  return file_list

def makedirs(dir_name, base_path='.'):
  '''
  generate result directory.
  if directory name exits, try a new one.
  return the newly-made directory.
  '''
  directory_name = dir_name
  if not os.path.isabs(directory_name):
    directory_name = os.path.join(base_path, directory_name)
  if os.path.exists(directory_name):
    count = 1
    new_name = '{0}_{1}'.format(directory_name, count)
    while os.path.exists(new_name):
      count += 1
      new_name = '{0}_{1}'.format(directory_name, count)
    directory_name = new_name
  os.mkdir(directory_name)
  return directory_name

def add_noise(clean_sig, noise_sig, snr):
    Eclean = np.mean(np.square(clean_sig))
    Enoise = np.mean(np.square(noise_sig))
    ratio = np.sqrt(Eclean / Enoise / (10**(snr/10)))
    mix = clean_sig + ratio*noise_sig
    while np.max(np.abs(mix)) > 2**15-1:
        mix /= 2
    mix = np.int16(mix)
    return mix

def concat_frames(feature, frame_num):
    """
    feature: one utt
    shape: (feature_dim, frame)
    """
    dim_ori = feature.shape[0]
    N = feature.shape[1]
    n = frame_num
    feature_new_dim = len(feature) * (2*frame_num + 1)
    feature_new = np.zeros((feature_new_dim, N))
    for i in range(N):
        if i in range(0,n):
            for j in range(2*n+1):
                if j in range(n-i):
                    feature_new[j*dim_ori:(j+1)*dim_ori,i] = np.zeros((dim_ori))
                else:
                    feature_new[j*dim_ori:(j+1)*dim_ori,i] = feature[:,j-n+i]
        elif i in range(N-n, N):
            for j in range(2*n+1):
                if j in range(N-i+n,(2*n+1)):
                    feature_new[j*dim_ori:(j+1)*dim_ori,i] = np.zeros((dim_ori))
                else:
                    feature_new[j*dim_ori:(j+1)*dim_ori,i] = feature[:,j-n+i]
        else:
            for j in range(2*n+1):
                feature_new[j*dim_ori:(j+1)*dim_ori,i] = feature[:,j-n+i]
    return feature_new

def time_freq_represent(sig, frame_len=256, overlap_len=128,\
                        fft_len=256, if_window=True):
  '''
  calc the time-freq representation of the input signal
  inputs:
    sig:          input signal, time domain
    frame_len:    frame length(in points)
    overlap_len:  overlap length(in points)
    fft_len:      fft length(may equal to or twice of the length of frame)
    if_window:    true or false(applying of hanning window)
  outputs:
    tfmatrix:     tf representation of signal(half band)
    params:       params used to recover time domain signal
  '''  
  # generate window array
  if frame_len%2==1:
    print('WARNING: frame_len must be even, use default value(256)')
    frame_len = 256
  if if_window:
    win = np.hanning(frame_len+1)[:-1]
    scale = 1/win.sum()
  else:
    win = np.ones(frame_len)
    scale = 1
  
  # allocate the result array
  step = frame_len - overlap_len
  sig_len = len(sig)
  frame_num = int(np.ceil(sig_len/step))+1
  freq_bin_num = int(fft_len/2+1)
  tfmatrix = np.zeros([freq_bin_num,frame_num],dtype=np.complex)
  frame_matrix = np.zeros([frame_len, frame_num])
  
  # pad sig with trailing zeros
  sig = np.hstack((np.zeros(step), sig, np.zeros(frame_len)))
  for frame_i in range(0,frame_num):
    frame_matrix[:,frame_i] = sig[(frame_i)*step:(frame_i*step+frame_len)]*win
  
  
  # calc the tfmatrix
  tfmatrix = np.fft.rfft(frame_matrix,n=fft_len, axis=0)
  tfmatrix = tfmatrix*scale
  
  params = {}
  params['frame_len'] = frame_len
  params['overlap_len'] = overlap_len
  params['fft_len'] = fft_len
  params['if_window'] = if_window
  params['sig_len'] = sig_len
  
  return (tfmatrix, params)
#  return tfmatrix


def time_freq_synthesize(tfmatrix, params):
  '''
  get time domain signal from tf-representation
  inputs:
    tfmatrix:     tf-domain representation in shape (freq_bin_num, frame_num)
    params:       params used to recover signal, in dict
                  keys: frame_len, overlap_len, fft_len, if_window, sig_len
  outputs:
    sig:          time-domain signal, int16
  '''
  frame_len = params['frame_len']
  fft_len = params['fft_len']
  frame_matrix = np.fft.irfft(tfmatrix, n=fft_len, axis=0)
  if fft_len > frame_len:
    frame_matrix = frame_matrix[0:frame_len, :]
  
  if_window = params['if_window']
  if not if_window:
    print('WARNING: no window used! The result may not be true...')
    scale = 1
  else:
    win = np.hanning(frame_len+1)[:-1]
    scale = win.sum()
  
  # allocate sig array  
  overlap_len = params['overlap_len']
  step = frame_len - overlap_len
  frame_num = tfmatrix.shape[1]
  sig_len = step*(frame_num+1)
  sig = np.zeros(sig_len)
  
  # overlap-add
  for frame_i in range(frame_num):
    sig[frame_i*step:(frame_i+2)*step] += frame_matrix[:,frame_i]
  
  # remove the leading and trailing zeros added in represtation
  sig = sig[step:sig_len]
  origin_sig_len = params['sig_len']
  sig = np.around(sig[0:origin_sig_len]*scale)
  sig = np.array(sig, dtype=np.int16)
  
  return sig

def calc_SDR(sig_clean, sig_proc):
  '''
  calc SDR
  input:
    sig_clean:      clean signal (sig_len)
    sig_proc:       processed signal
  output:
    SDR:            SDR
  '''
  if sig_clean.shape != sig_proc.shape:
    print('ERROR:(calc_SDR) inputs shapes do not match.')
    return None
  sig_clean = np.asarray(sig_clean, dtype=np.int32)
  sig_proc = np.asarray(sig_proc, dtype=np.int32)
  E_sig = np.sum(sig_clean**2)
#  E_proc = np.sum(sig_proc**2)
#  E_distortion = np.abs(E_sig-E_proc)
  E_distortion = np.sum((sig_clean-sig_proc)**2)
  if E_distortion < 1e-10:
    E_distortion = 1e-10
  SDR = 10*np.log10(E_sig/E_distortion)

  return SDR


def get_ideal_mask(sig_clean, sig_noise,  method='IRM_s', lc=0):
  [tf_c, params] = time_freq_represent(sig_clean)
#  [tf_m, params] = time_freq_represent(sig_mixed)
  [tf_n, params] = time_freq_represent(sig_noise)
  tf_r = np.ones(tf_c.shape)
  [freq_num, frame_num] = tf_c.shape
  for freq_i in range(freq_num):
    for frame_i in range(frame_num):
      if method=='IBM':
        if np.abs(tf_c[freq_i, frame_i])<np.abs(tf_n[freq_i, frame_i])*10**(lc/20):
          tf_r[freq_i, frame_i] = 0
#      elif method=='IRM_t':
#        tf_r[freq_i, frame_i] = np.abs(tf_c[freq_i, frame_i])/(np.abs(tf_m[freq_i, frame_i])+1e-10)
#      elif method=='IRM_t_1':
#        tf_r[freq_i, frame_i] = np.abs(tf_c[freq_i, frame_i])/(np.abs(tf_m[freq_i, frame_i])+1e-10)
##        tf_r[freq_i, frame_i] = np.random.normal(0,0.2) +np.abs(tf_c[freq_i, frame_i])/(np.abs(tf_m[freq_i, frame_i])+1e-10)
#        if tf_r[freq_i,frame_i]>1:
#          tf_r[freq_i,frame_i]=1
      elif method=='IRM_s':
        cur_ec = np.abs(tf_c[freq_i, frame_i])**2
        cur_en = np.abs(tf_n[freq_i, frame_i])**2
        tf_r[freq_i, frame_i] = np.sqrt(cur_ec/(cur_ec+cur_en+1e-10))
      
  return tf_r


"""
Created on Wed Dec 13 14:20:28 2017

@author: jiaxp
"""

from scipy.signal import resample

def _thirdoct(fs, fft_len, channel_num, mid_freq_0):
  '''
  calc the 1/3 octave band
  inputs:
    fs:             sampling rate
    fft_len:        fft size
    channel_num:    number of bands
    mid_freq_0:     center frequency of the first 1/3 octave band
  outputs:
    bands_matrix:   band matrix
    center_freqs:   center frequencies
  '''
  freqs = np.linspace(0, fs, fft_len+1)[0:(fft_len//2+1)]
  band_index = np.arange(0,channel_num)
  center_freqs = 2**(band_index/3)*mid_freq_0
  #band_index = float(band_index)
  mid_freq_0 = float(mid_freq_0)

  freqs_L = np.sqrt(2**(band_index/3)*mid_freq_0*2**((band_index-1)/3)*mid_freq_0)
  freqs_R = np.sqrt(2**(band_index/3)*mid_freq_0*2**((band_index+1)/3)*mid_freq_0)

  bands_matrix = np.zeros([channel_num, len(freqs)])
  
  for ch_i in range(channel_num):
    L_index = np.argmin(np.abs(freqs-freqs_L[ch_i]))
    freqs_L[ch_i] = freqs[L_index]
    
    R_index = np.argmin(np.abs(freqs-freqs_R[ch_i]))
    freqs_R[ch_i] = freqs[R_index]
    
    bands_matrix[ch_i,L_index:R_index] = 1
  
  rnk = np.sum(bands_matrix,axis=1)
  for ch_i in np.arange(channel_num-1, 1, -1):
    if rnk[ch_i]>=rnk[ch_i-1] and rnk[ch_i]!=0:
      channel_num = ch_i +1
      break
  bands_matrix = bands_matrix[0:channel_num, :]
  center_freqs = center_freqs[0:channel_num]
  return bands_matrix, center_freqs


def _removeSilentFrames(x, y, dyn_range, frame_len, step):
  '''
  remove the frames whose energy is lower than dynamic range
  inputs:
    x:            clean signal
    y:            processed signal
    dyn_range:    dynamic range
    frame_len:    length of frames
    step:         step
  outputs:
    x_re:         clean signal without silent frames
    y_re:         processed signal without silent frames
  '''
  frames_start = np.arange(0,len(x)-frame_len,step)
  win = np.hanning(frame_len+2)[1:-1]
  mask = np.zeros(frames_start.shape)
  
  for frame_i in range(len(frames_start)):
    cur_index = frames_start[frame_i]
    mask[frame_i] = 20*np.log10(np.linalg.norm(
        x[cur_index:(cur_index+frame_len)]*win)/np.sqrt(frame_len))
  mask = mask>(np.max(mask)-dyn_range)
  
  x_re = np.zeros(x.shape)
  y_re = np.zeros(y.shape)
  count = 0
  cur_i1 = 0
  for frame_i in range(len(frames_start)):
    if mask[frame_i]:
      cur_i0 = frames_start[frame_i]
      cur_i1 = frames_start[count]
      x_re[cur_i1:(cur_i1+frame_len)] += x[cur_i0:(cur_i0+frame_len)]*win
      y_re[cur_i1:(cur_i1+frame_len)] += y[cur_i0:(cur_i0+frame_len)]*win
      count += 1
  x_re = x_re[:cur_i1+frame_len]
  y_re = y_re[:cur_i1+frame_len]
  
  return x_re, y_re
  
def _stdft(x, frame_len, step, fft_len):
  '''
  calc the stdft of input signal x
  inputs:
    x:           input signal
    frame_len:   frame length
    step:        step, overlap is frame_len - step
    fft_len:     length of fft
  outputs:
    x_stdft:     result of stdft
  '''
  x = x.flatten()
  frames_start = np.arange(0, len(x)-frame_len, step)
  x_stdft = np.zeros([len(frames_start), fft_len], dtype=np.complex)
  
  win = np.hanning(frame_len+2)[1:-1]
  
  for frame_i in range(len(frames_start)):
    cur_i = frames_start[frame_i]
    x_stdft[frame_i,:] = np.fft.fft(x[cur_i:cur_i+frame_len]*win,fft_len)

  return x_stdft


def _taa_corr(x,y):
  '''
  calc and return the corr between x and y
  '''
  xn = x - np.mean(x)
  xn = xn/np.sqrt(np.sum(xn**2))
  
  yn = y - np.mean(y)
  yn = yn/np.sqrt(np.sum(yn**2))
  
  return np.sum(xn*yn)
  


def calc_stoi(sig_clean, sig_proc, fs_org):
  '''
  calc stoi of the processed signal
  inputs:
    sig_clean:          clean signal
    sig_proc:           processed signal
    fs_org:             sampling frequency
  outpyts:
    stoi:               the calculated stoi
  '''
  if sig_clean.shape != sig_proc.shape:
    print('ERROR: calc_stoi, sig_clean and sig_proc must have the same shape.')
    return None
  
  x = sig_clean.flatten()
  y = sig_proc.flatten()
  
  fs = 10000
  frame_len = 256
  fft_len = 512
  step = frame_len // 2
  channel_num = 15    # num of 1/3 octave bands
  mid_freq_0 = 150    # center frequecy of first 1/3 octave band in Hz
  [bands_matrix, center_freqs] = _thirdoct(fs,fft_len,channel_num,mid_freq_0)
  conti_frame_num = 30# number of frames for intermediate intelligibility measure
  SDR_low_bound = -15 # lower SDR-bound
  dyn_range = 40      # speech dynamic range
  
  # resample signals if necessary
  if fs_org != fs:
    tar_len = int(np.round(len(x)/fs_org*fs))
    x = resample(x, tar_len)
    y = resample(y, tar_len)
  
  # remove silent frames
  [x, y] = _removeSilentFrames(x, y, dyn_range, frame_len, step)
  
  x_hat = _stdft(x, frame_len, step, fft_len)[:,0:fft_len//2+1].T
  y_hat = _stdft(y, frame_len, step, fft_len)[:,0:fft_len//2+1].T
  
  
#  print(np.sum(np.abs(x_hat),axis=0))
  
  frame_num = x_hat.shape[1]
  
  X = np.zeros([channel_num, frame_num])
  Y = np.zeros([channel_num, frame_num])
  
  for frame_i in range(frame_num):
    X[:,frame_i] = np.sqrt(np.dot(bands_matrix, np.abs(x_hat[:,frame_i])**2))
    Y[:,frame_i] = np.sqrt(np.dot(bands_matrix, np.abs(y_hat[:,frame_i])**2))
    
  d_interm = np.zeros([channel_num, frame_num-conti_frame_num+1])
  clip_thres = 10**(-1*SDR_low_bound/20)

  
  for frame_i in range(conti_frame_num-1,frame_num):
    X_seg = X[:,(frame_i-conti_frame_num+1):frame_i+1]
    Y_seg = Y[:,(frame_i-conti_frame_num+1):frame_i+1]
    alpha = np.sqrt(np.sum(X_seg**2,axis=1)/np.sum(Y_seg**2,axis=1))
    for ch_i in range(channel_num):
      Y_prime = np.min(np.vstack((Y_seg[ch_i,:]*alpha[ch_i],
                                  X_seg[ch_i,:]*(1+clip_thres))),axis=0)
      d_interm[ch_i, frame_i-conti_frame_num+1] = _taa_corr(X_seg[ch_i,:],
              Y_prime)
  return np.mean(d_interm)
  
def calc_pesq(matlab_eng, sig_clean, sig_proc, fs_org):
    '''
    calc stoi of the processed signal
    inputs:
    sig_clean:          clean signal
    sig_proc:           processed signal
    fs_org:             sampling frequency
    outpyts:
    stoi:               the calculated stoi
    '''
    if sig_clean.shape != sig_proc.shape:
        print('ERROR: calc_stoi, sig_clean and sig_proc must have the same shape.')
        return None
    pesq = matlab_eng.pesq(matlab_eng.cell2mat())   

  
if __name__=='__main__':
  from scipy.io import wavfile
  [fs, sig_0] = wavfile.read('sig_0.wav')
  [fs, sig_1] = wavfile.read('sig_1.wav')
  print(calc_stoi(sig_0/(2**15), sig_1/(2**15), fs))
