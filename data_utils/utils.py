import os
import collections
import cPickle
import numpy as np

from data_utils.parse_files import *

class WavLoader():
    def __init__(self, data_dir, batch_size, block_size, seq_length):
        self.data_dir = data_dir
        #The batch size specifies how many streams of data are processed in parallel at one time
        self.batch_size = batch_size
        #block size: the length of each fft slice (in s...?)
        self.block_size = block_size
        #sequence length specifies the length of each stream
        self.seq_length = seq_length
        
        #check if x,y,mean,var tensors exist
        x_flag = os.path.isfile(os.path.join('data',data_dir,data_dir+'_x.npy'))
        y_flag = os.path.isfile(os.path.join('data',data_dir,data_dir+'_y.npy'))
        m_flag = os.path.isfile(os.path.join('data',data_dir,data_dir+'_var.npy'))
        v_flag = os.path.isfile(os.path.join('data',data_dir,data_dir+'_mean.npy'))
        t_flag = os.path.isfile(os.path.join('data',data_dir,data_dir+'_tensor.npy'))
        if not (x_flag & y_flag & m_flag & v_flag & t_flag):
            #rm data/test/test_mean.npy; rm data/test/test_var.npy; rm data/test/test_x.npy; rm data/test/test_tensor.npy; rm data/test/test_y.npy
            print('Building Data Tensors')
            directory = os.path.join('data',data_dir,'src',"")
            out_file = os.path.join('data',data_dir,data_dir)
            convert_wav_files_to_nptensor(directory, block_size, seq_length, out_file)
        else:
            print('Data Tensors Found')
        
        self.x_data = np.load(os.path.join('data',data_dir,data_dir+'_x.npy'))
        self.y_data = np.load(os.path.join('data',data_dir,data_dir+'_y.npy'))
        
        self.freq_space = self.x_data.shape[2]
        self.create_batches()
        self.reset_batch_pointer()
        
    def create_batches(self):
        self.num_batches = self.x_data.size / (self.batch_size * self.seq_length)
        self.x_batches = np.split(self.x_data.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(self.y_data.reshape(self.batch_size, -1), self.num_batches, 1)
        
    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y
        
    def reset_batch_pointer(self):
        self.pointer = 0
            
        
            
