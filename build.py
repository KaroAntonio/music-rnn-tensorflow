import math
import os
import numpy as np
from data_utils.data import Data
from data_utils.parse_files import *

class WaveData(Data):
	def __init__(self, params):
		Data.__init__(self,params)

	def post_process(self, sequence):
		# sequence
		# postprocess a sequence in some way

		# ex. convert each vec to binary and print
		for vec in sequence:
			n_hot = [int(e+0.5) for e in vec]
			print(n_hot)

	def load_data(self):
		# seq data is one giant sequence of data,
		# from which x and y are taken.
		# data sequence length must be at least n_steps*batch_size
		# Return train , test data

		# Convinience
		data_dir = self.params['data_dir']
		n_steps = self.params['n_steps']
		block_size = self.params['block_size']

		directory = os.path.join('data',data_dir,'src',"")
		out_file = os.path.join('data',data_dir,data_dir)

		# Load Wave as Numpy
		convert_wav_files_to_nptensor(directory, block_size, n_steps, out_file)

		# Sequence
		seq_chunks = np.load(os.path.join('data',data_dir,data_dir+'_x.npy'))
		seq_len = seq_chunks.shape[0]*seq_chunks.shape[1]
		sequence = seq_chunks.reshape(seq_len,1024)

		return sequence,sequence
	
def build():
	# Define Params
	params = {
			'learning_rate':0.001,
			'batch_size':15,
			'n_steps':20,
			'n_hidden':128,
			'train_steps':10000,
			'display_step':50,
			'save_step':100,
			'gen_steps':30,
			'data_dir':'test',
			'block_size':512
			}

	# Set Data to your data class
	data = WaveData(params)

	return data, params

if __name__ == "__main__":
    data,params = build()
    print(params)
    x,y = data.next_train()
