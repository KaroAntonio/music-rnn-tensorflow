import math
import os
import numpy as np
from data_utils.data import Data
from data_utils.parse_files import *
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def build():
	# Define Params
	params = {
			'ckpt':None,
			'learning_rate':0.001,
			'batch_size':15,
			'n_steps':10,
			'n_hidden':1024,
			'train_steps':100000,
			'display_step':50,
			'save_step':100,
			'gen_steps':200,
			'data_dir':'test',
			'block_size':512,
			'clip_rate':3./4.
			}

	# Set Data to your data class
	data = WaveData(params)

	return data, params

class WaveData(Data):
	def __init__(self, params):
		Data.__init__(self,params)

	
	def post_process(self, sequence):
		# postprocess a sequence in some way

		sequence = np.array(sequence[self.params['n_steps']:])

		save_generated_example('generated.wav',np.array(sequence))
		mpimg.imsave('sequence.png',sequence)


	def get_seed(self):
		return self.next_train()

	def load_data(self):
		'''
		Return x, y sequences from which batches are sampled
		'''


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
	
	def show(self):
		#seq,_ = self.load_data()
		seq,_ = self.next_train()
		seq = seq.reshape(seq.shape[0]*seq.shape[1],seq.shape[2])

		fig = plt.figure(figsize=(10,10))
		sub = fig.add_subplot(111)
		sub.matshow(seq[:500],  cmap=plt.cm.gray)
		plt.show()

if __name__ == "__main__":
	data,params = build()
	print(params)
	x,y = data.next_train()
	data.show()
