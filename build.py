import math
import numpy as np
from data_utils.data import Data


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
		self.n_batches = self.params['n_batches'] = 2
		self.n_input = self.params['n_input'] = 8

		wave = self.gen_wave_1()
		return wave, wave

	def gen_wave_2(self):
		# Vector to vector sine wav
		data = []
		num_datas = (self.n_batches*
					self.batch_size*
					self.n_steps*
					(self.n_input/2))
		for i in range(num_datas):
			wave = (math.sin(i/5.)+1)/2.
			data += [[wave] * self.n_input ]

		return np.array(data)

	def gen_wave_1(self):
			# Double intersecting sine wave
			data = []
			num_datas = (self.n_batches*
						self.batch_size*
						self.n_steps*
						(self.n_input/2))
			for i in range(num_datas):
				wave = int((math.sin(i/5.)+1)*self.n_input/2)
				hot_wave = np.array(self.to_one_hot(wave, self.n_input))
				data += [hot_wave[::-1] + hot_wave]

			return np.array(data)

def build():
	# Define Params
	params = {
			'learning_rate':0.001,
			'batch_size':20,
			'n_steps':20,
			'n_hidden':128,
			}

	# Set Data to your data class
	data = SineData(params)

	return data, params

if __name__ == "__main__":
    data,params = build()
    print(params)
    x,y = data.next_train()
