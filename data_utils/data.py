import numpy as np
import math

class Data:
	def __init__(self, params):
		self.params = params
		self.train_pointer = -1
		self.test_pointer = -1

		self.batch_size = params['batch_size']
		self.n_steps = params['n_steps']

		self.prep_data()
		
	def load_data(self):
		'''
		Return 2 numpy array of 1-d vectors 
			of size n_input
			with values in range [0,1]
			for training and testing
		return train, test
		'''
		pass

	def post_process(self, sequence):
		'''
		Process sequence
		'''
		pass

	def prep_data(self):
		# Create Dats
		print('Loading Data...')
		self.train, self.test = self.load_data()

		min_size = self.params['n_steps']*self.params['batch_size']
		if len(self.train) < min_size:
			raise Exception('Data Sequence must be of length at least n_steps*batch_size: ' + str(min_size))

		self.params['n_input'] = self.train.shape[1]
		self.params['n_batches'] = self.train.shape[0]//self.batch_size

		# Prep Btches
		self.train_x, self.train_y = self.prep_batches(self.train)
		self.test_x, self.test_y = self.prep_batches(self.test)

	def prep_batches(self, data):
		x_data = []
		y_data = []
		for i in range(self.n_steps,len(data)):
			x_data += [data[i-self.n_steps:i]]
			y_data += [data[i]]

		self.num_batches = len(x_data)//self.batch_size
		x_data = x_data[:self.num_batches * self.batch_size]
		y_data = y_data[:self.num_batches * self.batch_size]

		x = np.split(np.array(x_data),self.num_batches)
		y = np.split(np.array(y_data),self.num_batches)
		return x,y

	def decode_one_hot_seq(self, one_hot_seq):
		# Decodes a vector of onehots to a vector of indices
		return [oh.index(1) for oh in one_hot_seq]

	def zero_batch(self):
		# Return a batch of zeros
		x_shape = ((self.batch_size, self.n_steps, self.n_input))
		y_shape = ((self.batch_size, self.n_input))
		x = np.zeros(x_shape)
		y = np.zeros(y_shape)
		return x,y

	def to_one_hot(self, i, size):
		'''
		Return a one hot rep of i in a vec size=size
		'''
		one_hot = np.zeros(size)
		one_hot[i] = 1
		return np.array(one_hot)

	def hot_to_i(self, one_hot):
		'''
		Return the int version of one_hot
		'''
		return one_hot.index(1)
	
	def next_batch(self, x_batches,y_batches,ptr):
		return x_batches[ptr], y_batches[ptr]

	def next_train(self):
		#return x,y for next train batch
		self.train_pointer = (self.train_pointer+1)%self.num_batches
		return self.next_batch(self.train_x, self.train_y, self.train_pointer)

	def next_test(self):
		#return x,y for next test batch
		self.test_pointer = (self.test_pointer+1)%self.num_batches
		return self.next_batch(self.test_x, self.test_y, self.test_pointer)

class SineData(Data):
	def __init__(self, params):
		self.n_batches = params['n_batches'] = 2
		self.n_input = params['n_input'] = 8
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
		return self.gen_wave_1()
		
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
