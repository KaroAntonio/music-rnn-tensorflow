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

	def get_seed(self):
		'''
		Return seed batch
		'''

	def show(self):
		'''
		Graph or otherwise represent Data in a way that could not be done in a string repr
		'''

	def prep_data(self):
		# Create Dats
		print('Loading Data...')
		self.train, self.test = self.load_data()

		# Validate Sequence
		min_size = self.params['n_steps']*self.params['batch_size']

		if len(self.train) < min_size or len(self.test) < min_size:
			err_str = 'Data Sequence must be of length at least n_steps*batch_size: '
			raise Exception(err_str + str(min_size))
			
		self.n_input = self.params['n_input'] = self.train.shape[1]
		self.params['n_batches'] = self.train.shape[0]//self.batch_size
		self.n_batches = self.params['n_batches']

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

		# Shuffle Data
		np.random.seed(0)
		x_data = np.random.permutation(x_data)
		np.random.seed(0)
		y_data = np.random.permutation(y_data)
		
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

